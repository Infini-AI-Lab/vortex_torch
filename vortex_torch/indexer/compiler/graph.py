import torch
from typing import List, Dict, Set, DefaultDict, Tuple, Optional, Callable
from collections import defaultdict, deque
import os
from ..context import Context
from ...abs import vTensor, vOp
from ...utils import Schedule

# =====================================================================
# Data structures
# =====================================================================

class UnionFind:
    def __init__(self):
        self.parent: Dict[int, int] = {}

    def add(self, x: int):
        if x not in self.parent:
            self.parent[x] = x

    def find(self, x: int) -> int:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: int, b: int):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


class OpDAG:
    """
    A lightweight DAG over ops (nodes) with directed edges derived from
    tensor producer/consumer relationships.
    """
    def __init__(self):
        self.nodes: List[int] = []                                       # op ids in topo order
        self.successors: DefaultDict[int, Set[int]] = defaultdict(set)   # op -> consumer ops
        self.predecessors: DefaultDict[int, Set[int]] = defaultdict(set) # op -> producer ops

    def add_node(self, op_id: int):
        self.nodes.append(op_id)

    def add_edge(self, producer_op_id: int, consumer_op_id: int):
        if consumer_op_id not in self.successors[producer_op_id]:
            self.successors[producer_op_id].add(consumer_op_id)
            self.predecessors[consumer_op_id].add(producer_op_id)


class SubgraphDAG:
    """
    A view over the op-level DAG at the subgraph granularity.
    Used during fusion to check whether a merge would create a cycle.
    """
    def __init__(self, op_dag: OpDAG, uf: UnionFind):
        self._op_dag = op_dag
        self._uf = uf

    def _build_sg_successors(self) -> DefaultDict[int, Set[int]]:
        """Rebuild subgraph-level successor map from current UF state."""
        succ: DefaultDict[int, Set[int]] = defaultdict(set)
        for op_id in self._op_dag.nodes:
            sg = self._uf.find(op_id)
            for s in self._op_dag.successors[op_id]:
                s_sg = self._uf.find(s)
                if sg != s_sg:
                    succ[sg].add(s_sg)
        return succ

    def can_merge(self, sg_a: int, sg_b: int) -> bool:
        """One-directional cycle check for merging ``sg_a`` and ``sg_b``.

        Specifically: is there a path from ``sg_b`` to ``sg_a`` that does
        not use the direct ``sg_a -> sg_b`` edge? Such a back-path means
        that merging would create a cycle on the ``sg_a``-side.

        For a **bidirectional** safety check (required when fusing W ops
        that have no direct edge between them — e.g., sibling ops sharing
        an input), call :meth:`can_merge_bidir` instead.
        """
        succ = self._build_sg_successors()
        # Remove the direct edge for the check
        succ[sg_a].discard(sg_b)

        visited: Set[int] = {sg_b}
        queue = deque([sg_b])
        while queue:
            node = queue.popleft()
            for nxt in succ[node]:
                if nxt == sg_a:
                    return False  # back-path exists -> would create cycle
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return True  # safe

    def can_merge_bidir(self, sg_a: int, sg_b: int) -> bool:
        """Can ``sg_a`` and ``sg_b`` be fused with no direction assumed?

        A merge creates a cycle iff there is a **non-trivial** path
        between them in *either* direction — a path of length >= 2 that
        passes through at least one other subgraph. Direct edges
        (``a -> b`` or ``b -> a``) alone are always fine: after fusing
        they become within-subgraph dependencies handled by the op
        topological order.

        Note: this is NOT the same as ``can_merge(a, b) and can_merge(b, a)``.
        :meth:`can_merge` only strips the ``sg_a -> sg_b`` edge, which
        means one of the two calls still "sees" the direct edge and
        refuses adjacent W-pairs.
        """
        succ = self._build_sg_successors()

        def _reaches_via_other(src: int, dst: int) -> bool:
            # Step through successors of ``src`` that are NOT ``dst``
            # (filtering out any direct ``src -> dst`` edge), then BFS.
            visited: Set[int] = {src}
            queue: deque = deque()
            for nxt in succ[src]:
                if nxt != dst and nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
            while queue:
                node = queue.popleft()
                for nxt in succ[node]:
                    if nxt == dst:
                        return True
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
            return False

        return not (
            _reaches_via_other(sg_a, sg_b) or _reaches_via_other(sg_b, sg_a)
        )


class Graph:
    def __init__(
        self,
        tensor_list: List[vTensor],
        op_list: List[vOp],
        output_tensor_to_op_list: List[Optional[int]],
        op_to_input_tensor_list: List[List[int]],
        op_to_output_tensor_list: List[List[int]],
        input_tensor_ids: List[int],
        output_tensor_ids: List[int],
        global_input_tensor_ids: List[int],
        global_output_tensor_ids: List[int],
    ):
        self.tensor_list: List[vTensor] = tensor_list
        self.op_list: List[vOp] = op_list
        self.output_tensor_to_op_list: List[Optional[int]] = output_tensor_to_op_list
        self.op_to_input_tensor_list: List[List[int]] = op_to_input_tensor_list
        # List of output tensor ids per op. Single-output ops carry a
        # one-element list; multi-output ops (e.g. ``TopK`` returning
        # ``(block_table, seqlens)``) carry one entry per output. Codegens
        # that only need the first output should index with ``[op_id][0]``.
        self.op_to_output_tensor_list: List[List[int]] = op_to_output_tensor_list
        self.input_tensor_ids: List[int] = input_tensor_ids
        self.output_tensor_ids: List[int] = output_tensor_ids
        self.global_input_tensor_ids: List[int] = global_input_tensor_ids
        self.global_output_tensor_ids: List[int] = global_output_tensor_ids
        self.schedule = op_list[0].schedule if (op_list and op_list[0] is not None) else None
        self.forward: Callable = None

    def __repr__(self) -> str:
        return (
            f"Graph("
            f"num_tensors={len(self.tensor_list)}, "
            f"num_ops={len(self.op_list)}, "
            f"input_tensor_ids={self.input_tensor_ids}, "
            f"output_tensor_ids={self.output_tensor_ids}, "
            f"global_input_tensor_ids={self.global_input_tensor_ids}, "
            f"global_output_tensor_ids={self.global_output_tensor_ids})"
        )


# =====================================================================
# Helpers
# =====================================================================

def _as_tensor_id_list(x) -> List[int]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _build_local_graph(
    global_tensor_list: List[vTensor],
    global_op_list: List[vOp],
    global_output_tensor_to_op_list,
    global_op_to_input_tensor_list: List[List[int]],
    global_op_to_output_tensor_list,
    selected_global_op_ids: List[int],
    global_input_tensor_ids: List[int],
    global_output_tensor_ids: List[int],
) -> Graph:
    """
    Build a self-contained local Graph from a subset of global op ids.
    All structural ids inside the returned Graph are remapped to local ids.
    """
    global_op_to_output_tensor_ids: List[List[int]] = [
        _as_tensor_id_list(x) for x in global_op_to_output_tensor_list
    ]

    selected_global_op_set: Set[int] = set(selected_global_op_ids)
    selected_global_tensor_ids: Set[int] = set()

    for global_op_id in selected_global_op_ids:
        for tid in global_op_to_input_tensor_list[global_op_id]:
            selected_global_tensor_ids.add(tid)
        for tid in global_op_to_output_tensor_ids[global_op_id]:
            selected_global_tensor_ids.add(tid)

    sorted_global_tensor_ids: List[int] = sorted(selected_global_tensor_ids)

    global_op_id_to_local: Dict[int, int] = {
        gid: lid for lid, gid in enumerate(selected_global_op_ids)
    }
    global_tensor_id_to_local: Dict[int, int] = {
        gid: lid for lid, gid in enumerate(sorted_global_tensor_ids)
    }

    local_tensor_list = [global_tensor_list[gid] for gid in sorted_global_tensor_ids]
    local_op_list = [global_op_list[gid] for gid in selected_global_op_ids]

    local_output_tensor_to_op_list: List[Optional[int]] = []
    for gtid in sorted_global_tensor_ids:
        producer = global_output_tensor_to_op_list[gtid]
        if producer is None or producer not in selected_global_op_set:
            local_output_tensor_to_op_list.append(None)
        else:
            local_output_tensor_to_op_list.append(global_op_id_to_local[producer])

    local_op_to_input_tensor_list: List[List[int]] = [
        [global_tensor_id_to_local[tid] for tid in global_op_to_input_tensor_list[gid]]
        for gid in selected_global_op_ids
    ]

    # Multi-output ops are supported: each entry is the list of output
    # tensor_ids (1+) for that op. Codegens that only consume the first
    # output index with ``[op_id][0]``.
    local_op_to_output_tensor_list: List[List[int]] = []
    for gid in selected_global_op_ids:
        outs = global_op_to_output_tensor_ids[gid]
        if len(outs) < 1:
            raise RuntimeError(
                f"Graph.op_to_output_tensor_list: op {gid} has no outputs"
            )
        local_op_to_output_tensor_list.append(
            [global_tensor_id_to_local[t] for t in outs]
        )

    return Graph(
        tensor_list=local_tensor_list,
        op_list=local_op_list,
        output_tensor_to_op_list=local_output_tensor_to_op_list,
        op_to_input_tensor_list=local_op_to_input_tensor_list,
        op_to_output_tensor_list=local_op_to_output_tensor_list,
        input_tensor_ids=[global_tensor_id_to_local[t] for t in global_input_tensor_ids],
        output_tensor_ids=[global_tensor_id_to_local[t] for t in global_output_tensor_ids],
        global_input_tensor_ids=list(global_input_tensor_ids),
        global_output_tensor_ids=list(global_output_tensor_ids),
    )


# =====================================================================
# Phase 1: Convert tensor-based context into an op-level DAG
# =====================================================================

def _build_op_dag(
    op_list: List[vOp],
    output_tensor_to_op_list,
    op_to_input_tensor_list: List[List[int]],
    op_to_output_tensor_list,
    final_output_tensor_ids: List[int],
    side_effect_op_ids: List[int] = (),
) -> Tuple[OpDAG, Set[int]]:
    """
    Build an OpDAG containing only the ops reachable from any of the
    final output tensors. Nodes are stored in topological order
    (dependencies before dependents).

    A pipeline has three kinds of roots:
      * the attention output ``tensor_id == 1``,
      * tensors that are produced but have no in-graph consumer
        (orphan sinks),
      * side-effect ops such as ``Save`` that intentionally do not
        claim their target tensor as a producer — passed via
        ``side_effect_op_ids``.
    All three seed the reverse DFS so their dependencies survive.
    """
    visited: Set[int] = set()
    topo_order: List[int] = []

    def dfs(op_id: int):
        if op_id in visited:
            return
        visited.add(op_id)
        for tid in op_to_input_tensor_list[op_id]:
            producer = output_tensor_to_op_list[tid]
            if producer is not None:
                dfs(producer)
        topo_order.append(op_id)

    for tid in final_output_tensor_ids:
        producer = output_tensor_to_op_list[tid]
        if producer is None:
            # Fine: a cache-field target of a ``Save`` op has no producer
            # in ``output_tensor_to_op_list`` by design; the Save op is
            # seeded separately via ``side_effect_op_ids``.
            continue
        dfs(producer)

    for op_id in side_effect_op_ids:
        dfs(op_id)

    reachable: Set[int] = set(topo_order)

    dag = OpDAG()
    for op_id in topo_order:
        dag.add_node(op_id)

    for consumer_op_id in topo_order:
        for tid in op_to_input_tensor_list[consumer_op_id]:
            producer_op_id = output_tensor_to_op_list[tid]
            if producer_op_id is not None and producer_op_id in reachable:
                dag.add_edge(producer_op_id, consumer_op_id)

    return dag, reachable


# =====================================================================
# Phase 2: Fuse W-connected ops on the op DAG (cycle-safe)
# =====================================================================

def _fuse_w_ops(op_dag: OpDAG, op_list: List[vOp]) -> UnionFind:
    """
    Fuse any pair of W-scheduled ops whose merge does not introduce a
    cycle in the subgraph DAG.

    Unlike an edge-only fuser, we do **not** require a direct W→W
    producer/consumer edge between the two ops. Siblings that share an
    input (e.g. two reductions over the same KV field) can be fused into
    a single kernel as long as they don't also sit on opposite ends of a
    path through another subgraph.

    Returns the UnionFind representing the final subgraph assignments.
    """
    uf = UnionFind()
    for op_id in op_dag.nodes:
        uf.add(op_id)

    sg_dag = SubgraphDAG(op_dag, uf)

    def _w_subgraph_reps() -> List[int]:
        """Current unique W-subgraph representatives, in op-topo order."""
        seen_set: Set[int] = set()
        reps: List[int] = []
        for op_id in op_dag.nodes:
            if op_list[op_id].schedule != Schedule.W:
                continue
            rep = uf.find(op_id)
            if rep not in seen_set:
                seen_set.add(rep)
                reps.append(rep)
        return reps

    changed = True
    while changed:
        changed = False
        reps = _w_subgraph_reps()
        # O(N^2) pair scan per iteration; N shrinks by one on each
        # successful merge, so the loop terminates in <= N outer passes.
        for i in range(len(reps)):
            sg_a = uf.find(reps[i])
            for j in range(i + 1, len(reps)):
                sg_b = uf.find(reps[j])
                if sg_a == sg_b:
                    continue
                if sg_dag.can_merge_bidir(sg_a, sg_b):
                    uf.union(sg_a, sg_b)
                    changed = True
                    # ``reps`` is now stale — restart the outer sweep.
                    break
            if changed:
                break

    return uf


# =====================================================================
# Phase 3: Convert fused op groups back into Graph objects
# =====================================================================

def _build_all_graphs(
    op_dag: OpDAG,
    uf: UnionFind,
    reachable_ops: Set[int],
    tensor_list: List[vTensor],
    op_list: List[vOp],
    output_tensor_to_op_list,
    op_to_input_tensor_list: List[List[int]],
    op_to_output_tensor_list,
    tensor_to_consumers: DefaultDict[int, List[int]],
    final_output_tensor_ids: List[int],
) -> Tuple[Graph, List[Graph]]:
    """Convert the fused subgraph groups back into Graph objects."""

    op_to_output_tensor_ids: List[List[int]] = [
        _as_tensor_id_list(x) for x in op_to_output_tensor_list
    ]
    topo_op_ids = op_dag.nodes
    final_output_set: Set[int] = set(final_output_tensor_ids)

    # --- Full graph ---
    full_input_set: Set[int] = set()
    for op_id in topo_op_ids:
        for tid in op_to_input_tensor_list[op_id]:
            producer = output_tensor_to_op_list[tid]
            if producer is None or producer not in reachable_ops:
                full_input_set.add(tid)

    full_graph = _build_local_graph(
        global_tensor_list=tensor_list,
        global_op_list=op_list,
        global_output_tensor_to_op_list=output_tensor_to_op_list,
        global_op_to_input_tensor_list=op_to_input_tensor_list,
        global_op_to_output_tensor_list=op_to_output_tensor_list,
        selected_global_op_ids=topo_op_ids,
        global_input_tensor_ids=sorted(full_input_set),
        global_output_tensor_ids=sorted(final_output_set),
    )

    # --- Collect & topo-sort subgraphs ---
    sg_key_to_op_ids: DefaultDict[int, List[int]] = defaultdict(list)
    sg_key_order: List[int] = []
    seen: Set[int] = set()
    for op_id in topo_op_ids:
        sg_key = uf.find(op_id)
        if sg_key not in seen:
            seen.add(sg_key)
            sg_key_order.append(sg_key)
        sg_key_to_op_ids[sg_key].append(op_id)

    sg_key_to_id: Dict[int, int] = {k: i for i, k in enumerate(sg_key_order)}
    sg_successors: DefaultDict[int, Set[int]] = defaultdict(set)
    sg_indegree: Dict[int, int] = {i: 0 for i in range(len(sg_key_order))}

    for op_id in topo_op_ids:
        c_sg = sg_key_to_id[uf.find(op_id)]
        for pred in op_dag.predecessors[op_id]:
            p_sg = sg_key_to_id[uf.find(pred)]
            if p_sg != c_sg and c_sg not in sg_successors[p_sg]:
                sg_successors[p_sg].add(c_sg)
                sg_indegree[c_sg] += 1

    queue = deque([s for s, d in sg_indegree.items() if d == 0])
    topo_sg_ids: List[int] = []
    while queue:
        s = queue.popleft()
        topo_sg_ids.append(s)
        for nxt in sg_successors[s]:
            sg_indegree[nxt] -= 1
            if sg_indegree[nxt] == 0:
                queue.append(nxt)

    if len(topo_sg_ids) != len(sg_key_order):
        raise RuntimeError("Cycle detected in subgraph DAG")

    # Map ops to their final topo-sorted subgraph index
    op_to_subgraph_id: Dict[int, int] = {}
    for new_id, old_id in enumerate(topo_sg_ids):
        for op_id in sg_key_to_op_ids[sg_key_order[old_id]]:
            op_to_subgraph_id[op_id] = new_id

    # --- Build each subgraph ---
    subgraphs: List[Graph] = []
    for new_id, old_id in enumerate(topo_sg_ids):
        sg_op_ids = sg_key_to_op_ids[sg_key_order[old_id]]

        input_set: Set[int] = set()
        output_set: Set[int] = set()

        for op_id in sg_op_ids:
            for tid in op_to_input_tensor_list[op_id]:
                producer = output_tensor_to_op_list[tid]
                if producer is None:
                    input_set.add(tid)
                elif producer in op_to_subgraph_id and op_to_subgraph_id[producer] != new_id:
                    input_set.add(tid)

        for op_id in sg_op_ids:
            for out_tid in op_to_output_tensor_ids[op_id]:
                if out_tid in final_output_set:
                    output_set.add(out_tid)
                    continue
                for consumer in tensor_to_consumers.get(out_tid, []):
                    if consumer in op_to_subgraph_id and op_to_subgraph_id[consumer] != new_id:
                        output_set.add(out_tid)
                        break

        subgraphs.append(_build_local_graph(
            global_tensor_list=tensor_list,
            global_op_list=op_list,
            global_output_tensor_to_op_list=output_tensor_to_op_list,
            global_op_to_input_tensor_list=op_to_input_tensor_list,
            global_op_to_output_tensor_list=op_to_output_tensor_list,
            selected_global_op_ids=sg_op_ids,
            global_input_tensor_ids=sorted(input_set),
            global_output_tensor_ids=sorted(output_set),
        ))

    return full_graph, subgraphs


# =====================================================================
# Main entry point
# =====================================================================

def contruct_graph(ctx: Context) -> Tuple[Graph, List[Graph]]:

    tensor_list: List[vTensor] = ctx.tensor_list
    op_list: List[vOp] = ctx.op_list
    output_tensor_to_op_list = ctx.output_tensor_to_op_list
    op_to_input_tensor_list: List[List[int]] = ctx.op_to_input_tensor_list
    op_to_output_tensor_list = ctx.op_to_output_tensor_list

    # --- consumer lookup (used both to derive the sink set and to decide
    # which tensors cross subgraph boundaries in Phase 3) ---
    tensor_to_consumers: DefaultDict[int, List[int]] = defaultdict(list)
    for consumer_op_id, input_tids in enumerate(op_to_input_tensor_list):
        for tid in input_tids:
            tensor_to_consumers[tid].append(consumer_op_id)

    # Terminal tensors that must survive DCE. Always include the
    # attention output ``tensor_id == 1``; additionally include every
    # tensor that is produced but has no in-graph consumer (orphan
    # sinks); additionally include every cache-field target of a
    # ``Save`` (these don't appear as "orphans" because their producer
    # slot is intentionally left ``None`` to avoid a Load → Save cycle —
    # see ``indexer/save_load.py``).
    side_effect_op_ids: List[int] = list(getattr(ctx, "side_effect_op_ids", []))
    save_target_tids = {
        tid
        for op_id in side_effect_op_ids
        for tid in op_to_output_tensor_list[op_id]
    }
    final_output_tensor_ids: List[int] = sorted({
        1,
        *(
            tid for tid, producer in enumerate(output_tensor_to_op_list)
            if producer is not None and not tensor_to_consumers.get(tid)
        ),
        *save_target_tids,
    })

    # ---------------------------------------------------------------
    # Phase 1: Convert to op-level DAG (ops as nodes, tensor edges)
    # ---------------------------------------------------------------
    op_dag, reachable_ops = _build_op_dag(
        op_list=op_list,
        output_tensor_to_op_list=output_tensor_to_op_list,
        op_to_input_tensor_list=op_to_input_tensor_list,
        op_to_output_tensor_list=op_to_output_tensor_list,
        final_output_tensor_ids=final_output_tensor_ids,
        side_effect_op_ids=side_effect_op_ids,
    )

    # ---------------------------------------------------------------
    # Phase 2: Fuse W-connected ops (cycle-safe merges only)
    # ---------------------------------------------------------------
    # Ablation hook: leave every Schedule.W op in its own generated kernel.
    # Cross-subgraph tensors are then materialized by _build_all_graphs, giving
    # a true unfused execution while preserving operator semantics.
    if os.environ.get("VORTEX_DISABLE_FUSION", "0") == "1":
        uf = UnionFind()
        for op_id in op_dag.nodes:
            uf.add(op_id)
    else:
        uf = _fuse_w_ops(op_dag, op_list)

    # ---------------------------------------------------------------
    # Phase 3: Convert back to Graph objects (full + subgraphs)
    # ---------------------------------------------------------------
    full_graph, subgraphs = _build_all_graphs(
        op_dag=op_dag,
        uf=uf,
        reachable_ops=reachable_ops,
        tensor_list=tensor_list,
        op_list=op_list,
        output_tensor_to_op_list=output_tensor_to_op_list,
        op_to_input_tensor_list=op_to_input_tensor_list,
        op_to_output_tensor_list=op_to_output_tensor_list,
        tensor_to_consumers=tensor_to_consumers,
        final_output_tensor_ids=final_output_tensor_ids,
    )

    return full_graph, subgraphs
