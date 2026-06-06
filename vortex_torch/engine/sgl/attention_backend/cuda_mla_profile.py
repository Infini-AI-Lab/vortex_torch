from __future__ import annotations

"""
``cuda_mla_profile`` — a profiling twin of the :class:`VortexCudaMLABackend`.

Execution is **identical** to ``cuda_mla`` (same indexer, same hand-written CUDA
block-sparse MLA decode, same flashinfer prefill). On top of that, for **every
decoded token** it measures, per layer and per attention head, how good the
sparse KV selection actually is against the *dense* ground truth:

* **p-coverage** — the fraction of the full softmax mass that lands on the KV
  tokens the sparse method selected. ``sum_{t in S} softmax(q·k_t)`` where ``S``
  is the selected token set and the softmax is over **all** cached tokens. 1.0
  means the selection captured all the attention mass.
* **recall@N** — of the exact top-``N`` tokens by attention score (dense), how
  many fall inside the selected set ``S``. ``|topN ∩ S| / N``. ``N`` is a
  user-specified hyper-parameter (a comma list is allowed).

Both are accumulated as running means per ``(layer, head)`` and written to a JSON
report. The dense scores are recomputed in PyTorch (one ``[H,d]·[d,T]`` matmul
per request per layer) — this is **not** cuda-graph compatible and is much slower
than plain ``cuda_mla``; run with ``disable_cuda_graph=True`` and a small number
of prompts. It exists to *measure* a flow's selection quality, not to serve.

Configuration (env vars, read once at backend init):

* ``VORTEX_MLA_PROFILE_OUT``       — JSON report path (default ``mla_profile.json``).
* ``VORTEX_MLA_PROFILE_RECALL_N``  — comma list of N for recall@N (default ``16,64,128``).
* ``VORTEX_MLA_PROFILE_FLUSH``     — flush the report every K decode calls
  (default 200) so a killed worker still leaves recent stats.

Registered as the attention backend named ``cuda_mla_profile`` (see
``integration.py``); the MHA-prefill dispatch handler is self-registered on
import, mirroring ``cuda_mla``.
"""
import atexit
import json
import math
import os
from typing import TYPE_CHECKING

import torch

from .cuda_mla import VortexCudaMLABackend

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner


# ---------------------------------------------------------------------------- #
# Dispatch handler — same routing as cuda_mla (per-head MHA for every extend
# batch, absorbed MLA for decode). Self-registered on import so the engine can
# select attention_backend="cuda_mla_profile" with the correct prefill path.
# ---------------------------------------------------------------------------- #
def _register_cuda_mla_profile_dispatch() -> None:
    try:
        from sglang.srt.models.deepseek_common.attention_backend_handler import (
            AttentionBackendRegistry,
            _dispatch_mla_subtype,
        )
        from sglang.srt.compilation.piecewise_context_manager import (
            is_in_piecewise_cuda_graph,
        )
        from sglang.srt.server_args import get_global_server_args
        from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
            AttnForwardMethod,
        )
    except Exception:
        return

    def _handle_attention_cuda_mla_profile(attn, forward_batch):
        if is_in_piecewise_cuda_graph():
            return AttnForwardMethod.MLA
        if get_global_server_args().enable_deterministic_inference:
            return _dispatch_mla_subtype(attn, forward_batch)
        if forward_batch.forward_mode.is_extend_without_speculative():
            return AttnForwardMethod.MHA
        return _dispatch_mla_subtype(attn, forward_batch)

    AttentionBackendRegistry.register(
        "cuda_mla_profile", _handle_attention_cuda_mla_profile
    )


_register_cuda_mla_profile_dispatch()


class VortexCudaMLAProfileBackend(VortexCudaMLABackend):
    """``cuda_mla`` decode + per-token per-head coverage / recall@N profiling."""

    def __init__(self, model_runner: "ModelRunner", skip_prefill: bool = False):
        super().__init__(model_runner, skip_prefill=skip_prefill)

        self._prof_out = os.environ.get("VORTEX_MLA_PROFILE_OUT", "mla_profile.json")
        n_str = os.environ.get("VORTEX_MLA_PROFILE_RECALL_N", "16,64,128")
        self._prof_recall_N = sorted(
            {int(x) for x in n_str.replace(" ", "").split(",") if x}
        ) or [64]
        self._prof_max_N = max(self._prof_recall_N)
        self._prof_flush_every = int(os.environ.get("VORTEX_MLA_PROFILE_FLUSH", "200"))

        H = self.num_qo_heads
        dev = self.device
        # Running sums per layer_id; lazily created. p-coverage [H]; recall sums
        # [num_N, H]; count = #tokens (requests) seen for that layer.
        self._prof_pcov: dict[int, torch.Tensor] = {}
        self._prof_recall: dict[int, torch.Tensor] = {}
        self._prof_count: dict[int, int] = {}
        self._prof_H = H
        self._prof_dev = dev
        self._prof_calls = 0
        self._prof_meta = {
            "model": getattr(model_runner.model_config, "model_path", None)
            or getattr(getattr(model_runner, "server_args", None), "model_path", None),
            "module": getattr(model_runner.server_args, "vortex_module_name", None),
            "attention_backend": "cuda_mla_profile",
            "block_size": int(self.block_size),
            "topk_val": int(getattr(model_runner.server_args, "vortex_topk_val", 0)),
            "num_heads": int(H),
            "recall_N": list(self._prof_recall_N),
            "layers_skip": list(self.layers_skip),
        }
        atexit.register(self._prof_dump)

    # ------------------------------------------------------------------ #
    def _prof_slots(self, layer_id: int):
        if layer_id not in self._prof_count:
            H, dev = self._prof_H, self._prof_dev
            self._prof_pcov[layer_id] = torch.zeros(H, dtype=torch.float64, device=dev)
            self._prof_recall[layer_id] = torch.zeros(
                len(self._prof_recall_N), H, dtype=torch.float64, device=dev
            )
            self._prof_count[layer_id] = 0
        return self._prof_pcov[layer_id], self._prof_recall[layer_id]

    @torch.no_grad()
    def _prof_accumulate(self, q, layer, forward_batch):
        """Recompute the dense attention distribution for this decode step and
        accumulate p-coverage + recall@N for ``layer`` over the batch."""
        # Don't run inside a cuda-graph capture (profiling is dynamic python).
        if torch.cuda.is_current_stream_capturing():
            return

        H = self.num_qo_heads
        bs = q.shape[0] if q.dim() >= 1 else 0
        if bs == 0:
            return
        query = q.contiguous().view(bs, H, self.kv_cache_dim).float()  # [bs,H,576]

        md = self.ctx.metadata
        block_tables = md.sparse_block_tables          # [bs, max_blocks] page ids
        seqlens = md.sparse_seqlens                    # [bs] selected token count
        latent = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id).view(
            -1, self.kv_cache_dim
        )
        bsz = self.block_size
        arangeB = torch.arange(bsz, device=self._prof_dev)
        scaling = float(layer.scaling)

        seq_lens = forward_batch.seq_lens.to(torch.long)
        req_idx = forward_batch.req_pool_indices.to(torch.long)

        pcov_sum, recall_sum = self._prof_slots(layer.layer_id)
        Ns = self._prof_recall_N

        for r in range(bs):
            T = int(seq_lens[r].item())
            if T <= 0:
                continue
            slots = self.req_to_token[req_idx[r], :T].to(torch.long)   # global rows
            lat = latent.index_select(0, slots).float()                # [T,576]
            scr = (query[r] @ lat.transpose(0, 1)) * scaling           # [H,T]
            p = torch.softmax(scr, dim=-1)                             # [H,T]

            nb = int(math.ceil(int(seqlens[r].item()) / bsz))
            nb = max(nb, 0)
            if nb > 0:
                pages = block_tables[r, :nb].to(torch.long)            # [nb]
                sel_rows = (pages.unsqueeze(1) * bsz + arangeB).reshape(-1)  # global
                sel_mask = torch.isin(slots, sel_rows)                 # [T] bool
            else:
                sel_mask = torch.zeros(T, dtype=torch.bool, device=self._prof_dev)

            # p-coverage per head
            pcov_sum += (p * sel_mask).sum(dim=1).to(torch.float64)

            # recall@N per head
            maxN = min(self._prof_max_N, T)
            topi = p.topk(maxN, dim=-1).indices                        # [H, maxN]
            hit = sel_mask[topi].to(torch.float64)                     # [H, maxN]
            csum = hit.cumsum(dim=1)                                    # [H, maxN]
            for j, N in enumerate(Ns):
                n = min(N, maxN)
                recall_sum[j] += csum[:, n - 1] / n

            self._prof_count[layer.layer_id] += 1

        self._prof_calls += 1
        if self._prof_flush_every and self._prof_calls % self._prof_flush_every == 0:
            self._prof_dump()

    # ------------------------------------------------------------------ #
    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ):
        # Run the real cuda_mla decode (writes cache, fills md.sparse_block_tables
        # via the indexer, runs the CUDA kernel). md now reflects THIS layer.
        out = super().forward_decode(
            q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs,
        )
        # Skipped layers ran dense (no sparse selection) — nothing to profile.
        if layer.layer_id not in self.layers_skip:
            try:
                self._prof_accumulate(q, layer, forward_batch)
            except Exception as e:  # never let profiling break decode
                if not getattr(self, "_prof_warned", False):
                    print(f"[cuda_mla_profile] profiling disabled after error: {e}",
                          flush=True)
                    self._prof_warned = True
        return out

    # ------------------------------------------------------------------ #
    def _prof_dump(self) -> None:
        if not self._prof_count:
            return
        layers = {}
        all_pcov, all_recall = [], {N: [] for N in self._prof_recall_N}
        for lid in sorted(self._prof_count):
            c = self._prof_count[lid]
            if c == 0:
                continue
            pcov = (self._prof_pcov[lid] / c).detach().cpu().tolist()
            rec = (self._prof_recall[lid] / c).detach().cpu()           # [num_N, H]
            recall = {}
            for j, N in enumerate(self._prof_recall_N):
                per_head = rec[j].tolist()
                recall[str(N)] = {
                    "per_head": per_head,
                    "mean": float(sum(per_head) / len(per_head)),
                }
                all_recall[N].append(recall[str(N)]["mean"])
            layers[str(lid)] = {
                "count": c,
                "p_coverage_per_head": pcov,
                "p_coverage_mean": float(sum(pcov) / len(pcov)),
                "recall": recall,
            }
            all_pcov.append(layers[str(lid)]["p_coverage_mean"])
        report = {
            "meta": self._prof_meta,
            "overall": {
                "p_coverage_mean": (float(sum(all_pcov) / len(all_pcov))
                                    if all_pcov else None),
                "recall_mean": {str(N): (float(sum(v) / len(v)) if v else None)
                                for N, v in all_recall.items()},
                "tokens_profiled": int(sum(self._prof_count.values())),
            },
            "layers": layers,
        }
        tmp = self._prof_out + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            os.replace(tmp, self._prof_out)
        except OSError as e:
            print(f"[cuda_mla_profile] could not write {self._prof_out}: {e}",
                  flush=True)
