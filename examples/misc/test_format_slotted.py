"""FORMAT.SLOTTED: a request-bound cache domain as a compiler FORMAT.

vortex has one cache domain: ``create_cache`` fields sized ``num_blocks x r x c`` and addressed by
block id, bound to sglang's page allocator. That is right for state summarising *stored* KV
(centroids, envelopes) -- it lives exactly as long as the page. It is wrong for state belonging to a
*request in flight*, whose motivating case is INT4's bf16 staging area for blocks whose scale is not
yet computable. Before SLOTTED that was special-cased across ``memory_pool.py`` and the flashinfer
backend.

The claim being tested is that this is a **format, not a set of kernels**: a SLOTTED field is an
ordinary compiler operand, so one fused kernel reads and reduces BOTH domains in a single compiled
path with no extra launch.

Most of this test reads the GENERATED SOURCE rather than only running it, and that is deliberate:
if both domains silently received identical PAGED addressing, every numeric result would still look
plausible while each request-domain access hit the wrong row. The source is where the two domains
are distinguishable.

Run: python examples/misc/test_format_slotted.py
"""

import os
import re
import sys
import uuid

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vortex_torch.abs import FORMAT, vTensor  # noqa: E402
from vortex_torch.cache import Mean  # noqa: E402
from vortex_torch.cache.compiler.compile import compile as compile_cache  # noqa: E402
from vortex_torch.cache.compiler.triton_impl.kernel_gen import (  # noqa: E402
    SLOT_MAP_ARG, SLOT_VAR,
)
from vortex_torch.cache.context import Context as CacheContext  # noqa: E402

fails = 0
NB, N_SLOTS, BT, D = 64, 8, 32, 128
PAGE, BLOCK, NKV = 32, 32, 2
CACHE_DIR = os.path.join("/tmp", f"vortex_slotted_{uuid.uuid4().hex[:8]}")


def build(slotted: bool):
    """Trace one flow and compile it; return (generated source, module path).

    The flow is ``Mean(dim=1)`` over K into a per-channel envelope -- the shape INT4's K scale
    uses. With ``slotted=True`` the INPUT is the request domain (the staged bf16 block) while the
    OUTPUT stays page-bound, which is the mixed-domain case: one kernel, both addressings.
    """
    ctx = CacheContext()
    ctx.page_size = PAGE
    ctx.block_size = BLOCK
    ctx.num_blocks_per_page = PAGE // BLOCK
    ctx.total_num_pages = NB
    ctx.total_num_blocks = NB
    ctx.head_dim = D
    ctx.head_num = NKV
    ctx.max_new_tokens_per_batch = 64
    ctx.num_sms = 108
    ctx.vortex_dtype = torch.bfloat16
    ctx.sparse_attention_name = f"slotted_{'on' if slotted else 'off'}_{uuid.uuid4().hex[:6]}"
    ctx.impl_backend = "triton"
    ctx.compilation_cache_dir = CACHE_DIR
    ctx.tensor_list = []
    ctx.op_list = []
    ctx.output_tensor_to_op_list = []
    ctx.op_to_input_tensor_list = []
    ctx.op_to_output_tensor_list = []
    ctx.tensor_id_to_tensor_name_map = {}
    ctx.compilation_header_lines = []
    ctx.auxilary_func_def_lines = []
    ctx._aux_total_bytes = 0
    ctx._aux_total_flops = 0
    object.__setattr__(ctx, "_created", True)

    def add(name, shape, fmt, dtype=torch.bfloat16):
        t = vTensor(shape=shape, dtype=dtype, device="cuda",
                    _format=fmt, tensor_id=len(ctx.tensor_list))
        ctx.tensor_list.append(t)
        ctx.output_tensor_to_op_list.append(None)
        ctx.tensor_id_to_tensor_name_map[t.tensor_id] = name
        return t

    src_fmt = FORMAT.SLOTTED if slotted else FORMAT.PAGED
    n_lead = N_SLOTS if slotted else NB
    stage = add("cache['stage_k']", (n_lead, BT, D), src_fmt)
    env = add("cache['env']", (NB, 1, D), FORMAT.PAGED)

    # A real (tiny) loc, same as verify.py's compile path uses -- profile asserts on its type.
    Mean(dim=1)(stage, env, loc=torch.zeros((1,), dtype=torch.int32, device="cpu"), ctx=ctx)
    cls = compile_cache(ctx)
    src_path = os.path.join(CACHE_DIR, f"{ctx.sparse_attention_name}_compiled_func.py")
    return ctx, cls, open(src_path).read()


print(f"1. a single-domain flow is UNCHANGED (no {SLOT_MAP_ARG} threaded anywhere)")
ctx_p, _cls_p, src_p = build(slotted=False)
ok = SLOT_MAP_ARG not in src_p
print(f"   plain PAGED flow mentions {SLOT_MAP_ARG}: {SLOT_MAP_ARG in src_p} "
      f"{'ok' if ok else '<-- FAIL: every existing flow just changed signature'}")
if not ok:
    fails += 1
ok = "block_id * 4096" in src_p
print(f"   and addresses its input by block_id directly: {'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1

print("\n2. the SLOTTED flow emits an INDIRECTION, not the same PAGED addressing")
ctx_s, cls_s, src_s = build(slotted=True)
checks = [
    (f"tl.load({SLOT_MAP_ARG} + block_id)", "loads slot = map[block_id] once"),
    (f"if {SLOT_VAR} < 0:", "drops the whole program on a miss"),
    (f"tl.maximum({SLOT_VAR}, 0)", "clamps as a second line of defence"),
    ("tensor_1_off = block_id * 128", "the PAGED output still uses block_id"),
]
for needle, why in checks:
    ok = needle in src_s
    print(f"   {why:<44s} {'ok' if ok else '<-- FAIL (missing: ' + needle + ')'}")
    if not ok:
        fails += 1
# The point of the whole exercise: the two domains are in ONE kernel.
n_kernels = len(re.findall(r"@triton\.jit", src_s))
ok = n_kernels == 1
print(f"   both domains in ONE kernel: {n_kernels} @triton.jit found "
      f"{'ok' if ok else '<-- FAIL: fused path was split'}")
if not ok:
    fails += 1

print(f"\n3. {SLOT_MAP_ARG} reaches the kernel through every layer of the call chain")
for pat, where in ((rf"def \w+_kernel\(\s*(?:[^)]*?)\b{SLOT_MAP_ARG}\b", "kernel signature"),
                   (rf"def \w+_impl\([^)]*{SLOT_MAP_ARG}=None", "impl wrapper"),
                   (rf"def \w+_interface\([^)]*{SLOT_MAP_ARG}=None", "subgraph interface"),
                   (rf"def forward\(self[^)]*{SLOT_MAP_ARG}=None", "forward() entry point")):
    ok = re.search(pat, src_s, re.S) is not None
    print(f"   {where:<22s} {'ok' if ok else '<-- FAIL: arg dropped here'}")
    if not ok:
        fails += 1
# The map is a property of the DOMAIN, so exactly one argument -- not one per SLOTTED tensor.
n_params = len(re.findall(rf"{SLOT_MAP_ARG}=None", src_s))
print(f"   declared once per function, not per tensor: {n_params} '=None' params")

print("\n4. it RUNS, and the numbers follow the map")
fn = cls_s()

torch.manual_seed(0)
stage = (torch.randn(N_SLOTS, BT, D, device="cuda") * 0.5).to(torch.bfloat16)
env = torch.zeros(NB, 1, D, dtype=torch.bfloat16, device="cuda")
slot_of = torch.full((NB,), -1, dtype=torch.int32, device="cuda")
# Deliberately NOT the identity: block b -> slot (b*3+1)%N_SLOTS. An implementation that ignored
# the map, or used block_id directly, would agree with the reference under an identity map.
resident = {}
for b in range(0, 12):
    resident[b] = (b * 3 + 1) % N_SLOTS
    slot_of[b] = resident[b]

# loc names the block-END tokens, which is what the cache kernel triggers on.
loc = torch.tensor([(b + 1) * BLOCK - 1 for b in range(0, 12, NKV)],
                   dtype=torch.int64, device="cuda")
fn.forward({"stage_k": stage, "env": env}, loc, ctx_s, slot_of_ptr=slot_of)
torch.cuda.synchronize()

# Which (block -> slot) pairs the kernel should have visited, derived the same way the kernel does.
bad = 0
visited = 0
for ti in range(loc.numel()):
    pos = int(loc[ti])
    for head in range(NKV):
        page_id = (pos // PAGE) * NKV + head
        blk = page_id * (PAGE // BLOCK) + (pos % PAGE) // BLOCK
        if blk >= NB:
            continue
        slot = int(slot_of[blk])
        if slot < 0:
            # A miss must leave the output untouched, not write zeros from a clamped row 0.
            if not bool((env[blk] == 0).all()):
                bad += 1
            continue
        visited += 1
        want = stage[slot].float().mean(dim=0, keepdim=True).to(torch.bfloat16)
        if not torch.allclose(env[blk].float(), want.float(), rtol=1e-2, atol=1e-2):
            bad += 1
ok = bad == 0 and visited > 0
print(f"   visited {visited} (block -> slot) pairs under a NON-identity map, mismatched={bad} "
      f"{'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1
    if visited == 0:
        print("      nothing was visited -- the trigger condition never fired, so this "
              "proved nothing")

print("\n5. a non-resident block is NOT written (the clamp must not be the only guard)")
env2 = torch.full((NB, 1, D), 7.0, dtype=torch.bfloat16, device="cuda")
miss_of = torch.full((NB,), -1, dtype=torch.int32, device="cuda")   # nothing resident
fn.forward({"stage_k": stage, "env": env2}, loc, ctx_s, slot_of_ptr=miss_of)
torch.cuda.synchronize()
ok = bool((env2 == 7.0).all())
print(f"   all-miss map leaves the page-domain output untouched: {'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1

print("\n6. forgetting the map is an ERROR, not row 0")
try:
    fn.forward({"stage_k": stage, "env": env}, loc, ctx_s)
    torch.cuda.synchronize()
    print("   omitting the map silently ran <-- FAIL")
    fails += 1
except AssertionError as e:
    print(f"   omitting the map asserts: ok ({str(e)[:60]}...)")

print()
if fails:
    print(f"*** {fails} FAILURE(S) ***")
    sys.exit(1)
print("FORMAT.SLOTTED TESTS PASS")
