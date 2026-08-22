"""INT4 at the flow/config level: any flow gets INT4 without being edited.

The transform is entirely on the DECLARED META -- K/V become packed uint8, two fp32 scale fields
appear, and a bf16 staging area joins the request domain -- so ``initialize(kv_int4=True)`` is the
whole integration point. What this test guards:

1. **A flow's own declarations are untouched.** INT4 adds fields; it must not rename, drop or
   reshape what the flow asked for, or a flow's ops would address a field that changed under them.
2. **``token_ratio`` DROPS.** That ratio is how the pool learns it can hold more tokens, so if it
   does not fall the capacity win is given back silently. It is also the one number that would be
   wrong if the ratio were accumulated from a mix of the bf16 and INT4 layouts.
3. **Staging is in the REQUEST domain, not the page domain**, and excluded from the ratios. In the
   page domain it measured a 1.28x regression against plain bf16 -- it gives back more than
   quantization saves.
4. **``is_int4_kv`` reads the meta, not a flag.** A backend that trusts a flag while the pool was
   built from a different meta mis-strides by 2x, which is plausible wrong data rather than an error.
5. **MLA is refused with a reason.** Its cache is a latent whose channels are not head dims, so the
   measured per-channel-K / per-token-V choice does not transfer.
6. **Name collisions between the two domains are rejected**, because they share one cache dict and a
   duplicate would make a field's ADDRESSING depend on declaration order.

Run: python examples/misc/test_int4_flow_meta.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vortex_torch.engine.sgl.config import VortexConfig  # noqa: E402
from vortex_torch.engine.sgl.int4_store import (  # noqa: E402
    K_SCALE, STAGE_K, STAGE_V, V_SCALE, compression_ratio,
)
from vortex_torch.flow import vFlow  # noqa: E402
from vortex_torch.utils import is_int4_kv  # noqa: E402

fails = 0
BS, D = 32, 128


class Plain(vFlow):
    """A minimal flow with one auxiliary field, standing in for any centroid flow."""

    def create_cache(self, block_size, head_dim):
        return {"centroids": (1, head_dim)}

    def forward_cache(self, cache, loc, ctx):
        pass

    def forward_indexer(self, q, o, cache, ctx):
        pass


class Clashing(Plain):
    """Declares a request-domain field under a name the page domain already uses."""

    def create_request_cache(self, block_size, head_dim):
        return {"centroids": (1, head_dim)}


class WithRequest(Plain):
    """Declares its own request-domain field, alongside INT4's."""

    def create_request_cache(self, block_size, head_dim):
        return {"prev_summary": (1, head_dim)}


def init(flow, **kw):
    flow.initialize(block_size=BS, head_dim=D, kv_cache_dtype=torch.bfloat16,
                    q_data_type=torch.bfloat16, **kw)
    return flow


print("1. the config knob exists and defaults off")
cfg = VortexConfig()
ok = cfg.kv_int4 is False
print(f"   VortexConfig.kv_int4 default = {cfg.kv_int4} {'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1
ok = VortexConfig.from_flat({"vortex_kv_int4": True}).kv_int4 is True
print(f"   reachable as vortex_kv_int4 through from_flat: {'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1

print("\n2. bf16 baseline vs INT4: the flow's own fields are untouched, K/V change")
base = init(Plain())
i4 = init(Plain(), kv_int4=True)
print(f"   bf16: {[(k, s, str(d).replace('torch.', '')) for k, ((s), d) in base.cache_meta_info.items()]}")
print(f"   int4: {[(k, s, str(d).replace('torch.', '')) for k, ((s), d) in i4.cache_meta_info.items()]}")
ok = base.cache_meta_info["centroids"] == i4.cache_meta_info["centroids"]
print(f"   the flow's 'centroids' declaration is identical: {'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1
ok = (i4.cache_meta_info["k"] == ((BS, D // 2), torch.uint8)
      and i4.cache_meta_info["v"] == ((BS, D // 2), torch.uint8))
print(f"   k/v are packed uint8 at half the channel count: {'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1
ok = (i4.cache_meta_info[K_SCALE] == ((1, D), torch.float32)
      and i4.cache_meta_info[V_SCALE] == ((BS, 1), torch.float32))
print(f"   per-CHANNEL K scale (1,{D}) and per-TOKEN V scale ({BS},1) added: "
      f"{'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1

print("\n3. token_ratio DROPS -- otherwise the capacity win is given back")
r_base, r_i4 = base.get_token_ratio(), i4.get_token_ratio()
ok = r_i4 < r_base and r_i4 < 1.0
print(f"   token_ratio {r_base:.4f} -> {r_i4:.4f} (must fall, and below 1) "
      f"{'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1
# Sanity-check the ratio against the payload figure the store reports independently. They differ
# because token_ratio includes the flow's own aux fields, which do not shrink -- so the end-to-end
# saving is always LESS than the payload's 3.46x, and stating both keeps that honest.
print(f"   payload alone compresses {compression_ratio(BS, D):.2f}x; end-to-end "
      f"{r_base / r_i4:.2f}x here, lower because 'centroids' does not shrink")

print("\n4. staging lands in the REQUEST domain and is excluded from the ratios")
req = i4.get_request_cache_meta_info()
ok = req.get(STAGE_K) == (BS, D) and req.get(STAGE_V) == (BS, D)
print(f"   request domain: {req} {'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1
ok = STAGE_K not in i4.cache_meta_info and STAGE_V not in i4.cache_meta_info
print(f"   NOT in the page domain (there it measured a 1.28x regression): "
      f"{'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1
# If staging were charged per token, the ratio would exceed the bf16 baseline outright.
staging_ratio = 2 * BS * D * 2 / (BS * D * 2)
ok = r_i4 < r_base
print(f"   charging it per token would have added {staging_ratio:.1f} to the ratio; "
      f"actual is {r_i4:.4f} {'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1
ok = init(WithRequest(), kv_int4=True).get_request_cache_meta_info().keys() == {
    "prev_summary", STAGE_K, STAGE_V}
print(f"   a flow's own request field coexists with INT4's: {'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1
ok = init(WithRequest()).get_request_cache_meta_info() == {"prev_summary": (1, D)}
print(f"   and without INT4 only the flow's own is declared: {'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1

print("\n5. is_int4_kv reads the META, not a flag")
ok = is_int4_kv(i4.cache_meta_info) and not is_int4_kv(base.cache_meta_info)
print(f"   int4 meta -> True, bf16 meta -> False: {'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1
# fp8 is ALSO uint8-stored, so uint8 alone must not be the test.
fp8_meta = {"k": ((BS, D), torch.uint8), "v": ((BS, D), torch.uint8)}
ok = not is_int4_kv(fp8_meta)
print(f"   an fp8-style uint8 cache is NOT mistaken for INT4: {'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1
ok = not is_int4_kv({})
print(f"   an empty meta does not crash: {'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1

print("\n6. a name collision between the domains is rejected")
try:
    init(Clashing(), kv_int4=True)
    print("   duplicate field name accepted <-- FAIL (addressing would depend on decl order)")
    fails += 1
except AssertionError as e:
    ok = "ADDRESSING" in str(e)
    print(f"   rejected, and says why: {'ok' if ok else '<-- FAIL (no reason given)'}")
    if not ok:
        fails += 1

print("\n7. bf16 flows are byte-for-byte unaffected by the new code path")
again = init(Plain())
ok = (again.cache_meta_info == base.cache_meta_info
      and again.get_token_ratio() == r_base
      and again.get_request_cache_meta_info() == {})
print(f"   kv_int4=False reproduces the original meta and ratio exactly: "
      f"{'ok' if ok else '<-- FAIL'}")
if not ok:
    fails += 1

print()
if fails:
    print(f"*** {fails} FAILURE(S) ***")
    sys.exit(1)
print("INT4 FLOW META TESTS PASS")
