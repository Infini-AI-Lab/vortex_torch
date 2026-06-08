#!/usr/bin/env python
"""Model-soup merge of independently-trained compressor shards.

Two (or more) single-GPU processes trained on disjoint data shards from a
SHARED init (same --seed) produce checkpoints in the same loss basin; their
parameter-space average ("model soup") is a sound NCCL-free stand-in for
data-parallel training of this shallow, identity-warm-started compressor.

    python -m vortex_torch.compressor.merge \
        --in result/compressor/longrun_s0.pt result/compressor/longrun_s1.pt \
        --out result/compressor/longrun_merged.pt
"""
from __future__ import annotations

import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", nargs="+", required=True,
                    help="shard checkpoints to average (>=2).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in args.inp]
    sds = [c["state_dict"] for c in cks]
    keys = set(sds[0])
    assert all(set(s) == keys for s in sds), "shard checkpoints have mismatched keys"

    merged = {}
    for k in sds[0]:
        ts = [s[k].float() for s in sds]
        merged[k] = (sum(ts) / len(ts)).to(sds[0][k].dtype)

    out = {"state_dict": merged, "config": cks[0]["config"],
           "layer_ids": cks[0]["layer_ids"], "merged_from": args.inp}
    torch.save(out, args.out)
    # carry the config json alongside (same shape as the trainer's save_ckpt)
    import json
    with open(args.out + ".json", "w") as f:
        json.dump(cks[0]["config"], f, indent=2)
    print(f"[merge] averaged {len(sds)} shards -> {args.out} ({len(merged)} tensors)")


if __name__ == "__main__":
    main()
