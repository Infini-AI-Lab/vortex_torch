"""Scoring-method contract for the offline recall harness.

The harness exists to answer ONE question: **under a token budget, how well does
an algorithm identify the real top-k attended tokens — in each attention head?**
The headline metric is per-head token recall@budget (mean over heads AND the
*worst* head, which usually gates quality).

A *method* is a Python module exposing one of:

    NAME = "my_method"
    def block_scores(ctx) -> torch.Tensor          # [num_blocks]  (request-level: one
                                                   #  shared page set for all heads, like vortex)
    # or, for a per-head method (its per-head ceiling):
    def block_scores_headwise(ctx) -> torch.Tensor # [Hkv, num_blocks]  (each kv-head picks
                                                   #  its own pages)

Higher = keep. The harness picks the top-`budget` blocks (shared or per-head),
then scores them against ground-truth full attention: **per-head token recall**,
block-mass recall, and an output proxy. Drop a new file here (or `--method
path.py`) to test an idea in seconds — no GPU engine boot.

Precomputed on the Ctx (so methods stay one-liners):
- request-level pooled-over-kv-heads: ``q_bar`` [D]; ``kmean``/``kmin``/``kmax``
  [num_blocks, D]; ``accum`` [num_blocks].
- per-kv-head: ``kmean_h``/``kmin_h``/``kmax_h`` [Hkv, num_blocks, D]; ``q`` [Hq, D].
"""

from dataclasses import dataclass
import torch


@dataclass
class Ctx:
    q: torch.Tensor        # [Hq, D]   decode query, all heads
    K: torch.Tensor        # [Hkv, S, D]
    V: torch.Tensor        # [Hkv, S, D]
    accum_pos: torch.Tensor  # [S]  accumulated attention (recent queries), per position
    scaling: float
    block_size: int
    G: int                 # Hq // Hkv

    def __post_init__(self):
        # K/V may be stored fp16 in the trace; all scoring math is float.
        self.q = self.q.float()
        self.K = self.K.float()
        self.V = self.V.float()
        self.accum_pos = self.accum_pos.float()
        self.Hq, self.D = self.q.shape
        self.Hkv, self.S, _ = self.K.shape
        self.num_blocks = (self.S + self.block_size - 1) // self.block_size
        self.q_bar = self.q.mean(0)                                   # [D]
        # pad to whole blocks, pool keys per block over positions then over kv-heads
        nb, bs = self.num_blocks, self.block_size
        pad = nb * bs - self.S
        Kp = self.K
        if pad:
            Kp = torch.cat([self.K, self.K.new_zeros(self.Hkv, pad, self.D)], dim=1)
        Kb = Kp.reshape(self.Hkv, nb, bs, self.D)                     # [Hkv, nb, bs, D]
        # mask padded positions for min/max correctness
        valid = torch.ones(nb, bs, dtype=torch.bool)
        if pad:
            valid.view(-1)[self.S:] = False
        big = torch.where(valid[None, :, :, None], Kb, Kb.new_full((), float("inf")))
        sml = torch.where(valid[None, :, :, None], Kb, Kb.new_full((), float("-inf")))
        self.kmean_h = Kb.mean(2)                                    # [Hkv, nb, D]
        self.kmin_h = big.amin(2)                                    # [Hkv, nb, D]
        self.kmax_h = sml.amax(2)                                    # [Hkv, nb, D]
        self.kmean = self.kmean_h.mean(0)                           # [nb, D]  (pooled)
        self.kmin = self.kmin_h.mean(0)                            # [nb, D]
        self.kmax = self.kmax_h.mean(0)                            # [nb, D]
        # accumulated attention per block
        ap = self.accum_pos
        if pad:
            ap = torch.cat([ap, ap.new_zeros(pad)])
        self.accum = ap.reshape(nb, bs).sum(1)                       # [nb]
