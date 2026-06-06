"""StreamingLLM: attention sink (block 0) + most-recent window. Positional,
ignores the query."""
import torch
NAME = "streaming"
def block_scores(ctx):
    nb = ctx.num_blocks
    s = torch.arange(nb, dtype=torch.float32)   # recency: later blocks score higher
    s[0] = nb + 10.0                            # force the sink block
    return s
