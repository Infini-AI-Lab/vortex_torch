"""H2O heavy-hitter: rank blocks by accumulated attention from recent queries
(history-based; does not see the current query's target)."""
NAME = "h2o"
def block_scores(ctx):
    return ctx.accum.clone()              # [nb]
