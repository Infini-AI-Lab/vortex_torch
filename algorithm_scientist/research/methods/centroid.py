"""Block centroid · query (the vortex `block_sparse_attention` baseline)."""
NAME = "centroid"
def block_scores(ctx):
    return ctx.kmean @ ctx.q_bar          # [nb]
