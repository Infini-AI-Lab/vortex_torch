# Sparse distillation with verl: vortex rollout + sparse-attention training

SFT a model on long chain-of-thought math (`nvidia/Nemotron-SFT-Math-v4`) with
**vortex sparse attention in the training forward/backward**, then score it on
**AIME24/AIME25** through **vortex_torch + sglang** (full-GPU KV). The question
being asked: does training under the same restricted context the model will be
*served* under recover the quality that sparse serving otherwise costs?

## verl is unmodified

The whole integration is one config line plus a registration:

```
verl/workers/config/model.py:185          AutoConfig.from_pretrained(attn_implementation=…)
verl/workers/engine/fsdp/transformer_impl.py:257   from_pretrained(config=hf_config)
```

transformers then dispatches every layer through `ALL_ATTENTION_FUNCTIONS`, so
setting `model.override_config.attn_implementation: vortex_sparse` is sufficient
— *provided the name is registered in the worker process before the model is
built*. `hookpath/sitecustomize.py` does that (python imports `sitecustomize`
automatically, before user code); `sparse_hook.py` reads the geometry from
`VORTEX_SPARSE_*`, which Ray/torchrun propagate.

Verified: the custom function is invoked **once per layer** (28/28 on Qwen3-0.6B).

## Run it

```bash
# 1. data: filter to the length window where sparsity actually engages
python -m application.verl_sparse_distill.prepare_data \
    --out-dir /scratch/zhuominc/data/nemotron_math \
    --model Qwen/Qwen3-4B --min-tokens 4096 --max-tokens 32768 --max-rows 20000

# 2. train (sparse), and the dense control that differs ONLY in attention
DATA=/scratch/zhuominc/data/nemotron_math NGPU=8 \
    bash application/verl_sparse_distill/run_sft.sh sparse
DATA=/scratch/zhuominc/data/nemotron_math NGPU=8 \
    bash application/verl_sparse_distill/run_sft.sh dense

# 3. score on AIME24/25 through vortex_torch + sglang (full-GPU KV)
python -m application.verl_sparse_distill.evaluate_aime \
    --ckpt /scratch/zhuominc/ckpt_verl_sparse/global_step_200 \
    --tasks aime24 aime25 --trials 16
```

## The two things that will bite you

**1. verl's `MultiTurnSFTDataset` deletes the reasoning trace.** It templates each
turn separately and concatenates ids; Qwen3 only emits `<think>` for a whole
conversation and strips it from a single turn's `content`. Measured: **13 094
tokens whole-conversation vs 1 166 per-turn, `<think>` absent**. verl asserts —
and `ignore_input_ids_mismatch=True` "fixes" it by training on ~9% of the tokens
with the CoT gone, which produces a believable loss curve for an experiment that
is not happening. `dataset.py` (`ReasoningSFTDataset`, wired via verl's
`data.custom_cls`) templates once and masks the prompt instead. The symptom is a
worker abort naming Arrow, not the template:
`Check failed: (off) <= (length) Slice offset (84) > length (63)`; index the
dataset single-process to see the real assertion.

**2. The budget conventions differ between the two projects.**

    vortex_torch:  selected = topk_val + reserved_bos + reserved_eos   (additive)
    vortex_train:  Budget.topk is the TOTAL, reservations included

Training a total of 18 blocks must be served with `topk_val=16, bos=1, eos=1`,
**not** `topk_val=18`. `patch.install` converts on the training side and
`evaluate_aime.write_selection` converts back, emitting the
`vortex_selection.json` that `evaluate.py` treats as the source of truth — so the
checkpoint carries its own budget and a later evaluation cannot silently use a
different one.

## Verified

| check | result |
|---|---|
| custom `attn_implementation` reaches every layer | 28/28 layers |
| sparse path engages above the budget | T=4096 → 28 sparse; T=512 → 28 dense (`seqlen < 1152`) |
| sparse fwd + bwd under FSDP | grad_norm finite, no NaN |
| `ReasoningSFTDataset` preserves CoT | 62/62 rows: lengths match prep, `<think>` present, prompt masked |
| verl SFT end to end | 3 optimizer steps, `val/loss 1.011`, peak 49.7 GB |
