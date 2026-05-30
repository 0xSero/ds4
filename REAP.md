# DwarfStar REAP: running pruned DeepSeek V4 Flash

This fork (`0xSero/ds4`) extends [antirez/ds4](https://github.com/antirez/ds4)
("DwarfStar") so the same native engine can load and run **REAP-pruned**
DeepSeek V4 Flash checkpoints in addition to the stock Flash and PRO models.

REAP (Router-weighted Expert Activation Pruning) permanently removes the
least-used routed experts from a Mixture-of-Experts model. The remaining layout
is byte-for-byte a DeepSeek V4 Flash model with a smaller routed-expert count,
so DwarfStar can run it once the engine stops assuming exactly 256 experts.

Everything in the upstream `README.md` still applies. This document only covers
what is specific to the pruned variants.

## Model lineup

| Name | Routed experts | Base | Source checkpoint | DS4 GGUF |
|---|---:|---|---|---|
| **Flash Spark** | 160 (K160) | DeepSeek V4 Flash (256) | [0xSero/DeepSeek-V4-Flash-180B](https://huggingface.co/0xSero/DeepSeek-V4-Flash-180B) | [0xSero/DeepSeek-V4-Flash-Spark-GGUF](https://huggingface.co/0xSero/DeepSeek-V4-Flash-Spark-GGUF) |
| **Flash Spark Mini** | 144 (K144) | DeepSeek V4 Flash (256) | [0xSero/DeepSeek-V4-Flash-162B](https://huggingface.co/0xSero/DeepSeek-V4-Flash-162B) | [0xSero/DeepSeek-V4-Flash-Spark-Mini-GGUF](https://huggingface.co/0xSero/DeepSeek-V4-Flash-Spark-Mini-GGUF) |

Both checkpoints keep the full Flash architecture otherwise: 43 layers, 6 routed
experts used per token, 1 shared expert, the compressed/indexer attention stack,
and the 1M-token context window. Only the routed-expert pool is smaller.

> **Status:** these are **experimental** REAP checkpoints
> (`status: experimental_checkpoint_not_ready_for_production` in each
> `config.json`). They are useful for local evaluation, not a drop-in
> replacement for the official Flash weights. See the per-model HF cards for
> validation artifacts.

## What the fork changes

The pruned models differ from stock Flash in exactly one structural way — fewer
routed experts — so the changes are narrow and additive:

- **`ds4.c`** — `ds4_select_shape_from_metadata()` now accepts a Flash shape
  whose routed-expert count is anything in `1..256`, reported as
  `"DeepSeek V4 Flash REAP"`. Every other shape field must still match Flash.
- **`ds4_cuda.cu`** — the router-select kernels (`router_select_kernel`,
  `router_select_parallel_kernel`, `router_select_warp_topk_kernel`, and their
  batch entry points) take `n_expert` / `n_expert_used` / `expert_weight_scale`
  as runtime parameters instead of hardcoded `256` / `6` / `1.5`, with bounds
  checks on the logits/probs/selection buffers. A `moe_down_q4K_qwarp32_kernel`
  extends the Q4_K routed path to multi-token prefill.
- **`gguf-tools/deepseek4-quantize.c`** — reads `n_routed_experts` from the HF
  `config.json`, rewrites the `deepseek4.expert_count` GGUF metadata, and
  reshapes the routed-expert tensors (`ne[2]`), router gate
  (`ffn_gate_inp.weight`), and router bias (`exp_probs_b.bias`) to the pruned
  count. Runtime-critical DS4 tensors (indexer, compressors, hyper-connection,
  router gate, embeddings, output head) are protected from over-quantization.

These are the only engine edits; the rest of DwarfStar is upstream.

## Run a pruned model

Build the engine for your backend exactly as upstream (`make` for Metal,
`make cuda-spark` for a DGX Spark, `make cuda-generic` for other CUDA GPUs),
then download a prebuilt DS4 GGUF and run it:

```sh
# 180B / K160 — "Flash Spark"
./download_model.sh spark
./ds4 -p "Hello"
./ds4-server --ctx 100000

# 162B / K144 — "Flash Spark Mini"
./download_model.sh spark-mini
./ds4-server --ctx 100000
```

`download_model.sh` links `./ds4flash.gguf` to whichever model you fetched, so
the default `./ds4` / `./ds4-server` commands pick it up. Each GGUF ships a
`.sha256` sidecar in its HF repo if you want to verify the download.

Only the compact `Q2-REAP-ds4` profile is published today (≈81 GB class, sized
for the same 96/128 GB machines as stock Flash Q2). The mixed-precision
`Q4-Dynamic-REAP-ds4` profile is produced by the conversion pipeline below and
will be added to the same repos once validated.

## Build the GGUFs yourself

The DS4 GGUFs are quantized from the HF safetensors with the patched quantizer.
A metadata-only "template head" from an upstream Flash Q2 GGUF supplies the DS4
tensor layout; the quantizer then rewrites the expert count and restreams the
pruned expert weights. See the **"Generate 0xSero REAP GGUFs"** section of
[`gguf-tools/README.md`](gguf-tools/README.md) for the direct invocation.

Convenience wrappers (intended to run on the Spark / conversion host, paths
overridable via env vars) live in [`scripts/`](scripts/):

| Script | Purpose |
|---|---|
| `convert_reap_ggufs.sh` | Build the compact Q2 pair |
| `convert_reap_ggufs_q4_dynamic.sh` | Build the quality-biased Q4-Dynamic pair |
| `run_reap_gguf_release_pipeline.sh` | Wait for the Q2 job, build Q4-Dynamic, upload both GGUF repos |
| `upload_reap_ggufs.py` | Push GGUFs + model cards + checksums to Hugging Face |
| `run_reap_validation_suite.sh` | Context / API / tool-call / Terminal-Bench validation |
| `run_reap_terminal_bench.sh` | Terminal-Bench 2.0 harness against a served endpoint |
| `reap_context_probe.py` | Long-context correctness probe |
| `summarize_reap_validation.py` | Collapse a validation run into a summary |
| `upload_reap_validation_results.py` | Publish validation artifacts under `validation/<run-id>/` |

## Related

- **Spark deployment (vLLM):** [0xSero/deepseek-spark](https://github.com/0xSero/deepseek-spark)
- **REAP pruning + validation pipeline:** the `deepseek-flash-reap` toolkit that produced these checkpoints
- **Upstream engine:** [antirez/ds4](https://github.com/antirez/ds4) — all credit for DwarfStar itself

## Credits

DwarfStar is written by Salvatore Sanfilippo (antirez) and contributors; it
exists thanks to the path opened by `llama.cpp` and GGML. This fork only adds
pruned-expert support and the REAP conversion/validation tooling on top. Please
read the upstream `README.md` acknowledgements — they apply here in full.
