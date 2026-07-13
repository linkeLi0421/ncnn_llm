# Qwen3-ASR ncnn Completion Status

This status table is the working definition for whether the Qwen3-ASR ncnn task
is meaningfully complete. A demo audio file running successfully is not enough.

| Layer | Current status | Local next step | VM / upstream next step |
| --- | --- | --- | --- |
| Audio preprocessing parity | C++ `--dump-mel-summary` and `--dump-mel-raw` are available. The summary records sample rate, mono conversion, shape, dtype, basic stats, first values, `n_fft`, hop, mel bins, frames, padding/truncation, and chunk overlap. | Keep using `*_mel.json` for every fixture run. | Add PyTorch mel summary with the same schema and compare C++ vs PyTorch. |
| Three Chinese fixture categories | Local macOS TTS fixtures exist for short Mandarin, long Mandarin, and Mandarin mixed with English/numbers/punctuation. `make_chinese_fixtures.sh` can regenerate them. | Replace or supplement with real Mandarin recordings when available. | Run the same fixtures through the original PyTorch Qwen3-ASR pipeline. |
| Normalized text parity | `evaluate_fixtures.py` now compares raw, cleaned, and normalized text. Current local result: `zh_short_tts` passes; `zh_long_tts` and `zh_mixed_tts` fail, which exposes long-audio stitching and mixed English/abbreviation issues. | Improve postprocess/chunk stitching only when it does not regress `pdx_voice`. | Compare PyTorch normalized text before deciding whether failures are ncnn conversion issues or expected model/pipeline behavior. |
| Module-level localization | ncnn result JSON records audio embedding, merged text/audio embedding, hidden state, full logits, and selected first-step logits summaries. Older experiments already compared exported modules. | Keep these summaries in every fixture run; only add more ncnn-side split points if the exported graph exposes a separable projector/adaptor. | Add PyTorch summaries for speech encoder, projector/adaptor, and first decoder logits; compare `max_abs`, `mean_abs`, `p99_abs`, cosine similarity, and top-k agreement. |
| Multi-platform smoke | macOS CPU-only build and smoke are available. Result JSON records threads, total time, RTF, chunk count, and chunking strategy. `collect_platform_smoke.py` generates a platform smoke report. | Keep macOS as correctness/output-contract smoke, not performance proof. | Add Linux and Windows build/smoke. For Vulkan/ARM/mobile, record a separate performance table and keep it separate from PyTorch accuracy parity. |
| Minimal regression command | `qwen3_asr_main` supports `--text-out`, `--json-out`, and `--dump-mel-summary`. `run_fixture.sh` wraps the command into a repeatable fixture runner. | Use `run_fixture.sh` for all local fixture regeneration. | Publish the same command shape in the final technical report so FunASR/Qwen3-ASR users can reproduce and report conversion or postprocess issues. |

## Current Local Summary

Run:

```bash
python3 tests/qwen3_asr/evaluate_fixtures.py \
  --fixtures tests/qwen3_asr/fixtures.local.json \
  --json-out /Users/link/llk/test_audio/chinese_fixtures/eval_report.json \
  --markdown-out /Users/link/llk/test_audio/chinese_fixtures/eval_report.md
```

Current local normalized-text result:

| fixture | status | reason |
| --- | --- | --- |
| `zh_short_tts` | PASS | Short Mandarin fixture matches after normalized comparison. |
| `zh_long_tts` | FAIL | Long-audio stitching reaches the tail now, but still has text differences such as `剪检查` and `对其`. |
| `zh_mixed_tts` | FAIL | Mixed English abbreviation output differs: `OpenAI API` normalizes to a different form in current ncnn output. |

## Principle

Do not mark the issue as complete until PyTorch and ncnn have comparable artifacts
for preprocessing, end-to-end normalized text, and module-level localization.
