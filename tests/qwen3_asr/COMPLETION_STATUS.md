# Qwen3-ASR ncnn Completion Status

This status table is the working definition for whether the Qwen3-ASR ncnn task
is meaningfully complete. A demo audio file running successfully is not enough.

| Layer | Current status | Local next step | VM / upstream next step |
| --- | --- | --- | --- |
| Audio preprocessing parity | C++ `--dump-mel-summary` and `--dump-mel-raw` are available. The summary records sample rate, mono conversion, shape, dtype, basic stats, first values, `n_fft`, hop, mel bins, frames, padding/truncation, and chunk overlap. | Keep using `*_mel.json` for every fixture run. | Add PyTorch mel summary with the same schema and compare C++ vs PyTorch. |
| Three Chinese fixture categories | Local macOS TTS fixtures exist for short Mandarin, long Mandarin, and Mandarin mixed with English/numbers/punctuation. `make_chinese_fixtures.sh` can regenerate them. | Replace or supplement with real Mandarin recordings when available. | Run the same fixtures through the original PyTorch Qwen3-ASR pipeline. |
| Normalized text parity | `evaluate_fixtures.py` now compares raw, cleaned, strict-normalized, and semantic-normalized text. Current local strict result: `zh_short_tts` passes; `zh_long_tts` and `zh_mixed_tts` fail. Semantic result: `zh_short_tts` and `zh_mixed_tts` pass, while `zh_long_tts` still fails. | Improve long-audio postprocess/chunk stitching only when it does not regress `pdx_voice`; keep strict and semantic pass/fail separate. | Compare PyTorch normalized text before deciding whether failures are ncnn conversion issues or expected model/pipeline behavior. |
| Module-level localization | ncnn result JSON records audio embedding, merged text/audio embedding, hidden state, full logits, and selected first-step logits summaries. Older experiments already compared exported modules. | Keep these summaries in every fixture run; only add more ncnn-side split points if the exported graph exposes a separable projector/adaptor. | Add PyTorch summaries for speech encoder, projector/adaptor, and first decoder logits; compare `max_abs`, `mean_abs`, `p99_abs`, cosine similarity, and top-k agreement. |
| Multi-platform smoke | macOS CPU-only build and smoke are available. Result JSON records threads, total time, RTF, chunk count, chunking strategy, and peak RSS for the three local Chinese fixtures. `collect_platform_smoke.py` generates a platform smoke report. | Keep macOS as correctness/output-contract smoke, not performance proof. | Add Linux and Windows build/smoke. For Vulkan/ARM/mobile, record a separate performance table and keep it separate from PyTorch accuracy parity. |
| Minimal regression command | `qwen3_asr_main` supports `--text-out`, `--json-out`, and `--dump-mel-summary`. `run_fixture.sh` wraps the command into a repeatable fixture runner. | Use `run_fixture.sh` for all local fixture regeneration. | Publish the same command shape in the final technical report so FunASR/Qwen3-ASR users can reproduce and report conversion or postprocess issues. |

## Current Local Summary

Run:

```bash
python3 tests/qwen3_asr/evaluate_fixtures.py \
  --fixtures tests/qwen3_asr/fixtures.local.json \
  --json-out /Users/link/llk/test_audio/chinese_fixtures/eval_report.json \
  --markdown-out /Users/link/llk/test_audio/chinese_fixtures/eval_report.md
```

Current local text result:

| fixture | strict | semantic | reason |
| --- | --- | --- | --- |
| `zh_short_tts` | PASS | PASS | Short Mandarin fixture matches. |
| `zh_long_tts` | FAIL | FAIL | Long-audio stitching reaches the tail now, but still has text differences such as `剪检查` and `对其`. |
| `zh_mixed_tts` | FAIL | PASS | Mixed English abbreviation differs under strict output contract, but documented semantic normalization maps it back to `OpenAI API`. |

Current macOS CPU-only smoke:

| fixture | chunks | RTF | peak RSS | chunking |
| --- | ---: | ---: | ---: | --- |
| `zh_short_tts` | 2 | 10.71 | 3485.5 MiB | fixed overlap, tail aligned |
| `zh_long_tts` | 9 | 11.36 | 3463.8 MiB | fixed overlap, tail aligned |
| `zh_mixed_tts` | 4 | 10.33 | 3448.0 MiB | fixed overlap, tail aligned |

## Principle

Do not mark the issue as complete until PyTorch and ncnn have comparable artifacts
for preprocessing, end-to-end normalized text, and module-level localization.
