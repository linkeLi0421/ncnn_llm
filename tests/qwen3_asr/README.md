# Qwen3-ASR Fixture Evaluation

This directory defines the local evaluation contract for Qwen3-ASR ncnn parity.
It follows the validation split suggested from the FunASR / Qwen3-ASR side:

1. Fix and summarize audio preprocessing.
2. Compare end-to-end text on multiple fixture categories.
3. Keep module-level summaries for localization.
4. Keep platform smoke/performance separate from PyTorch accuracy parity.
5. Provide a minimal regression entry point.

See `COMPLETION_STATUS.md` for the current completion table and remaining gaps.

## Local Fixtures

`fixtures.local.json` currently covers three local macOS TTS fixtures:

| id | category |
| --- | --- |
| `zh_short_tts` | short Mandarin |
| `zh_long_tts` | long Mandarin |
| `zh_mixed_tts` | Mandarin mixed with English, numbers, punctuation |

The fixture schema has nullable PyTorch fields. That is intentional: the local
Mac can validate the ncnn output contract, while PyTorch parity will be filled
in later on the Linux VM.

To regenerate the local TTS fixtures on macOS:

```bash
tests/qwen3_asr/make_chinese_fixtures.sh \
  --out-dir /Users/link/llk/test_audio/chinese_fixtures
```

## ncnn Regression Entry

Use `run_fixture.sh` so each run produces the same three artifacts:

- final text;
- structured result JSON;
- mel summary JSON.
- optional module raw `.f32` tensors for numeric parity.

Example:

```bash
tests/qwen3_asr/run_fixture.sh \
  --binary /Users/link/llk/build/ncnn_llm-macos-qwen3-asr/qwen3_asr_main \
  --model /Users/link/llk/models/qwen3_asr_0_6b_runtime_text128 \
  --audio /Users/link/llk/test_audio/chinese_fixtures/zh_short_16k.wav \
  --out-dir /Users/link/llk/test_audio/chinese_fixtures \
  --id zh_short_default_final \
  --threads 6 \
  --frames 1878 \
  --max-new-tokens 64 \
  --dump-module-raw \
  --measure-memory
```

## Evaluation

```bash
python3 tests/qwen3_asr/evaluate_fixtures.py \
  --fixtures tests/qwen3_asr/fixtures.local.json \
  --json-out /Users/link/llk/test_audio/chinese_fixtures/eval_report.json \
  --markdown-out /Users/link/llk/test_audio/chinese_fixtures/eval_report.md
```

The script compares:

- raw text;
- cleaned text;
- normalized text;
- semantic-normalized text for documented abbreviation variants;
- ncnn mel summary presence and preprocessing fields;
- optional PyTorch mel summary when available;
- optional module-level summary when available.
- optional raw tensor metrics when both sides provide `raw_path`.

Strict pass/fail is based on normalized text. Semantic pass/fail is reported
separately and is intentionally limited to documented abbreviation variants such
as `Open I A P P.I` versus `OpenAI API`. Raw and cleaned text are kept in the
JSON report so postprocess problems are not confused with model parity.

The ncnn result JSON also carries module-localization summaries for each chunk:

- `audio_embedding`;
- `first_step.text_embeds`;
- `first_step.merged_embeds`;
- `first_step.hidden`;
- `first_step.logits`;
- `first_step.selected_logits`.

`run_pytorch_fixtures.py` can now generate matching PyTorch-side summaries for
the first fixed chunk by default:

- speech encoder / audio embedding output;
- merged text/audio embedding;
- first decoder hidden/logits;
- selected first-step logits and greedy next token.

These summaries intentionally use the same fixed/static audio path as the ncnn
runner for module localization, not necessarily the dynamic feature mask used by
the end-to-end PyTorch transcribe call. Pass `--dump-module-raw` on both the
ncnn and PyTorch runners to write comparable `.f32` tensors for numeric parity.

## PyTorch Baseline

On the Linux/PyTorch environment, run the original Qwen3-ASR path and fill the
nullable PyTorch fields in a generated fixture file:

```bash
tests/qwen3_asr/run_pytorch_fixtures.py \
  --model /data/models/Qwen3-ASR-0.6B \
  --fixtures tests/qwen3_asr/fixtures.local.json \
  --out-dir /data/results/qwen3_asr/pytorch \
  --updated-fixtures /data/results/qwen3_asr/fixtures.vm.json \
  --path-rewrite /Users/link/llk/test_audio/chinese_fixtures=/data/test_audio/chinese_fixtures \
  --device-map cuda \
  --language Chinese \
  --max-new-tokens 128 \
  --frames 1878 \
  --module-summary-chunks 1 \
  --dump-module-raw
```

The script writes:

- `*_pytorch.json` for original `Qwen3ASRModel.transcribe()` text output;
- `*_pytorch_mel.json` for official feature-extractor summaries on the same
  fixed overlap chunks used by the ncnn runner;
- `*_pytorch_modules.json` for PyTorch module-level summaries on fixed chunks;
- `*_module_raw/*.f32` when `--dump-module-raw` is set;
- an updated fixture JSON that can be passed to `evaluate_fixtures.py`.

## Platform Smoke

After generating `eval_report.json`, collect a local platform smoke report:

```bash
tests/qwen3_asr/collect_platform_smoke.py \
  --binary /Users/link/llk/build/ncnn_llm-macos-qwen3-asr/qwen3_asr_main \
  --model /Users/link/llk/models/qwen3_asr_0_6b_runtime_text128 \
  --eval-report /Users/link/llk/test_audio/chinese_fixtures/eval_report.json \
  --fixtures tests/qwen3_asr/fixtures.local.json \
  --threads 6 \
  --json-out /Users/link/llk/test_audio/chinese_fixtures/platform_smoke.json \
  --markdown-out /Users/link/llk/test_audio/chinese_fixtures/platform_smoke.md
```

This report is a CPU-only smoke/performance record. It should not be mixed with
PyTorch accuracy parity.

Current local macOS CPU-only smoke:

| fixture | strict | semantic | chunks | RTF | peak RSS |
| --- | --- | --- | ---: | ---: | ---: |
| `zh_short_tts` | PASS | PASS | 2 | 10.71 | 3485.5 MiB |
| `zh_long_tts` | FAIL | FAIL | 9 | 11.36 | 3463.8 MiB |
| `zh_mixed_tts` | FAIL | PASS | 4 | 10.33 | 3448.0 MiB |

## VM Work Still Needed

The Linux/PyTorch environment can now fill:

- PyTorch result JSON for each fixture;
- PyTorch mel summary JSON for each fixture;
- PyTorch module-level summaries for speech encoder output, merged embedding,
  hidden state, first decoder logits, and selected logits.
- optional raw `.f32` tensors for ncnn/PyTorch module parity.

When both sides provide `raw_path`, `evaluate_fixtures.py` computes:

- `max_abs`;
- `mean_abs`;
- `p99_abs`;
- cosine similarity;
- top-k token agreement for logits.
