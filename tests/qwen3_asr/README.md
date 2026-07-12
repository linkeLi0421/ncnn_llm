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

## ncnn Regression Entry

Use `run_fixture.sh` so each run produces the same three artifacts:

- final text;
- structured result JSON;
- mel summary JSON.

Example:

```bash
tests/qwen3_asr/run_fixture.sh \
  --binary /Users/link/llk/build/ncnn_llm-macos-qwen3-asr/qwen3_asr_main \
  --model /Users/link/llk/models/qwen3_asr_0_6b_runtime_text128 \
  --audio /Users/link/llk/test_audio/chinese_fixtures/zh_short_16k.wav \
  --out-dir /Users/link/llk/test_audio/chinese_fixtures \
  --id zh_short_default_final \
  --threads 4 \
  --max-new-tokens 64
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
- ncnn mel summary presence and preprocessing fields;
- optional PyTorch mel summary when available;
- optional module-level summary when available.

Current pass/fail is based on normalized text only. Raw and cleaned text are kept
in the JSON report so postprocess problems are not confused with model parity.

## VM Work Still Needed

The following fields require the Linux/PyTorch environment:

- PyTorch result JSON for each fixture.
- PyTorch mel summary JSON for each fixture.
- Module-level parity summaries:
  - speech encoder output;
  - projector/adaptor output if separable;
  - first decoder logits.

Suggested numeric comparisons for module-level parity:

- `max_abs`;
- `mean_abs`;
- `p99_abs`;
- cosine similarity;
- top-k token agreement for logits.
