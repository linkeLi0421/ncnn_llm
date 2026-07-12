# Qwen3-ASR Fixture Evaluation

This directory defines the local evaluation contract for Qwen3-ASR ncnn parity.
It follows the validation split suggested from the FunASR / Qwen3-ASR side:

1. Fix and summarize audio preprocessing.
2. Compare end-to-end text on multiple fixture categories.
3. Keep module-level summaries for localization.
4. Keep platform smoke/performance separate from PyTorch accuracy parity.
5. Provide a minimal regression entry point.

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

Example:

```bash
qwen3_asr_main \
  --model /Users/link/llk/models/qwen3_asr_0_6b_runtime_text128 \
  --audio-wav /Users/link/llk/test_audio/chinese_fixtures/zh_short_16k.wav \
  --generate-from-features \
  --max-new-tokens 64 \
  --threads 4 \
  --text-out /Users/link/llk/test_audio/chinese_fixtures/zh_short_default_final.txt \
  --json-out /Users/link/llk/test_audio/chinese_fixtures/zh_short_default_final.json \
  --dump-mel-summary /Users/link/llk/test_audio/chinese_fixtures/zh_short_default_final_mel.json
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
