#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_fixture.sh --binary BIN --model DIR --audio WAV --out-dir DIR [--id NAME] [--threads N] [--max-new-tokens N] [--energy-chunking]

Example:
  tests/qwen3_asr/run_fixture.sh \
    --binary /Users/link/llk/build/ncnn_llm-macos-qwen3-asr/qwen3_asr_main \
    --model /Users/link/llk/models/qwen3_asr_0_6b_runtime_text128 \
    --audio /Users/link/llk/test_audio/chinese_fixtures/zh_short_16k.wav \
    --out-dir /Users/link/llk/test_audio/chinese_fixtures \
    --id zh_short_default_final
USAGE
}

binary=""
model=""
audio=""
out_dir=""
id=""
threads="4"
max_new_tokens="64"
energy_chunking="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --binary)
      binary="$2"
      shift 2
      ;;
    --model)
      model="$2"
      shift 2
      ;;
    --audio)
      audio="$2"
      shift 2
      ;;
    --out-dir)
      out_dir="$2"
      shift 2
      ;;
    --id)
      id="$2"
      shift 2
      ;;
    --threads)
      threads="$2"
      shift 2
      ;;
    --max-new-tokens)
      max_new_tokens="$2"
      shift 2
      ;;
    --energy-chunking)
      energy_chunking="true"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$binary" || -z "$model" || -z "$audio" || -z "$out_dir" ]]; then
  usage >&2
  exit 2
fi

if [[ -z "$id" ]]; then
  id="$(basename "$audio")"
  id="${id%.*}"
fi

mkdir -p "$out_dir"

cmd=(
  "$binary"
  --model "$model"
  --audio-wav "$audio"
  --generate-from-features
  --max-new-tokens "$max_new_tokens"
  --threads "$threads"
  --text-out "$out_dir/${id}.txt"
  --json-out "$out_dir/${id}.json"
  --dump-mel-summary "$out_dir/${id}_mel.json"
)

if [[ "$energy_chunking" == "true" ]]; then
  cmd+=(--energy-chunking)
fi

"${cmd[@]}"

python3 -m json.tool "$out_dir/${id}.json" >/dev/null
python3 -m json.tool "$out_dir/${id}_mel.json" >/dev/null

echo "text_out=$out_dir/${id}.txt"
echo "json_out=$out_dir/${id}.json"
echo "mel_summary=$out_dir/${id}_mel.json"
