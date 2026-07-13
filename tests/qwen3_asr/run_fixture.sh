#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_fixture.sh --binary BIN --model DIR --audio WAV --out-dir DIR [--id NAME] [--threads N] [--max-new-tokens N] [--energy-chunking] [--measure-memory]

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
measure_memory="false"

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
    --measure-memory)
      measure_memory="true"
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

if [[ "$measure_memory" == "true" ]]; then
  if [[ ! -x /usr/bin/time ]]; then
    echo "/usr/bin/time is required for --measure-memory" >&2
    exit 3
  fi
  /usr/bin/time -l "${cmd[@]}" 2> "$out_dir/${id}_time.txt"
  python3 - "$out_dir/${id}_time.txt" "$out_dir/${id}_metrics.json" <<'PY'
import json
import re
import sys
from pathlib import Path

time_log = Path(sys.argv[1])
metrics_path = Path(sys.argv[2])
text = time_log.read_text(encoding="utf-8", errors="replace")

def find(pattern):
    m = re.search(pattern, text)
    return int(m.group(1)) if m else None

metrics = {
    "time_log": str(time_log),
    "max_resident_set_size_bytes": find(r"(\d+)\s+maximum resident set size"),
    "page_reclaims": find(r"(\d+)\s+page reclaims"),
    "page_faults": find(r"(\d+)\s+page faults"),
    "voluntary_context_switches": find(r"(\d+)\s+voluntary context switches"),
    "involuntary_context_switches": find(r"(\d+)\s+involuntary context switches"),
}
metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
PY
else
  "${cmd[@]}"
fi

python3 -m json.tool "$out_dir/${id}.json" >/dev/null
python3 -m json.tool "$out_dir/${id}_mel.json" >/dev/null
if [[ "$measure_memory" == "true" ]]; then
  python3 -m json.tool "$out_dir/${id}_metrics.json" >/dev/null
fi

echo "text_out=$out_dir/${id}.txt"
echo "json_out=$out_dir/${id}.json"
echo "mel_summary=$out_dir/${id}_mel.json"
if [[ "$measure_memory" == "true" ]]; then
  echo "time_log=$out_dir/${id}_time.txt"
  echo "metrics=$out_dir/${id}_metrics.json"
fi
