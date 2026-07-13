#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  make_chinese_fixtures.sh --out-dir DIR [--voice VOICE]

Creates three local Mandarin fixture WAV files:
  zh_short_16k.wav
  zh_long_16k.wav
  zh_mixed_16k.wav

The generated WAV files are 16 kHz, mono, PCM16.

This script uses macOS `say` and `ffmpeg`, so it is intended for local fixture
generation only. These TTS fixtures are smoke tests, not a replacement for real
Mandarin recordings.
USAGE
}

out_dir=""
voice="Eddy (Chinese (China mainland))"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)
      out_dir="$2"
      shift 2
      ;;
    --voice)
      voice="$2"
      shift 2
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

if [[ -z "$out_dir" ]]; then
  usage >&2
  exit 2
fi

if ! command -v say >/dev/null 2>&1; then
  echo "macOS say command is required" >&2
  exit 3
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required" >&2
  exit 3
fi

src_dir="$out_dir/src"
mkdir -p "$src_dir" "$out_dir"

say -v "$voice" -o "$src_dir/zh_short.aiff" \
  "今天天气很好，我们一起测试语音识别。"

say -v "$voice" -o "$src_dir/zh_long.aiff" \
  "这是一段较长的中文语音测试。我们希望模型能够稳定地识别连续的句子，并且在音频变长以后，仍然保持合理的文本输出。这个样例主要用于检查切块，拼接，以及端到端文本对齐是否可靠。"

say -v "$voice" -o "$src_dir/zh_mixed.aiff" \
  "今天是二零二六年七月十二日，OpenAI API 测试成功，订单编号是一二三四五。"

ffmpeg -y -hide_banner -loglevel error -i "$src_dir/zh_short.aiff" \
  -ac 1 -ar 16000 -sample_fmt s16 "$out_dir/zh_short_16k.wav"
ffmpeg -y -hide_banner -loglevel error -i "$src_dir/zh_long.aiff" \
  -ac 1 -ar 16000 -sample_fmt s16 "$out_dir/zh_long_16k.wav"
ffmpeg -y -hide_banner -loglevel error -i "$src_dir/zh_mixed.aiff" \
  -ac 1 -ar 16000 -sample_fmt s16 "$out_dir/zh_mixed_16k.wav"

cat > "$out_dir/fixtures_text.json" <<'JSON'
{
  "zh_short_16k.wav": "今天天气很好，我们一起测试语音识别。",
  "zh_long_16k.wav": "这是一段较长的中文语音测试。我们希望模型能够稳定地识别连续的句子，并且在音频变长以后，仍然保持合理的文本输出。这个样例主要用于检查切块、拼接，以及端到端文本对齐是否可靠。",
  "zh_mixed_16k.wav": "今天是二零二六年七月十二日，OpenAI API 测试成功，订单编号是一二三四五。"
}
JSON

echo "created fixtures in $out_dir"
