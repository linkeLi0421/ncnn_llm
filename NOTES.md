# Qwen3-ASR ncnn Notes

This repo is tracking early work for Tencent/ncnn issue #6790: exporting
Qwen3-ASR to an ncnn-friendly layout.

## Current Status

- Added an exporter scaffold at `export/qwen3_asr_export.py`.
- The exporter loads Qwen3-ASR from Hugging Face/Transformers, saves the
  processor files, and traces separate TorchScript modules for:
  - audio encoder
  - text token embedding
  - text backbone
  - lm head
- The exporter writes `model.json`, `vocab.txt`, and `merges.txt`.
- `--convert-ncnn` can run `pnnx` on the generated TorchScript files.
- README files include basic Qwen3-ASR export commands.

## Hardware/Software Used

- GPU target: NVIDIA RTX 4090 24 GB.
- Driver/CUDA checked on the VM: NVIDIA driver 570.124.06, CUDA 12.8.
- PyTorch CUDA build was used for exporter validation.
- ncnn and pnnx are expected to be built locally on the export machine.

## Validation Done

- Local Python compile check passed:
  `python3 -m py_compile export/qwen3_asr_export.py`
- Remote GPU VM checks passed:
  - exporter compile
  - CLI `--help`
  - module import
  - CUDA device detection
  - tokenizer helper smoke test for `vocab.txt` and `merges.txt`

## Not Done Yet

- Full Qwen3-ASR model export was not run because the VM did not have the model
  cached locally and downloading it is a multi-GB operation.
- The generated ncnn models still need runtime integration and numerical checks
  against PyTorch outputs.
