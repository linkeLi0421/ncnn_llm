# Qwen3-ASR native Windows + WSL2 build evidence

Date: 2026-07-15

Branch baseline: `qwen3-asr-export` at `35174d2`.

## Environment

```text
Host: Windows 11 AMD64, Intel Core i9-12950HX, 31.6 GiB RAM
WSL2: Ubuntu 24.04, GCC 13.3.0, CMake 3.28, Ninja 1.11
Windows native: MSYS2 UCRT64, GCC 16.1.0, CMake 4.4, Ninja 1.13
ncnn: 1.0.20260715, CPU build, Vulkan disabled
nlohmann-json: 3.12.0
```

## WSL2 build and test

```text
$ cmake -S . -B /home/zhangzherui/build/qwen3-asr -G Ninja \
    -Dncnn_DIR=/home/zhangzherui/install/ncnn/lib/cmake/ncnn \
    -DBUILD_TESTING=ON
$ cmake --build /home/zhangzherui/build/qwen3-asr -j 8
[9/9] Linking CXX executable qwen3_asr_main
$ ctest --test-dir /home/zhangzherui/build/qwen3-asr --output-on-failure
1/1 Test #1: qwen3_asr_platform_smoke_py ...... Passed
100% tests passed, 0 tests failed out of 1
```

## Native Windows build and test

```text
-- The CXX compiler identification is GNU 16.1.0
-- Found ncnn: 20260715
[4/4] Linking CXX executable qwen3_asr_main.exe
$ ctest --test-dir C:/Users/zhangzherui/Desktop/Tencent/qwen3-asr-build --output-on-failure
1/1 Test #1: qwen3_asr_platform_smoke_py ...... Passed
100% tests passed out of 1

$ qwen3_asr_main.exe --help
Usage: qwen3_asr_main.exe [--model DIR] [--audio-wav FILE | --audio-features-raw FILE]
                         [--language NAME] [--max-new-tokens N]
                         [--use-kv-cache] [--threads N] [--vulkan]
```

## Native Windows smoke collector output

```text
# Qwen3-ASR Platform Smoke

- OS: Windows 11 (AMD64)
- CPU: Intel64 Family 6 Model 151 Stepping 2, GenuineIntel
- Threads: 8
- Vulkan: false
- Binary: qwen3_asr_main.exe
```

The smoke collector previously queried macOS-only `sysctl` keys and always
described its result as macOS. The accompanying test now exercises Windows and
Linux metadata paths on every CMake/CTest run.
