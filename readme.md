# ncnn_llm

中文版: [README_CN](README_CN.md)

**ncnn_llm** provides Large Language Model (LLM) support for the [ncnn](https://github.com/Tencent/ncnn) framework.

ncnn is a high-performance neural network inference framework specifically optimized for mobile and embedded devices. By integrating LLMs into ncnn, this project enables the execution of complex natural language processing tasks in resource-constrained environments (edge devices, mobile phones, IoT).

---

## 🚀 Project Origin

This project originated from **nihui's** implementation of the `kvcache` feature for ncnn, which opened the door for running LLMs on the framework. Motivated by the spirit of open-source contribution, this repository organizes and expands upon that functionality into an independent project.

The goal is to provide a complete pipeline, making it easier for developers to use LLMs on ncnn and contribute to the ecosystem.

> **⚠️ Important Note:**
> ncnn's support for `kvcache` is currently in an **experimental stage**. You **must** compile ncnn from the `master` branch to ensure you have the latest features required for this project to run.

---

## 📊 Model Support Matrix

The project is currently in active development. Below is the current compatibility status of various models.

### ✅ Perfectly Supported

*These models run smoothly with the implemented tokenizer and inference pipeline.*

* **MiniCPM4**
* **Qwen3**
* **Qwen3.5**
* **Qwen2.5-VL**
* **NLLB** (No Language Left Behind)

### ⚠️ Running with Issues

*These models can be loaded and run, but may experience bugs or suboptimal performance.*

* (None currently)

### 🚧 Theoretical Support (Work in Progress)

*These models should theoretically work but are currently failing or unverified in the current build.*

* TinyLlama-1.1B-Chat-v1.0
* Qwen2.5-0.5B
* Llama-3.2-1B-Instruct
* DeepSeek-R1-Distill-Qwen-1.5b

### 🔜 Coming Soon

* PaddleOCR-VL

---

## 🛠️ Build and Usage

This project uses `xmake` for building.

### Prerequisites

- **xmake** - Build system
- **ncnn** (master branch) - Neural network inference framework
- **OpenCV** (optional) - For vision-language model support
- **nlohmann_json** - JSON library

### 1. Clone the Repository

```bash
git clone https://github.com/futz12/ncnn_llm.git
cd ncnn_llm
```

### 2. Build

```bash
xmake build
```

### 3. Run (Example: llm_ncnn_run)

Ensure you have downloaded the model weights (see below) before running.

```bash
xmake run llm_ncnn_run --model ./assets/qwen3_0.6b
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--model` | Path to model directory (required) |
| `--threads` | Number of threads (default: auto) |
| `--vulkan` | Enable Vulkan GPU acceleration |
| `--vulkan-device` | Specify Vulkan device ID |
| `--image` | Path to image file (for VL models) |
| `--builtin-tools` | Enable built-in tools |

### Example Output

```text
Chat with Qwen3-0.6B! Type 'exit' or 'quit' to end the conversation.
User: Hello
Assistant: 
Hello! How can I assist you today?
User: What is OpenCV?
Assistant: OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library...
```

---

## 🚀 llm_ncnn_run (CLI)

`llm_ncnn_run` is a unified example that supports:
- CLI chat mode
- Built-in tools (random/add)
- Vision-language model support (with OpenCV)

### Build

```bash
xmake build llm_ncnn_run
```

### Run

```bash
xmake run llm_ncnn_run --model ./assets/qwen3_0.6b
```

Notes:
- Model path must be a valid directory containing model files.
- Download models from https://mirrors.sdu.edu.cn/ncnn_modelzoo/

---

## 📊 Benchmark

The project includes a benchmark tool for performance testing.

### Build

```bash
xmake build benchllm
```

### Run

```bash
xmake run benchllm [loop_count] [num_threads] [powersave] [gpu_device] [cooling_down] [seqlen]
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `loop_count` | 4 | Number of benchmark iterations |
| `num_threads` | auto | Number of CPU threads |
| `powersave` | 2 | CPU powersave mode (0-2) |
| `gpu_device` | -1 | Vulkan device ID (-1 for CPU only) |
| `cooling_down` | 1 | Enable cooling down between tests |
| `seqlen` | 233 | Sequence length for benchmark |

---

## 🧪 Testing

The project includes unit tests.

### Build and Run Tests

```bash
xmake build test_llm
xmake run test_llm
```

---

## 📁 Project Structure

```
ncnn_llm/
├── src/                    # Core library source
│   ├── ncnn_llm_gpt.cpp    # Main LLM inference implementation
│   ├── sampling.cpp        # Token sampling strategies
│   ├── nllb_600m.cpp       # NLLB model support
│   └── utils/              # Utility modules
│       ├── tokenizer/      # Tokenizer implementations (BPE, Unigram)
│       ├── gdr.cpp         # GDR support
│       ├── prompt.cpp      # Prompt handling
│       └── rope_embed.cpp  # RoPE embedding
├── examples/               # Example applications
│   ├── llm_ncnn_run/       # Unified CLI runner
│   ├── bytelevelbpe_main.cpp
│   ├── nllb_main.cpp
│   └── unigram_main.cpp
├── benchmark/              # Performance benchmarks
│   └── benchllm.cpp
├── tests/                  # Unit tests
│   └── test_llm.cpp
├── export/                 # Model export scripts
│   └── nllb_export.py
└── xmake.lua              # Build configuration
```

---

## 📥 Model Zoo

You can download the converted ncnn-compatible model weights from the following mirror:

🔗 **[ncnn Model Zoo Mirror](https://mirrors.sdu.edu.cn/ncnn_modelzoo/)**

---

## 🔮 Roadmap

We are committed to improving ncnn_llm. Our future plans include:

* **Upstream Optimization:** Submitting optimization patches directly to the upstream ncnn repository to improve core LLM support.
* **Expanded Support:** Adding support for more model architectures and tokenizers.
* **Performance:** Optimizing inference speed and reducing memory footprint.
* **INT8 Quantization:** Implementing INT8 quantization support.
* **Documentation:** Improving the export pipeline docs and adding more C++ usage examples.

*Note: While we provide a complete export pipeline, older pipelines may become obsolete as the library evolves. Please refer to the latest example code for adjustments.*

---

## 🤝 Community & Contact

We welcome everyone to pay attention to and participate in this project to jointly promote the development of ncnn in the field of Large Language Models!

* **QQ Group:** `767178345`

---

## 📝 License

Apache 2.0