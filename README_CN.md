# ncnn_llm

英文版: [readme.md](readme.md)

**ncnn_llm** 为 [ncnn](https://github.com/Tencent/ncnn) 框架提供大语言模型 (LLM) 和嵌入模型支持。

ncnn 是一个为移动端和嵌入式设备深度优化的高性能神经网络推理框架。通过将 LLM 集成到 ncnn 中，本项目使得在资源受限的环境（边缘设备、手机、IoT）上运行复杂的自然语言处理任务成为可能。

---

## 🚀 项目起源

本项目源自 **nihui** 为 ncnn 实现的 `kvcache` 功能，这为在 ncnn 上运行 LLM 打开了大门。受开源精神鼓舞，本仓库将该功能整理并扩展为一个独立项目。

目标是提供完整的流水线，方便开发者在 ncnn 上使用 LLM，并推动生态建设。

> **⚠️ 重要提示：**
> ncnn 对 `kvcache` 的支持目前仍处于 **实验阶段**。你 **必须** 从 `master` 分支编译 ncnn，以确保具备运行本项目所需的最新特性。

---

## 📊 模型支持矩阵

项目仍在积极开发中。以下是当前模型兼容性状态。

### ✅ 完全支持

*这些模型可以使用已实现的分词器和推理流程顺利运行。*

**LLM 模型：**
* **MiniCPM4**
* **Qwen3**
* **Qwen3.5**
* **Qwen2.5-VL**
* **NLLB** (No Language Left Behind)

**嵌入模型：**
* **Jina-Embeddings-v5-Text-Nano** - 文本嵌入模型 (768维)
* **Jina-CLIP-v2** - 多模态嵌入模型 (文本+图像, 1024维)

### ⚠️ 可运行但存在问题

*这些模型可以加载并运行，但可能存在 bug 或性能欠佳。*

* (暂无)

### 🚧 理论支持（开发中）

*这些模型理论上可以工作，但在当前版本中仍失败或未验证。*

* TinyLlama-1.1B-Chat-v1.0
* Qwen2.5-0.5B
* Llama-3.2-1B-Instruct
* DeepSeek-R1-Distill-Qwen-1.5b

### 🔜 即将支持

* PaddleOCR-VL

---

## 🛠️ 构建与使用

本项目使用 `xmake` 构建。

### 依赖项

- **xmake** - 构建系统
- **ncnn** (master 分支) - 神经网络推理框架
- **OpenCV** (可选) - 用于视觉语言模型支持
- **nlohmann_json** - JSON 库

### 1. 克隆仓库

```bash
git clone https://github.com/futz12/ncnn_llm.git
cd ncnn_llm
```

### 2. 构建

```bash
xmake build
```

### 3. 运行（示例：llm_ncnn_run）

运行前请确保已下载模型权重（见下文）。

```bash
xmake run llm_ncnn_run --model ./assets/qwen3_0.6b
```

### 命令行选项

| 选项 | 描述 |
|------|------|
| `--model` | 模型目录路径（必需） |
| `--threads` | 线程数（默认：自动） |
| `--vulkan` | 启用 Vulkan GPU 加速 |
| `--vulkan-device` | 指定 Vulkan 设备 ID |
| `--image` | 图像文件路径（用于 VL 模型） |
| `--builtin-tools` | 启用内置工具 |

### 示例输出

```text
Chat with Qwen3-0.6B! Type 'exit' or 'quit' to end the conversation.
User: Hello
Assistant: 
Hello! How can I assist you today?
User: What is OpenCV?
Assistant: OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library...
```

---

## 🚀 llm_ncnn_run（CLI）

`llm_ncnn_run` 是一个统一示例，支持：
- CLI 对话模式
- 内置工具（random/add）
- 视觉语言模型支持（需要 OpenCV）

### 构建

```bash
xmake build llm_ncnn_run
```

### 运行

```bash
xmake run llm_ncnn_run --model ./assets/qwen3_0.6b
```

说明：
- 模型路径必须是包含模型文件的有效目录。
- 从 https://mirrors.sdu.edu.cn/ncnn_modelzoo/ 下载模型

---

## 🔤 嵌入模型（Embedding）

本项目支持文本嵌入和多模态嵌入模型，提供统一的 `ncnn_embedding` 接口。

### 支持的模型

| 模型 | 类型 | 维度 | 说明 |
|------|------|------|------|
| Jina-Embeddings-v5-Text-Nano | 文本嵌入 | 768 | 多语言文本嵌入 |
| Jina-CLIP-v2 | 多模态嵌入 | 1024 | 文本+图像嵌入 |

### 构建

```bash
xmake build embedding_main
xmake build clip_main
```

### 文本嵌入示例

```bash
xmake run embedding_main --model ./assets/jina-embeddings-v5-text-nano
```

输出示例：
```text
Text-Text Similarity Matrix:
               今天天气.. 今天天气.. 我喜欢吃.. 我喜欢吃.. The weather ..
今天天气.. 1.0000         0.7116         0.6915         0.6872         0.6823
...
```

### CLIP 多模态嵌入示例

```bash
xmake run clip_main --model ./assets/jina_clip_v2 --image ./assets/ganyu.jpg
```

输出示例：
```text
Text-Image Similarity Matrix:
                              assets/ganyu.jpg
a cat                         0.0996
a dog                         0.0595
blue hair anime character     0.2720
蓝色头发动漫角色              0.2823
```

### API 使用

```cpp
#include "ncnn_embedding.h"

// 加载模型
ncnn_embedding embed("./assets/jina_clip_v2", false, 4);

// 文本编码
std::vector<float> text_vec = embed.encode_text("Hello world");

// 图像编码（仅 CLIP 模型）
if (embed.supports_image()) {
    std::vector<float> image_vec = embed.encode_image_file("./image.jpg");
}

// 计算相似度
float similarity = cosine_similarity(text_vec, image_vec);
```

### model.json 配置格式

**文本嵌入模型：**
```json
{
    "model_type": "embedding",
    "params": {
        "encoder_param": "model.ncnn.param",
        "encoder_bin": "model.ncnn.bin"
    },
    "tokenizer": {
        "type": "bbpe",
        "vocab_file": "vocab.txt",
        "merges_file": "merges.txt"
    },
    "setting": {
        "embed_dim": 768,
        "rope": {
            "type": "RoPE_full",
            "rope_head_dim": 64,
            "rope_theta": 1000000.0
        }
    }
}
```

**CLIP 多模态模型：**
```json
{
    "model_type": "clip",
    "params": {
        "text_encoder_param": "text.ncnn.param",
        "text_encoder_bin": "text.ncnn.bin",
        "vision_encoder_param": "vision.ncnn.param",
        "vision_encoder_bin": "vision.ncnn.bin"
    },
    "tokenizer": {
        "type": "unigram",
        "vocab_file": "vocab.txt"
    },
    "setting": {
        "text_embed_dim": 1024,
        "vision_embed_dim": 1024,
        "image_size": 512,
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "rope": {
            "type": "RoPE_full",
            "rope_head_dim": 64,
            "rope_theta": 1000000.0
        }
    }
}
```

---

## 📊 性能测试

项目包含性能测试工具。

### 构建

```bash
xmake build benchllm
```

### 运行

```bash
xmake run benchllm [loop_count] [num_threads] [powersave] [gpu_device] [cooling_down] [seqlen]
```

### 参数说明

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `loop_count` | 4 | 基准测试迭代次数 |
| `num_threads` | 自动 | CPU 线程数 |
| `powersave` | 2 | CPU 省电模式 (0-2) |
| `gpu_device` | -1 | Vulkan 设备 ID（-1 表示仅 CPU） |
| `cooling_down` | 1 | 测试间启用冷却 |
| `seqlen` | 233 | 基准测试序列长度 |

---

## 🧪 测试

项目包含单元测试。

### 构建并运行测试

```bash
xmake build test_llm
xmake run test_llm
```

---

## 📁 项目结构

```
ncnn_llm/
├── src/                    # 核心库源码
│   ├── ncnn_llm_gpt.cpp    # 主 LLM 推理实现
│   ├── ncnn_embedding.cpp  # 嵌入模型实现
│   ├── sampling.cpp        # Token 采样策略
│   ├── nllb_600m.cpp       # NLLB 模型支持
│   └── utils/              # 工具模块
│       ├── tokenizer/      # 分词器实现 (BPE, Unigram)
│       ├── image_utils.cpp # 图像处理工具
│       ├── gdr.cpp         # GDR 支持
│       ├── prompt.cpp      # Prompt 处理
│       └── rope_embed.cpp  # RoPE 嵌入
├── examples/               # 示例应用
│   ├── llm_ncnn_run/       # 统一 CLI 运行器
│   ├── embedding_main.cpp  # 文本嵌入示例
│   ├── clip_main.cpp       # CLIP 多模态嵌入示例
│   ├── nllb_main.cpp       # NLLB 翻译示例
│   └── unigram_main.cpp    # Unigram 分词器示例
├── benchmark/              # 性能基准测试
│   └── benchllm.cpp
├── tests/                  # 单元测试
│   └── test_llm.cpp
├── export/                 # 模型导出脚本
│   └── nllb_export.py
└── xmake.lua              # 构建配置
```

---

## 📥 模型库

你可以从以下镜像下载已转换的 ncnn 模型权重：

🔗 **[ncnn 模型库镜像](https://mirrors.sdu.edu.cn/ncnn_modelzoo/)**

---

## 🔮 路线图

我们将持续改进 ncnn_llm，未来计划包括：

* **上游优化：** 向 ncnn 上游提交优化补丁，提升核心 LLM 支持。
* **支持扩展：** 增加更多模型架构与分词器支持。
* **性能：** 优化推理速度并降低内存占用。
* **INT8 量化：** 实现 INT8 量化支持。
* **文档：** 完善导出流水线文档并增加更多 C++ 使用示例。

*注：虽然我们提供完整的导出流水线，但旧的流程可能会随库演进而过时。请参考最新示例代码进行调整。*

---

## 🤝 社区与联系

欢迎大家关注并参与本项目，共同推动 ncnn 在大语言模型领域的发展！

* **QQ群：** `767178345`

---

## 📝 许可证

Apache 2.0