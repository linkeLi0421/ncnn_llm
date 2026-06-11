#include <iostream>
#include <string>
#include "ncnn_llm_ocr.h"

int main(int argc, char** argv) {
    std::string model_path = "assets/glm_ocr";
    std::string image_path;
    std::string prompt = "Read the text in the image.";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--image" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        }
    }

    if (image_path.empty()) {
        fprintf(stderr, "Usage: %s --image <image_path> [--model <model_path>] [--prompt <prompt>]\n", argv[0]);
        return 1;
    }

    printf("Loading GLM-OCR model from %s\n", model_path.c_str());

    ncnn_llm_ocr ocr(model_path, false, 4);
    if (!ocr.ok()) {
        fprintf(stderr, "Failed to load GLM-OCR model\n");
        return 1;
    }

    printf("Loading image: %s\n", image_path.c_str());
    ncnn::Mat bgr = load_image_to_ncnn_mat(image_path);
    if (ncnn_mat_empty(bgr)) {
        fprintf(stderr, "Failed to load image: %s\n", image_path.c_str());
        return 1;
    }

    printf("Running OCR prefill with prompt: %s\n", prompt.c_str());
    auto ctx = ocr.prefill(prompt, bgr);

    printf("Generating text:\n");

    GenerateConfig cfg;
    cfg.max_new_tokens = 256;
    cfg.temperature = 0.0f;
    cfg.top_p = 0.00001f;
    cfg.top_k = 1;
    cfg.repetition_penalty = 1.1f;
    cfg.do_sample = 0;

    ctx = ocr.generate(ctx, cfg, [](const std::string& token) {
        printf("%s", token.c_str());
        fflush(stdout);
    });

    printf("\n\nDone.\n");
    return 0;
}
