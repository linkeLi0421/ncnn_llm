#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "ncnn_embedding.h"

static float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-10f);
}

static void print_similarity_matrix(const std::vector<std::vector<float>>& text_embeds,
                                    const std::vector<std::vector<float>>& image_embeds,
                                    const std::vector<std::string>& texts,
                                    const std::vector<std::string>& images) {
    printf("\nText-Image Similarity Matrix:\n");
    printf("%-30s", "");
    for (const auto& img : images) {
        printf("%-20s", img.c_str());
    }
    printf("\n");

    for (size_t i = 0; i < texts.size(); ++i) {
        printf("%-30s", texts[i].c_str());
        for (size_t j = 0; j < image_embeds.size(); ++j) {
            float sim = cosine_similarity(text_embeds[i], image_embeds[j]);
            printf("%-20.4f", sim);
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {
    std::string model_path = "assets/jina_clip_v2";
    std::string image_path = "";

    if (argc > 1) {
        model_path = argv[1];
    }
    if (argc > 2) {
        image_path = argv[2];
    }

    ncnn_embedding embed(model_path, false, 4);

    if (!embed.ok()) {
        fprintf(stderr, "Failed to load embedding model\n");
        return 1;
    }

    std::vector<std::string> texts = {
        "a cat",
        "a dog",
        "a car",
        "blue hair anime character",
        "蓝色头发动漫角色",
        "一只猫",
        "一辆汽车"
    };

    std::vector<std::vector<float>> text_embeds;
    printf("\nEncoding texts...\n");
    for (const auto& text : texts) {
        auto e = embed.encode_text(text);
        text_embeds.push_back(e);
        printf("  '%s' -> %d dims\n", text.c_str(), (int)e.size());
    }

    std::vector<std::string> image_files;
    std::vector<std::vector<float>> image_embeds;

    if (!image_path.empty() && embed.supports_image()) {
        printf("\nEncoding image: %s\n", image_path.c_str());
        auto e = embed.encode_image_file(image_path);
        image_embeds.push_back(e);
        image_files.push_back(image_path);
        printf("  -> %d dims\n", (int)e.size());
    }

    if (!image_embeds.empty()) {
        print_similarity_matrix(text_embeds, image_embeds, texts, image_files);
    }

    return 0;
}