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

static void print_similarity_matrix(const std::vector<std::vector<float>>& embeds,
                                    const std::vector<std::string>& labels) {
    printf("%-15s", "");
    for (const auto& l : labels) {
        std::string s = l.length() > 12 ? l.substr(0, 12) + ".." : l;
        printf("%-15s", s.c_str());
    }
    printf("\n");

    for (size_t i = 0; i < embeds.size(); ++i) {
        std::string s = labels[i].length() > 12 ? labels[i].substr(0, 12) + ".." : labels[i];
        printf("%-15s", s.c_str());
        for (size_t j = 0; j < embeds.size(); ++j) {
            float sim = cosine_similarity(embeds[i], embeds[j]);
            printf("%-15.4f", sim);
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {
    std::string model_path = "assets/jina-embeddings-v5-text-nano";

    if (argc > 1) {
        model_path = argv[1];
    }

    ncnn_embedding embed(model_path, false, 4);

    if (!embed.ok()) {
        fprintf(stderr, "Failed to load embedding model\n");
        return 1;
    }

    std::vector<std::string> texts = {
        "今天天气很好",
        "今天天气不错",
        "我喜欢吃苹果",
        "我喜欢吃香蕉",
        "The weather is nice today"
    };

    printf("\nEncoding texts...\n");
    std::vector<std::vector<float>> text_embeds;
    for (const auto& text : texts) {
        auto e = embed.encode_text(text);
        text_embeds.push_back(e);
        printf("  '%s' -> %d dims\n", text.c_str(), (int)e.size());
    }

    printf("\nText-Text Similarity Matrix:\n");
    print_similarity_matrix(text_embeds, texts);

    return 0;
}