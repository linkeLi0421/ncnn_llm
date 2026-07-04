#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "ncnn_qwen3_asr.h"

struct Args {
    std::string model_path = "./assets/qwen3_asr_0.6b";
    std::string audio_features_raw;
    int mel_bins = 128;
    int frames = 256;
    std::vector<int> token_ids;
    bool use_vulkan = false;
    int threads = 4;
};

static void print_usage(const char* prog) {
    std::cout
        << "Usage: " << prog << " [--model DIR] [--audio-features-raw FILE --mel-bins N --frames N]\n"
        << "                 [--tokens comma,separated,ids] [--threads N] [--vulkan]\n\n"
        << "Examples:\n"
        << "  " << prog << " --model ./assets/qwen3_asr_0.6b --audio-features-raw mel.f32 --mel-bins 128 --frames 256\n"
        << "  " << prog << " --model ./assets/qwen3_asr_0.6b --tokens 1,2,3,4,5,6,7,8\n";
}

static std::vector<int> parse_int_list(const std::string& text) {
    std::vector<int> ids;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            ids.push_back(std::stoi(item));
        }
    }
    return ids;
}

static Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        auto need_value = [&](const char* flag) {
            if (i + 1 >= argc) {
                std::cerr << flag << " requires a value\n";
                print_usage(argv[0]);
                std::exit(1);
            }
        };

        if (arg == "--model") {
            need_value("--model");
            args.model_path = argv[++i];
        } else if (arg == "--audio-features-raw") {
            need_value("--audio-features-raw");
            args.audio_features_raw = argv[++i];
        } else if (arg == "--mel-bins") {
            need_value("--mel-bins");
            args.mel_bins = std::stoi(argv[++i]);
        } else if (arg == "--frames") {
            need_value("--frames");
            args.frames = std::stoi(argv[++i]);
        } else if (arg == "--tokens") {
            need_value("--tokens");
            args.token_ids = parse_int_list(argv[++i]);
        } else if (arg == "--threads") {
            need_value("--threads");
            args.threads = std::stoi(argv[++i]);
        } else if (arg == "--vulkan") {
            args.use_vulkan = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            std::cerr << "Unknown arg: " << arg << "\n";
            print_usage(argv[0]);
            std::exit(1);
        }
    }
    return args;
}

static bool read_file(const std::string& path, std::vector<float>& values) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    f.seekg(0, std::ios::end);
    const std::streamoff size = f.tellg();
    f.seekg(0, std::ios::beg);
    if (size < 0 || (size % (std::streamoff)sizeof(float)) != 0) {
        return false;
    }
    values.resize((size_t)size / sizeof(float));
    f.read(reinterpret_cast<char*>(values.data()), size);
    return (bool)f;
}

static void print_mat_shape(const char* name, const ncnn::Mat& mat) {
    std::cout << name << ": dims=" << mat.dims
              << " w=" << mat.w
              << " h=" << mat.h
              << " d=" << mat.d
              << " c=" << mat.c
              << " elemsize=" << mat.elemsize
              << " elempack=" << mat.elempack
              << " total=" << mat.total()
              << "\n";
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    ncnn_qwen3_asr asr(args.model_path, args.use_vulkan, args.threads);
    if (!asr.ok()) {
        std::cerr << "Failed to load Qwen3-ASR model\n";
        return 1;
    }

    if (!args.audio_features_raw.empty()) {
        std::vector<float> features;
        if (!read_file(args.audio_features_raw, features)) {
            std::cerr << "Failed to read audio features: " << args.audio_features_raw << "\n";
            return 2;
        }
        const size_t expected = (size_t)args.mel_bins * (size_t)args.frames;
        if (features.size() != expected) {
            std::cerr << "Bad audio feature count: got " << features.size()
                      << ", expected " << expected << "\n";
            return 3;
        }

        ncnn::Mat mel(args.frames, args.mel_bins, (void*)features.data(), sizeof(float), 1);
        mel = mel.clone();
        ncnn::Mat audio = asr.run_audio_encoder(mel);
        if (audio.total() == 0) {
            return 4;
        }
        print_mat_shape("audio_encoder", audio);
    }

    if (!args.token_ids.empty()) {
        std::vector<int> mask(args.token_ids.size(), 1);
        ncnn::Mat embeds = asr.run_text_embed(args.token_ids);
        if (embeds.total() == 0) {
            return 5;
        }
        print_mat_shape("text_embed", embeds);

        ncnn::Mat hidden = asr.run_text_backbone(embeds, mask);
        if (hidden.total() == 0) {
            return 6;
        }
        print_mat_shape("text_backbone", hidden);

        ncnn::Mat logits = asr.run_lm_head(hidden);
        if (logits.total() == 0) {
            return 7;
        }
        print_mat_shape("lm_head", logits);
        std::cout << "next_token=" << asr.select_next_token_from_logits(logits) << "\n";
    }

    if (args.audio_features_raw.empty() && args.token_ids.empty()) {
        std::cout << "Model loaded. Provide --audio-features-raw or --tokens to run a smoke test.\n";
    }

    return 0;
}
