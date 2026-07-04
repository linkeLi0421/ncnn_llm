#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "ncnn_qwen3_asr.h"
#include "utils/audio_utils.h"

struct Args {
    std::string model_path = "./assets/qwen3_asr_0.6b";
    std::string audio_features_raw;
    std::string audio_wav;
    int mel_bins = 128;
    int frames = 256;
    std::vector<int> token_ids;
    std::string context;
    std::string language;
    int max_new_tokens = 0;
    bool generate_from_features = false;
    bool use_vulkan = false;
    int threads = 4;
};

static void print_usage(const char* prog) {
    std::cout
        << "Usage: " << prog << " [--model DIR] [--audio-wav FILE | --audio-features-raw FILE]\n"
        << "                 [--mel-bins N --frames N]\n"
        << "                 [--tokens comma,separated,ids] [--generate-from-features]\n"
        << "                 [--context TEXT] [--language NAME] [--max-new-tokens N]\n"
        << "                 [--threads N] [--vulkan]\n\n"
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
        } else if (arg == "--audio-wav") {
            need_value("--audio-wav");
            args.audio_wav = argv[++i];
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
        } else if (arg == "--context") {
            need_value("--context");
            args.context = argv[++i];
        } else if (arg == "--language") {
            need_value("--language");
            args.language = argv[++i];
        } else if (arg == "--max-new-tokens") {
            need_value("--max-new-tokens");
            args.max_new_tokens = std::stoi(argv[++i]);
        } else if (arg == "--generate-from-features") {
            args.generate_from_features = true;
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

    ncnn::Mat audio;
    if (!args.audio_wav.empty() && !args.audio_features_raw.empty()) {
        std::cerr << "Use only one of --audio-wav or --audio-features-raw\n";
        return 2;
    }

    if (!args.audio_wav.empty()) {
        std::vector<float> wav;
        int sample_rate = 0;
        if (!load_wav_pcm16(args.audio_wav, wav, sample_rate)) {
            std::cerr << "Failed to read PCM16 WAV: " << args.audio_wav << "\n";
            return 2;
        }
        if (sample_rate != 16000) {
            std::cerr << "Only 16 kHz WAV is supported for now, got " << sample_rate << "\n";
            return 3;
        }
        ncnn::Mat mel = wav_to_whisper_log_mel(wav, sample_rate, args.mel_bins, args.frames);
        audio = asr.run_audio_encoder(mel);
        if (audio.total() == 0) {
            return 4;
        }
        print_mat_shape("audio_encoder", audio);
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
        audio = asr.run_audio_encoder(mel);
        if (audio.total() == 0) {
            return 4;
        }
        print_mat_shape("audio_encoder", audio);
    }

    if (args.generate_from_features) {
        if (audio.total() == 0) {
            std::cerr << "--generate-from-features requires --audio-features-raw\n";
            return 8;
        }
        std::vector<int> prompt_ids = asr.build_prompt_ids(audio.h, args.context, args.language);
        if (prompt_ids.empty()) {
            std::cerr << "Failed to build Qwen3-ASR prompt ids\n";
            return 9;
        }
        if ((int)prompt_ids.size() >= asr.text_seq_len()) {
            std::cerr << "Prompt length " << prompt_ids.size()
                      << " leaves no decode room in static text_seq_len "
                      << asr.text_seq_len() << "\n";
            return 10;
        }

        std::cout << "prompt_len=" << prompt_ids.size()
                  << " audio_tokens=" << audio.h
                  << " text_seq_len=" << asr.text_seq_len()
                  << "\n";

        std::vector<int> generated;
        std::vector<int> running_ids = prompt_ids;
        int decode_steps = args.max_new_tokens > 0 ? args.max_new_tokens : 1;
        for (int i = 0; i < decode_steps && (int)running_ids.size() < asr.text_seq_len(); i++) {
            int next = asr.decode_next_token(running_ids, audio);
            if (next < 0) {
                return 11;
            }
            generated.push_back(next);
            running_ids.push_back(next);
            std::cout << "generated_token[" << i << "]=" << next << "\n";
        }
        std::cout << "generated_text=" << asr.decode(generated, true) << "\n";
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

    if (args.audio_wav.empty() && args.audio_features_raw.empty() && args.token_ids.empty()) {
        std::cout << "Model loaded. Provide --audio-features-raw or --tokens to run a smoke test.\n";
    }

    return 0;
}
