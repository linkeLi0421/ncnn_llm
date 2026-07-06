#include <algorithm>
#include <cctype>
#include <chrono>
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
    std::string dump_mel_raw;
    int mel_bins = 128;
    int frames = 256;
    std::vector<int> token_ids;
    std::string context;
    std::string language;
    int max_new_tokens = 0;
    int chunk_overlap_frames = 32;
    bool generate_from_features = false;
    bool use_kv_cache = false;
    bool timing = false;
    bool use_vulkan = false;
    int threads = 4;
};

static void print_usage(const char* prog) {
    std::cout
        << "Usage: " << prog << " [--model DIR] [--audio-wav FILE | --audio-features-raw FILE]\n"
        << "                 [--mel-bins N --frames N]\n"
        << "                 [--tokens comma,separated,ids] [--generate-from-features]\n"
        << "                 [--dump-mel-raw FILE]\n"
        << "                 [--context TEXT] [--language NAME] [--max-new-tokens N]\n"
        << "                 [--chunk-overlap-frames N] [--use-kv-cache] [--timing]\n"
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
        } else if (arg == "--dump-mel-raw") {
            need_value("--dump-mel-raw");
            args.dump_mel_raw = argv[++i];
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
        } else if (arg == "--chunk-overlap-frames") {
            need_value("--chunk-overlap-frames");
            args.chunk_overlap_frames = std::stoi(argv[++i]);
        } else if (arg == "--generate-from-features") {
            args.generate_from_features = true;
        } else if (arg == "--use-kv-cache") {
            args.use_kv_cache = true;
        } else if (arg == "--timing") {
            args.timing = true;
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

static std::string trim_copy(const std::string& text) {
    size_t first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    size_t last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

static std::string strip_final_sentence_punctuation(std::string text) {
    text = trim_copy(text);
    while (!text.empty()) {
        char ch = text.back();
        if (ch != '.' && ch != '!' && ch != '?') {
            break;
        }
        text.pop_back();
        text = trim_copy(text);
    }
    return text;
}

static std::string normalize_word(const std::string& word) {
    std::string out;
    for (unsigned char ch : word) {
        if (std::isalnum(ch)) {
            out.push_back((char)std::tolower(ch));
        }
    }
    return out;
}

static std::vector<std::string> split_words(const std::string& text) {
    std::vector<std::string> words;
    std::stringstream ss(text);
    std::string word;
    while (ss >> word) {
        std::string normalized = normalize_word(word);
        if (!normalized.empty()) {
            words.push_back(normalized);
        }
    }
    return words;
}

static std::string drop_leading_words(const std::string& text, size_t count) {
    if (count == 0) {
        return trim_copy(text);
    }
    size_t pos = 0;
    size_t dropped = 0;
    while (dropped < count) {
        pos = text.find_first_not_of(" \t\r\n", pos);
        if (pos == std::string::npos) {
            return {};
        }
        pos = text.find_first_of(" \t\r\n", pos);
        dropped++;
        if (pos == std::string::npos) {
            return {};
        }
    }
    return trim_copy(text.substr(pos));
}

static size_t overlap_word_count(const std::string& left, const std::string& right) {
    std::vector<std::string> left_words = split_words(left);
    std::vector<std::string> right_words = split_words(right);
    size_t max_overlap = std::min<size_t>(8, std::min(left_words.size(), right_words.size()));
    for (size_t count = max_overlap; count > 0; count--) {
        bool same = true;
        for (size_t i = 0; i < count; i++) {
            if (left_words[left_words.size() - count + i] != right_words[i]) {
                same = false;
                break;
            }
        }
        if (same) {
            return count;
        }
    }
    return 0;
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

static bool write_mat_raw(const std::string& path, const ncnn::Mat& mat) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    for (int y = 0; y < mat.h; y++) {
        const float* row = mat.row(y);
        f.write(reinterpret_cast<const char*>(row), (std::streamsize)mat.w * (std::streamsize)sizeof(float));
    }
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

static double elapsed_ms(std::chrono::steady_clock::time_point start,
                         std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static Qwen3ASRResult decode_audio(ncnn_qwen3_asr& asr,
                                   const ncnn::Mat& audio,
                                   const Args& args,
                                   const std::string& prefix) {
    const auto decode_start = std::chrono::steady_clock::now();
    std::vector<int> prompt_ids = asr.build_prompt_ids(audio.h, args.context, args.language);
    if (prompt_ids.empty()) {
        std::cerr << "Failed to build Qwen3-ASR prompt ids\n";
        return {};
    }
    const bool use_kv = args.use_kv_cache && asr.has_kv_decoder();
    if ((!use_kv && (int)prompt_ids.size() >= asr.text_seq_len()) ||
        (use_kv && (int)prompt_ids.size() > asr.text_seq_len())) {
        std::cerr << "Prompt length " << prompt_ids.size()
                  << " leaves no decode room in static text_seq_len "
                  << asr.text_seq_len() << "\n";
        return {};
    }

    std::cout << prefix << "prompt_len=" << prompt_ids.size()
              << " audio_tokens=" << audio.h
              << " text_seq_len=" << asr.text_seq_len()
              << " kv_decoder=" << (use_kv ? "true" : "false")
              << "\n";

    std::vector<int> generated;
    int decode_steps = args.max_new_tokens > 0 ? args.max_new_tokens : 1;

    if (use_kv) {
        Qwen3ASRKVDecodeState state;
        const auto prefill_start = std::chrono::steady_clock::now();
        int next = asr.prefill_kv(prompt_ids, audio, state);
        const auto prefill_end = std::chrono::steady_clock::now();
        if (next < 0) {
            return {};
        }
        if (args.timing) {
            std::cout << prefix << "prefill_time_ms=" << elapsed_ms(prefill_start, prefill_end) << "\n";
        }
        for (int i = 0; i < decode_steps; i++) {
            if (asr.should_stop_token(next)) {
                std::cout << prefix << "stop_token=" << next << "\n";
                break;
            }
            generated.push_back(next);
            std::cout << prefix << "generated_token[" << i << "]=" << next << "\n";
            if (i + 1 >= decode_steps) {
                break;
            }
            const auto step_start = std::chrono::steady_clock::now();
            next = asr.decode_next_token_kv(next, state);
            const auto step_end = std::chrono::steady_clock::now();
            if (next < 0) {
                return {};
            }
            if (args.timing) {
                std::cout << prefix << "decode_step_time_ms[" << (i + 1) << "]="
                          << elapsed_ms(step_start, step_end) << "\n";
            }
        }
    } else {
        std::vector<int> running_ids = prompt_ids;
        for (int i = 0; i < decode_steps && (int)running_ids.size() < asr.text_seq_len(); i++) {
            const auto step_start = std::chrono::steady_clock::now();
            int next = asr.decode_next_token(running_ids, audio);
            const auto step_end = std::chrono::steady_clock::now();
            if (next < 0) {
                return {};
            }
            if (args.timing) {
                std::cout << prefix << "decode_step_time_ms[" << i << "]="
                          << elapsed_ms(step_start, step_end) << "\n";
            }
            if (asr.should_stop_token(next)) {
                std::cout << prefix << "stop_token=" << next << "\n";
                break;
            }
            generated.push_back(next);
            running_ids.push_back(next);
            std::cout << prefix << "generated_token[" << i << "]=" << next << "\n";
        }
    }

    Qwen3ASRResult result = asr.parse_output(generated);
    std::cout << prefix << "generated_raw=" << result.raw_text << "\n";
    std::cout << prefix << "language=" << result.language << "\n";
    std::cout << prefix << "text=" << result.text << "\n";
    const auto decode_end = std::chrono::steady_clock::now();
    if (args.timing) {
        std::cout << prefix << "decode_total_time_ms=" << elapsed_ms(decode_start, decode_end) << "\n";
    }
    return result;
}

static std::string join_text(const std::vector<std::string>& parts) {
    std::string out;
    for (size_t i = 0; i < parts.size(); i++) {
        std::string part = trim_copy(parts[i]);
        if (part.empty()) {
            continue;
        }
        if (i + 1 < parts.size()) {
            part = strip_final_sentence_punctuation(part);
        }
        if (!out.empty()) {
            size_t repeated_words = overlap_word_count(out, part);
            if (repeated_words > 0) {
                part = drop_leading_words(part, repeated_words);
                if (part.empty()) {
                    continue;
                }
            }
        }
        if (!out.empty()) {
            char last = out.back();
            if (last != ' ' && last != '\n') {
                out.push_back(' ');
            }
            if (!part.empty() && part[0] >= 'A' && part[0] <= 'Z') {
                part[0] = (char)std::tolower((unsigned char)part[0]);
            }
        }
        out += part;
    }
    return out;
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

        if (args.generate_from_features) {
            constexpr int hop = 160;
            if (args.chunk_overlap_frames < 0 || args.chunk_overlap_frames >= args.frames) {
                std::cerr << "--chunk-overlap-frames must be >= 0 and smaller than --frames\n";
                return 3;
            }
            const size_t chunk_samples = (size_t)args.frames * hop;
            const size_t stride_samples = (size_t)(args.frames - args.chunk_overlap_frames) * hop;
            const size_t min_tail_samples = chunk_samples / 2;
            std::vector<std::string> chunk_texts;
            std::string language;
            int chunk_index = 0;
            for (size_t start = 0; start < wav.size(); start += stride_samples) {
                if (start > 0 && wav.size() - start < min_tail_samples) {
                    break;
                }
                size_t end = std::min(start + chunk_samples, wav.size());
                std::vector<float> chunk(wav.begin() + (std::vector<float>::difference_type)start,
                                         wav.begin() + (std::vector<float>::difference_type)end);
                ncnn::Mat mel = wav_to_whisper_log_mel(chunk, sample_rate, args.mel_bins, args.frames);
                if (chunk_index == 0 && !args.dump_mel_raw.empty() && !write_mat_raw(args.dump_mel_raw, mel)) {
                    std::cerr << "Failed to write mel dump: " << args.dump_mel_raw << "\n";
                    return 4;
                }
                const auto audio_start = std::chrono::steady_clock::now();
                audio = asr.run_audio_encoder(mel);
                const auto audio_end = std::chrono::steady_clock::now();
                if (audio.total() == 0) {
                    return 4;
                }
                std::string prefix = "chunk[" + std::to_string(chunk_index) + "]_";
                if (args.timing) {
                    std::cout << prefix << "audio_encoder_time_ms=" << elapsed_ms(audio_start, audio_end) << "\n";
                }
                print_mat_shape((prefix + "audio_encoder").c_str(), audio);
                Qwen3ASRResult result = decode_audio(asr, audio, args, prefix);
                if (!result.text.empty()) {
                    chunk_texts.push_back(result.text);
                }
                if (language.empty() && !result.language.empty()) {
                    language = result.language;
                }
                chunk_index++;
            }
            std::cout << "language=" << language << "\n";
            std::cout << "text=" << join_text(chunk_texts) << "\n";
            return 0;
        }

        ncnn::Mat mel = wav_to_whisper_log_mel(wav, sample_rate, args.mel_bins, args.frames);
        if (!args.dump_mel_raw.empty() && !write_mat_raw(args.dump_mel_raw, mel)) {
            std::cerr << "Failed to write mel dump: " << args.dump_mel_raw << "\n";
            return 4;
        }
        const auto audio_start = std::chrono::steady_clock::now();
        audio = asr.run_audio_encoder(mel);
        const auto audio_end = std::chrono::steady_clock::now();
        if (audio.total() == 0) {
            return 4;
        }
        if (args.timing) {
            std::cout << "audio_encoder_time_ms=" << elapsed_ms(audio_start, audio_end) << "\n";
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
        const auto audio_start = std::chrono::steady_clock::now();
        audio = asr.run_audio_encoder(mel);
        const auto audio_end = std::chrono::steady_clock::now();
        if (audio.total() == 0) {
            return 4;
        }
        if (args.timing) {
            std::cout << "audio_encoder_time_ms=" << elapsed_ms(audio_start, audio_end) << "\n";
        }
        print_mat_shape("audio_encoder", audio);
    }

    if (args.generate_from_features) {
        if (audio.total() == 0) {
            std::cerr << "--generate-from-features requires --audio-features-raw\n";
            return 8;
        }
        decode_audio(asr, audio, args, "");
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
