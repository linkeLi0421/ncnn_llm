#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "ncnn_qwen3_asr.h"
#include "utils/audio_utils.h"

using nlohmann::json;

struct Args {
    std::string model_path = "./assets/qwen3_asr_0.6b";
    std::string audio_features_raw;
    std::string audio_wav;
    std::string dump_mel_raw;
    std::string dump_mel_summary;
    std::string text_out;
    std::string json_out;
    int mel_bins = 128;
    int frames = 256;
    std::vector<int> token_ids;
    std::string context;
    std::string language;
    int max_new_tokens = 0;
    int chunk_overlap_frames = 32;
    bool generate_from_features = false;
    bool use_kv_cache = false;
    bool energy_chunking = false;
    bool timing = false;
    bool use_vulkan = false;
    int threads = 4;
};

static void print_usage(const char* prog) {
    std::cout
        << "Usage: " << prog << " [--model DIR] [--audio-wav FILE | --audio-features-raw FILE]\n"
        << "                 [--mel-bins N --frames N]\n"
        << "                 [--tokens comma,separated,ids] [--generate-from-features]\n"
        << "                 [--dump-mel-raw FILE] [--dump-mel-summary FILE]\n"
        << "                 [--text-out FILE] [--json-out FILE]\n"
        << "                 [--context TEXT] [--language NAME] [--max-new-tokens N]\n"
        << "                 [--chunk-overlap-frames N] [--energy-chunking]\n"
        << "                 [--use-kv-cache] [--timing]\n"
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
        } else if (arg == "--dump-mel-summary") {
            need_value("--dump-mel-summary");
            args.dump_mel_summary = argv[++i];
        } else if (arg == "--text-out") {
            need_value("--text-out");
            args.text_out = argv[++i];
        } else if (arg == "--json-out") {
            need_value("--json-out");
            args.json_out = argv[++i];
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
        } else if (arg == "--energy-chunking") {
            args.energy_chunking = true;
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
        if (ch == '.' || ch == '!' || ch == '?') {
            text.pop_back();
            text = trim_copy(text);
            continue;
        }
        if (text.size() >= 3) {
            std::string tail = text.substr(text.size() - 3);
            if (tail == "。" || tail == "！" || tail == "？") {
                text.resize(text.size() - 3);
                text = trim_copy(text);
                continue;
            }
        }
        break;
    }
    return text;
}

static std::vector<std::string> utf8_units(const std::string& text) {
    std::vector<std::string> units;
    for (size_t i = 0; i < text.size();) {
        unsigned char ch = (unsigned char)text[i];
        size_t len = 1;
        if ((ch & 0x80) == 0) {
            len = 1;
        } else if ((ch & 0xE0) == 0xC0) {
            len = 2;
        } else if ((ch & 0xF0) == 0xE0) {
            len = 3;
        } else if ((ch & 0xF8) == 0xF0) {
            len = 4;
        }
        if (i + len > text.size()) {
            len = 1;
        }
        units.push_back(text.substr(i, len));
        i += len;
    }
    return units;
}

static bool is_ignorable_overlap_unit(const std::string& unit) {
    if (unit.empty()) {
        return true;
    }
    if (unit.size() == 1) {
        unsigned char ch = (unsigned char)unit[0];
        return std::isspace(ch) || std::ispunct(ch);
    }
    return unit == "。" || unit == "，" || unit == "、" || unit == "；" ||
           unit == "：" || unit == "！" || unit == "？" || unit == "（" ||
           unit == "）" || unit == "“" || unit == "”" || unit == "《" ||
           unit == "》";
}

static std::string normalize_overlap_unit(const std::string& unit) {
    if (is_ignorable_overlap_unit(unit)) {
        return {};
    }
    if (unit.size() == 1) {
        unsigned char ch = (unsigned char)unit[0];
        if (std::isalnum(ch)) {
            return std::string(1, (char)std::tolower(ch));
        }
    }
    return unit;
}

static std::vector<std::string> normalized_overlap_units(const std::string& text) {
    std::vector<std::string> out;
    for (const std::string& unit : utf8_units(text)) {
        std::string normalized = normalize_overlap_unit(unit);
        if (!normalized.empty()) {
            out.push_back(normalized);
        }
    }
    return out;
}

static size_t overlap_unit_count(const std::string& left, const std::string& right) {
    std::vector<std::string> left_units = normalized_overlap_units(left);
    std::vector<std::string> right_units = normalized_overlap_units(right);
    size_t max_overlap = std::min<size_t>(16, std::min(left_units.size(), right_units.size()));
    for (size_t count = max_overlap; count > 0; count--) {
        if (count == 1 && left_units[left_units.size() - 1].size() == 1) {
            unsigned char ch = (unsigned char)left_units[left_units.size() - 1][0];
            if (std::isalnum(ch)) {
                continue;
            }
        }
        bool same = true;
        for (size_t i = 0; i < count; i++) {
            if (left_units[left_units.size() - count + i] != right_units[i]) {
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

static std::string drop_leading_overlap_units(const std::string& text, size_t count) {
    if (count == 0) {
        return trim_copy(text);
    }
    std::vector<std::string> units = utf8_units(text);
    size_t seen = 0;
    size_t byte_pos = 0;
    for (const std::string& unit : units) {
        byte_pos += unit.size();
        if (!normalize_overlap_unit(unit).empty()) {
            seen++;
            if (seen >= count) {
                break;
            }
        }
    }
    std::string out = byte_pos < text.size() ? text.substr(byte_pos) : std::string();
    out = trim_copy(out);
    while (!out.empty()) {
        std::vector<std::string> out_units = utf8_units(out);
        if (out_units.empty() || !is_ignorable_overlap_unit(out_units.front())) {
            break;
        }
        out = trim_copy(out.substr(out_units.front().size()));
    }
    return out;
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

static bool starts_with_non_ascii(const std::string& text) {
    std::string trimmed = trim_copy(text);
    return !trimmed.empty() && ((unsigned char)trimmed[0] & 0x80) != 0;
}

static bool ends_with_non_ascii(const std::string& text) {
    std::string trimmed = trim_copy(text);
    if (trimmed.empty()) {
        return false;
    }
    size_t pos = trimmed.size() - 1;
    while (pos > 0 && ((unsigned char)trimmed[pos] & 0xC0) == 0x80) {
        pos--;
    }
    return ((unsigned char)trimmed[pos] & 0x80) != 0;
}

static double rms_energy(const std::vector<float>& samples, size_t start, size_t end) {
    if (start >= end || start >= samples.size()) {
        return 0.0;
    }
    end = std::min(end, samples.size());
    double sum_sq = 0.0;
    for (size_t i = start; i < end; i++) {
        sum_sq += (double)samples[i] * (double)samples[i];
    }
    return std::sqrt(sum_sq / (double)(end - start));
}

static size_t find_low_energy_boundary(const std::vector<float>& samples,
                                       size_t search_start,
                                       size_t search_end,
                                       int sample_rate) {
    search_start = std::min(search_start, samples.size());
    search_end = std::min(search_end, samples.size());
    if (search_start >= search_end) {
        return search_end;
    }

    const size_t frame = std::max<size_t>(1, (size_t)sample_rate / 20); // 50 ms
    const size_t step = std::max<size_t>(1, (size_t)sample_rate / 100); // 10 ms
    if (search_end - search_start <= frame) {
        return search_end;
    }

    double best_energy = std::numeric_limits<double>::infinity();
    size_t best_center = search_end;
    for (size_t pos = search_start; pos + frame <= search_end; pos += step) {
        const double e = rms_energy(samples, pos, pos + frame);
        if (e < best_energy) {
            best_energy = e;
            best_center = pos + frame / 2;
        }
    }
    return best_center;
}

static std::vector<std::pair<size_t, size_t>> build_energy_aligned_chunks(const std::vector<float>& wav,
                                                                          int sample_rate,
                                                                          const Args& args) {
    constexpr int hop = 160;
    const size_t chunk_samples = (size_t)args.frames * hop;
    const size_t overlap_samples = (size_t)args.chunk_overlap_frames * hop;
    const size_t min_chunk_samples = chunk_samples * 3 / 5;
    const size_t min_final_tail_samples = chunk_samples / 5;
    std::vector<std::pair<size_t, size_t>> ranges;

    for (size_t start = 0; start < wav.size();) {
        const size_t max_end = std::min(start + chunk_samples, wav.size());
        size_t end = max_end;
        if (max_end < wav.size() && max_end > start + min_chunk_samples) {
            end = find_low_energy_boundary(wav, start + min_chunk_samples, max_end, sample_rate);
            if (end <= start || end - start < min_chunk_samples / 2) {
                end = max_end;
            }
        }
        ranges.push_back({start, end});
        if (end >= wav.size()) {
            break;
        }
        if (wav.size() - end < min_final_tail_samples) {
            break;
        }
        size_t next = end > overlap_samples ? end - overlap_samples : 0;
        if (next <= start) {
            next = start + (chunk_samples - overlap_samples);
        }
        if (next + chunk_samples > wav.size() && wav.size() > chunk_samples) {
            next = wav.size() - chunk_samples;
        }
        if (next <= start) {
            break;
        }
        start = next;
    }

    return ranges;
}

static std::vector<std::pair<size_t, size_t>> build_fixed_overlap_chunks(const std::vector<float>& wav,
                                                                         const Args& args) {
    constexpr int hop = 160;
    const size_t chunk_samples = (size_t)args.frames * hop;
    const size_t stride_samples = (size_t)(args.frames - args.chunk_overlap_frames) * hop;
    const size_t min_final_tail_samples = chunk_samples / 5;
    std::vector<std::pair<size_t, size_t>> ranges;

    for (size_t start = 0; start < wav.size();) {
        size_t end = std::min(start + chunk_samples, wav.size());
        ranges.push_back({start, end});
        if (end >= wav.size()) {
            break;
        }
        size_t next = start + stride_samples;
        if (next + chunk_samples > wav.size()) {
            if (wav.size() - end < min_final_tail_samples) {
                break;
            }
            next = wav.size() > chunk_samples ? wav.size() - chunk_samples : wav.size();
        }
        if (next <= start) {
            break;
        }
        start = next;
    }

    return ranges;
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

static bool write_text_file(const std::string& path, const std::string& text) {
    std::ofstream f(path);
    if (!f) {
        return false;
    }
    f << text;
    if (text.empty() || text.back() != '\n') {
        f << "\n";
    }
    return (bool)f;
}

static bool write_json_file(const std::string& path, const json& value) {
    std::ofstream f(path);
    if (!f) {
        return false;
    }
    f << value.dump(2) << "\n";
    return (bool)f;
}

static json mat_summary_json(const ncnn::Mat& mat, int first_value_count = 12) {
    json out = {
        {"dims", mat.dims},
        {"w", mat.w},
        {"h", mat.h},
        {"d", mat.d},
        {"c", mat.c},
        {"elemsize", mat.elemsize},
        {"elempack", mat.elempack},
        {"total", mat.total()},
        {"dtype", "float32"}
    };
    if (mat.total() == 0) {
        out["min"] = nullptr;
        out["max"] = nullptr;
        out["mean"] = nullptr;
        out["std"] = nullptr;
        out["first_values"] = json::array();
        return out;
    }

    double sum = 0.0;
    double sum_sq = 0.0;
    float min_v = std::numeric_limits<float>::infinity();
    float max_v = -std::numeric_limits<float>::infinity();
    json first_values = json::array();
    size_t seen = 0;

    const float* data = reinterpret_cast<const float*>(mat.data);
    const size_t total = mat.total();
    for (size_t i = 0; i < total; i++) {
        const float v = data[i];
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
        sum += (double)v;
        sum_sq += (double)v * (double)v;
        if ((int)first_values.size() < first_value_count) {
            first_values.push_back(v);
        }
        seen++;
    }

    const double mean = sum / (double)seen;
    const double variance = std::max(0.0, sum_sq / (double)seen - mean * mean);
    out["min"] = min_v;
    out["max"] = max_v;
    out["mean"] = mean;
    out["std"] = std::sqrt(variance);
    out["first_values"] = first_values;
    return out;
}

static json mel_summary_json(const ncnn::Mat& mel,
                             const Args& args,
                             const std::string& source,
                             int sample_rate,
                             size_t input_samples,
                             int chunk_index = -1,
                             size_t chunk_start_sample = 0,
                             size_t chunk_end_sample = 0) {
    json out = mat_summary_json(mel);
    out["source"] = source;
    out["sample_rate"] = sample_rate;
    out["channels_after_load"] = 1;
    out["input_samples"] = input_samples;
    out["input_duration_sec"] = sample_rate > 0 ? (double)input_samples / (double)sample_rate : 0.0;
    out["mel_bins"] = args.mel_bins;
    out["frames"] = args.frames;
    out["n_fft"] = 400;
    out["hop_length"] = 160;
    out["padding"] = "reflect left edge and zero after end";
    out["truncation"] = "fixed frame count";
    out["chunk_overlap_frames"] = args.chunk_overlap_frames;
    if (chunk_index >= 0) {
        out["chunk_index"] = chunk_index;
        out["chunk_start_sample"] = chunk_start_sample;
        out["chunk_end_sample"] = chunk_end_sample;
        out["chunk_samples"] = chunk_end_sample - chunk_start_sample;
    }
    return out;
}

static json first_step_debug_json(const Qwen3ASRFirstStepDebug& debug) {
    if (debug.prompt_len <= 0 || debug.selected_logits.total() == 0) {
        return {{"available", false}};
    }
    return {
        {"available", true},
        {"prompt_len", debug.prompt_len},
        {"next_token", debug.next_token},
        {"text_embeds", mat_summary_json(debug.text_embeds, 0)},
        {"merged_embeds", mat_summary_json(debug.merged_embeds, 0)},
        {"hidden", mat_summary_json(debug.hidden, 0)},
        {"logits", mat_summary_json(debug.logits, 0)},
        {"selected_logits", mat_summary_json(debug.selected_logits, 12)}
    };
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
            } else {
                size_t repeated_units = overlap_unit_count(out, part);
                if (repeated_units > 0) {
                    part = drop_leading_overlap_units(part, repeated_units);
                    if (part.empty()) {
                        continue;
                    }
                }
            }
        }
        if (!out.empty()) {
            char last = out.back();
            if (last != ' ' && last != '\n' && !ends_with_non_ascii(out) && !starts_with_non_ascii(part)) {
                out.push_back(' ');
            }
            if (part.size() >= 2 && part[0] >= 'A' && part[0] <= 'Z' && part[1] != '.') {
                part[0] = (char)std::tolower((unsigned char)part[0]);
            }
        }
        out += part;
    }
    return out;
}

int main(int argc, char** argv) {
    const auto program_start = std::chrono::steady_clock::now();
    Args args = parse_args(argc, argv);

    json run_report = {
        {"model", args.model_path},
        {"audio_wav", args.audio_wav},
        {"audio_features_raw", args.audio_features_raw},
        {"mel_bins", args.mel_bins},
        {"frames", args.frames},
        {"generate_from_features", args.generate_from_features},
        {"use_kv_cache_requested", args.use_kv_cache},
        {"threads", args.threads},
        {"vulkan", args.use_vulkan},
        {"chunks", json::array()}
    };
    json mel_summaries = json::array();
    std::string final_language;
    std::string final_text;

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
        const double audio_duration_sec = (double)wav.size() / (double)sample_rate;
        run_report["sample_rate"] = sample_rate;
        run_report["audio_samples"] = wav.size();
        run_report["audio_duration_sec"] = audio_duration_sec;
        run_report["audio_format"] = "PCM16 WAV";
        run_report["mono_conversion"] = "average channels to mono";

        if (args.generate_from_features) {
            constexpr int hop = 160;
            if (args.chunk_overlap_frames < 0 || args.chunk_overlap_frames >= args.frames) {
                std::cerr << "--chunk-overlap-frames must be >= 0 and smaller than --frames\n";
                return 3;
            }
            std::vector<std::pair<size_t, size_t>> chunk_ranges = args.energy_chunking
                ? build_energy_aligned_chunks(wav, sample_rate, args)
                : build_fixed_overlap_chunks(wav, args);
            run_report["chunking_strategy"] = args.energy_chunking
                ? "energy_aligned_overlap"
                : "fixed_overlap_tail_aligned";
            run_report["chunk_count"] = chunk_ranges.size();
            run_report["chunk_samples"] = (size_t)args.frames * hop;
            run_report["chunk_overlap_samples"] = (size_t)args.chunk_overlap_frames * hop;
            run_report["energy_chunking"] = args.energy_chunking;
            std::vector<std::string> chunk_texts;
            std::string language;
            int chunk_index = 0;
            for (const auto& range : chunk_ranges) {
                size_t start = range.first;
                size_t end = range.second;
                std::vector<float> chunk(wav.begin() + (std::vector<float>::difference_type)start,
                                         wav.begin() + (std::vector<float>::difference_type)end);
                ncnn::Mat mel = wav_to_whisper_log_mel(chunk, sample_rate, args.mel_bins, args.frames);
                if (!args.dump_mel_summary.empty()) {
                    mel_summaries.push_back(mel_summary_json(mel, args, args.audio_wav, sample_rate,
                                                            wav.size(), chunk_index, start, end));
                }
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
                std::vector<int> debug_prompt_ids = asr.build_prompt_ids(audio.h, args.context, args.language);
                Qwen3ASRFirstStepDebug first_step_debug = asr.debug_first_step(debug_prompt_ids, audio);
                Qwen3ASRResult result = decode_audio(asr, audio, args, prefix);
                run_report["chunks"].push_back({
                    {"index", chunk_index},
                    {"start_sample", start},
                    {"end_sample", end},
                    {"language", result.language},
                    {"raw_text", result.raw_text},
                    {"text", result.text},
                    {"audio_embedding", mat_summary_json(audio, 0)},
                    {"first_step", first_step_debug_json(first_step_debug)}
                });
                if (!result.text.empty()) {
                    chunk_texts.push_back(result.text);
                }
                if (language.empty() && !result.language.empty()) {
                    language = result.language;
                }
                chunk_index++;
            }
            final_language = language;
            final_text = join_text(chunk_texts);
            std::cout << "language=" << final_language << "\n";
            std::cout << "text=" << final_text << "\n";
            run_report["language"] = final_language;
            run_report["text"] = final_text;
            const double total_time_ms = elapsed_ms(program_start, std::chrono::steady_clock::now());
            run_report["total_time_ms"] = total_time_ms;
            if (audio_duration_sec > 0.0) {
                run_report["rtf"] = total_time_ms / 1000.0 / audio_duration_sec;
            } else {
                run_report["rtf"] = nullptr;
            }
            if (!args.text_out.empty() && !write_text_file(args.text_out, final_text)) {
                std::cerr << "Failed to write text output: " << args.text_out << "\n";
                return 9;
            }
            if (!args.json_out.empty() && !write_json_file(args.json_out, run_report)) {
                std::cerr << "Failed to write json output: " << args.json_out << "\n";
                return 9;
            }
            if (!args.dump_mel_summary.empty() && !write_json_file(args.dump_mel_summary, mel_summaries)) {
                std::cerr << "Failed to write mel summary: " << args.dump_mel_summary << "\n";
                return 9;
            }
            return 0;
        }

        ncnn::Mat mel = wav_to_whisper_log_mel(wav, sample_rate, args.mel_bins, args.frames);
        if (!args.dump_mel_summary.empty()) {
            mel_summaries.push_back(mel_summary_json(mel, args, args.audio_wav, sample_rate, wav.size()));
        }
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
        run_report["audio_embedding"] = mat_summary_json(audio, 0);
        std::vector<int> debug_prompt_ids = asr.build_prompt_ids(audio.h, args.context, args.language);
        run_report["first_step"] = first_step_debug_json(asr.debug_first_step(debug_prompt_ids, audio));
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
        if (!args.dump_mel_summary.empty()) {
            mel_summaries.push_back(mel_summary_json(mel, args, args.audio_features_raw, 0, features.size()));
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
        run_report["audio_embedding"] = mat_summary_json(audio, 0);
        std::vector<int> debug_prompt_ids = asr.build_prompt_ids(audio.h, args.context, args.language);
        run_report["first_step"] = first_step_debug_json(asr.debug_first_step(debug_prompt_ids, audio));
    }

    if (args.generate_from_features) {
        if (audio.total() == 0) {
            std::cerr << "--generate-from-features requires --audio-features-raw\n";
            return 8;
        }
        Qwen3ASRResult result = decode_audio(asr, audio, args, "");
        final_language = result.language;
        final_text = result.text;
        run_report["language"] = final_language;
        run_report["raw_text"] = result.raw_text;
        run_report["text"] = final_text;
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
        run_report["text_embed"] = mat_summary_json(embeds, 0);
        run_report["text_backbone"] = mat_summary_json(hidden, 0);
        run_report["lm_head"] = mat_summary_json(logits, 0);
        run_report["next_token"] = asr.select_next_token_from_logits(logits);
    }

    if (args.audio_wav.empty() && args.audio_features_raw.empty() && args.token_ids.empty()) {
        std::cout << "Model loaded. Provide --audio-features-raw or --tokens to run a smoke test.\n";
    }

    if (!args.text_out.empty() && !write_text_file(args.text_out, final_text)) {
        std::cerr << "Failed to write text output: " << args.text_out << "\n";
        return 9;
    }
    const double total_time_ms = elapsed_ms(program_start, std::chrono::steady_clock::now());
    run_report["total_time_ms"] = total_time_ms;
    if (run_report.contains("audio_duration_sec") && run_report["audio_duration_sec"].is_number()) {
        const double audio_duration_sec = run_report["audio_duration_sec"].get<double>();
        if (audio_duration_sec > 0.0) {
            run_report["rtf"] = total_time_ms / 1000.0 / audio_duration_sec;
        } else {
            run_report["rtf"] = nullptr;
        }
    }
    if (!args.json_out.empty() && !write_json_file(args.json_out, run_report)) {
        std::cerr << "Failed to write json output: " << args.json_out << "\n";
        return 9;
    }
    if (!args.dump_mel_summary.empty() && !write_json_file(args.dump_mel_summary, mel_summaries)) {
        std::cerr << "Failed to write mel summary: " << args.dump_mel_summary << "\n";
        return 9;
    }

    return 0;
}
