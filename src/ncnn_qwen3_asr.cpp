#include "ncnn_qwen3_asr.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "utils/rope_embed.h"

ncnn_qwen3_asr::ncnn_qwen3_asr(const std::string& model_path, bool use_vulkan, int num_threads)
    : ncnn_llm_base(use_vulkan, num_threads > 0 ? num_threads : 4) {
    ok_ = load_model_file(model_path);
}

bool ncnn_qwen3_asr::load_model_file(const std::string& model_path) {
    try {
        json config;
        {
            std::ifstream ifs(model_path + "/model.json");
            if (!ifs) {
                throw std::runtime_error("failed to open model.json");
            }
            ifs >> config;
        }

        audio_encoder_net_ = std::make_shared<ncnn::Net>();
        text_embed_net_ = std::make_shared<ncnn::Net>();
        text_backbone_net_ = std::make_shared<ncnn::Net>();
        lm_head_net_ = std::make_shared<ncnn::Net>();

        configure_net(*audio_encoder_net_);
        configure_net(*text_embed_net_);
        configure_net(*text_backbone_net_);
        configure_net(*lm_head_net_);

        const auto& params = config["params"];
        const std::string audio_param = model_path + "/" + params["audio_encoder_param"].get<std::string>();
        const std::string audio_bin = model_path + "/" + params["audio_encoder_bin"].get<std::string>();
        const std::string text_embed_param = model_path + "/" + params["text_embed_param"].get<std::string>();
        const std::string text_embed_bin = model_path + "/" + params["text_embed_bin"].get<std::string>();
        const std::string text_backbone_param = model_path + "/" + params["text_backbone_param"].get<std::string>();
        const std::string text_backbone_bin = model_path + "/" + params["text_backbone_bin"].get<std::string>();
        const std::string text_prefill_kv_param =
            params.contains("text_prefill_kv_param") && !params["text_prefill_kv_param"].get<std::string>().empty()
                ? model_path + "/" + params["text_prefill_kv_param"].get<std::string>()
                : "";
        const std::string text_prefill_kv_bin =
            params.contains("text_prefill_kv_bin") && !params["text_prefill_kv_bin"].get<std::string>().empty()
                ? model_path + "/" + params["text_prefill_kv_bin"].get<std::string>()
                : "";
        const std::string text_decode_kv_param =
            params.contains("text_decode_kv_param") && !params["text_decode_kv_param"].get<std::string>().empty()
                ? model_path + "/" + params["text_decode_kv_param"].get<std::string>()
                : "";
        const std::string text_decode_kv_bin =
            params.contains("text_decode_kv_bin") && !params["text_decode_kv_bin"].get<std::string>().empty()
                ? model_path + "/" + params["text_decode_kv_bin"].get<std::string>()
                : "";
        const std::string lm_head_param = model_path + "/" + params["lm_head_param"].get<std::string>();
        const std::string lm_head_bin = model_path + "/" + params["lm_head_bin"].get<std::string>();

        printf("Loading Qwen3-ASR model from %s\n", model_path.c_str());
        printf("  audio_encoder: %s / %s\n", audio_param.c_str(), audio_bin.c_str());
        printf("  text_embed:    %s / %s\n", text_embed_param.c_str(), text_embed_bin.c_str());
        printf("  text_backbone: %s / %s\n", text_backbone_param.c_str(), text_backbone_bin.c_str());
        if (!text_prefill_kv_param.empty() && !text_decode_kv_param.empty()) {
            printf("  text_prefill_kv: %s / %s\n", text_prefill_kv_param.c_str(), text_prefill_kv_bin.c_str());
            printf("  text_decode_kv:  %s / %s\n", text_decode_kv_param.c_str(), text_decode_kv_bin.c_str());
        }
        printf("  lm_head:       %s / %s\n", lm_head_param.c_str(), lm_head_bin.c_str());

        text_backbone_has_attention_mask_ = param_has_input_blob(text_backbone_param, "in1");

        if (!load_one_net(*audio_encoder_net_, audio_param, audio_bin)) return false;
        if (!load_one_net(*text_embed_net_, text_embed_param, text_embed_bin)) return false;
        if (!load_one_net(*text_backbone_net_, text_backbone_param, text_backbone_bin)) return false;
        if (!text_prefill_kv_param.empty() && !text_decode_kv_param.empty()) {
            text_prefill_kv_net_ = std::make_shared<ncnn::Net>();
            text_decode_kv_net_ = std::make_shared<ncnn::Net>();
            configure_net(*text_prefill_kv_net_);
            configure_net(*text_decode_kv_net_);
            if (!load_one_net(*text_prefill_kv_net_, text_prefill_kv_param, text_prefill_kv_bin)) return false;
            if (!load_one_net(*text_decode_kv_net_, text_decode_kv_param, text_decode_kv_bin)) return false;
            has_kv_decoder_ = true;
        }
        if (!load_one_net(*lm_head_net_, lm_head_param, lm_head_bin)) return false;

        if (config.contains("setting")) {
            const auto& setting = config["setting"];
            if (setting.contains("audio_token_id")) {
                audio_token_id_ = setting["audio_token_id"].get<int>();
            }
            if (setting.contains("audio_start_token_id")) {
                audio_start_token_id_ = setting["audio_start_token_id"].get<int>();
            }
            if (setting.contains("audio_end_token_id")) {
                audio_end_token_id_ = setting["audio_end_token_id"].get<int>();
            }
            if (setting.contains("user_token_id")) {
                user_token_id_ = setting["user_token_id"].get<int>();
            }
            if (setting.contains("text_seq_len")) {
                text_seq_len_ = setting["text_seq_len"].get<int>();
            }
            if (setting.contains("kv_cache_len")) {
                kv_cache_len_ = setting["kv_cache_len"].get<int>();
            }
            if (setting.contains("text_config") && setting["text_config"].contains("hidden_size")) {
                hidden_size_ = setting["text_config"]["hidden_size"].get<int>();
            }
            if (setting.contains("text_config") && setting["text_config"].contains("vocab_size")) {
                vocab_size_ = setting["text_config"]["vocab_size"].get<int>();
            }
            if (setting.contains("text_config") && setting["text_config"].contains("num_hidden_layers")) {
                num_hidden_layers_ = setting["text_config"]["num_hidden_layers"].get<int>();
            }
            if (setting.contains("text_config")) {
                const auto& text_config = setting["text_config"];
                if (text_config.contains("head_dim")) {
                    rope_head_dim_ = text_config["head_dim"].get<int>();
                } else if (text_config.contains("hidden_size") && text_config.contains("num_attention_heads")) {
                    rope_head_dim_ = text_config["hidden_size"].get<int>() / text_config["num_attention_heads"].get<int>();
                }
                if (text_config.contains("rope_theta")) {
                    rope_theta_ = text_config["rope_theta"].get<float>();
                }
            }
        }

        if (config.contains("tokenizer")) {
            const auto& tok = config["tokenizer"];
            std::string tokenizer_type = "bpe";
            if (tok.contains("type")) {
                tokenizer_type = tok["type"].get<std::string>();
            }
            const std::string vocab_file = model_path + "/" + tok["vocab_file"].get<std::string>();
            const std::string merges_file = model_path + "/" + tok["merges_file"].get<std::string>();
            tokenizer_ = std::make_shared<BpeTokenizer>(BpeTokenizer::LoadFromFiles(
                vocab_file, merges_file, SpecialTokensConfig{}, false, true, tokenizer_type == "bbpe"
            ));
            if (tok.contains("additional_special_tokens")) {
                for (const auto& token : tok["additional_special_tokens"]) {
                    tokenizer_->AddAdditionalSpecialToken(token.get<std::string>(), true);
                }
            }
            if (audio_end_token_id_ < 0 && tok.contains("audio_eos_token")) {
                const std::string token = tok["audio_eos_token"].get<std::string>();
                auto it = tokenizer_->token_to_id().find(token);
                if (it != tokenizer_->token_to_id().end()) {
                    audio_end_token_id_ = it->second;
                }
            }
            if (vocab_size_ <= 0) {
                vocab_size_ = (int)tokenizer_->vocab_size();
            }
        }

        {
            std::ifstream ifs(model_path + "/processor/added_tokens.json");
            if (ifs) {
                json added_tokens;
                ifs >> added_tokens;
                if (added_tokens.contains("<asr_text>")) {
                    asr_text_token_id_ = added_tokens["<asr_text>"].get<int>();
                }
            }
        }

        printf("  hidden_size=%d vocab_size=%d text_seq_len=%d kv_cache_len=%d layers=%d rope_head_dim=%d rope_theta=%g\n",
               hidden_size_, vocab_size_, text_seq_len_, kv_cache_len_, num_hidden_layers_, rope_head_dim_, rope_theta_);
        printf("  audio_token_id=%d audio_start_token_id=%d audio_end_token_id=%d user_token_id=%d\n",
               audio_token_id_, audio_start_token_id_, audio_end_token_id_, user_token_id_);
        printf("  text_backbone_attention_mask=%s\n", text_backbone_has_attention_mask_ ? "true" : "false");
        printf("  text_kv_decoder=%s\n", has_kv_decoder_ ? "true" : "false");
        return true;
    } catch (const std::exception& e) {
        fprintf(stderr, "ncnn_qwen3_asr load failed: %s\n", e.what());
        return false;
    }
}

bool ncnn_qwen3_asr::load_one_net(ncnn::Net& net, const std::string& param_path, const std::string& bin_path) {
    if (net.load_param(param_path.c_str()) != 0) {
        fprintf(stderr, "failed to load param: %s\n", param_path.c_str());
        return false;
    }
    if (net.load_model(bin_path.c_str()) != 0) {
        fprintf(stderr, "failed to load bin: %s\n", bin_path.c_str());
        return false;
    }
    return true;
}

void ncnn_qwen3_asr::configure_net(ncnn::Net& net) {
    net.opt.num_threads = num_threads_;
    net.opt.use_vulkan_compute = use_vulkan_;
    net.opt.use_bf16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_packed = false;
}

bool ncnn_qwen3_asr::param_has_input_blob(const std::string& param_path, const std::string& blob_name) {
    std::ifstream ifs(param_path);
    std::string layer_type;
    std::string layer_name;
    while (ifs >> layer_type >> layer_name) {
        std::string rest;
        std::getline(ifs, rest);
        if (layer_type == "Input" && layer_name == blob_name) {
            return true;
        }
    }
    return false;
}

static int input_cache_blob(ncnn::Extractor& ex, int layer, bool key, const ncnn::Mat& value) {
    char numbered[16];
    std::snprintf(numbered, sizeof(numbered), "in%d", 3 + layer * 2 + (key ? 0 : 1));
    return ex.input(numbered, value);
}

static int extract_prefill_cache_blob(ncnn::Extractor& ex, int layer, bool key, ncnn::Mat& value) {
    char numbered[16];
    std::snprintf(numbered, sizeof(numbered), "out%d", 1 + layer * 2 + (key ? 0 : 1));
    return ex.extract(numbered, value);
}

static int extract_decode_cache_blob(ncnn::Extractor& ex, int layer, bool key, ncnn::Mat& value) {
    char numbered[16];
    std::snprintf(numbered, sizeof(numbered), "out%d", 1 + layer * 2 + (key ? 0 : 1));
    return ex.extract(numbered, value);
}

ncnn::Mat ncnn_qwen3_asr::run_audio_encoder(const ncnn::Mat& mel_features) const {
    ncnn::Mat out;
    ncnn::Extractor ex = audio_encoder_net_->create_extractor();
    if (ex.input("in0", mel_features) != 0) {
        fprintf(stderr, "Qwen3-ASR audio_encoder input failed\n");
        return out;
    }
    if (ex.extract("out0", out) != 0) {
        fprintf(stderr, "Qwen3-ASR audio_encoder extract failed\n");
    }
    return out;
}

ncnn::Mat ncnn_qwen3_asr::run_text_embed(const std::vector<int>& input_ids) const {
    ncnn::Mat input_ids_mat((int)input_ids.size(), 1, (void*)input_ids.data());
    input_ids_mat = input_ids_mat.clone();

    ncnn::Mat out;
    ncnn::Extractor ex = text_embed_net_->create_extractor();
    if (ex.input("in0", input_ids_mat) != 0) {
        fprintf(stderr, "Qwen3-ASR text_embed input failed\n");
        return out;
    }
    if (ex.extract("out0", out) != 0) {
        fprintf(stderr, "Qwen3-ASR text_embed extract failed\n");
    }
    return out;
}

ncnn::Mat ncnn_qwen3_asr::run_text_backbone(const ncnn::Mat& input_embeds,
                                            const std::vector<int>& attention_mask) const {
    ncnn::Mat mask_mat((int)attention_mask.size(), 1, (void*)attention_mask.data());
    mask_mat = mask_mat.clone();

    ncnn::Mat out;
    ncnn::Extractor ex = text_backbone_net_->create_extractor();
    if (ex.input("in0", input_embeds) != 0) {
        fprintf(stderr, "Qwen3-ASR text_backbone input embeds failed\n");
        return out;
    }
    if (text_backbone_has_attention_mask_) {
        if (ex.input("in1", mask_mat) != 0) {
            fprintf(stderr, "Qwen3-ASR text_backbone attention mask failed\n");
            return out;
        }
    }
    if (ex.extract("out0", out) != 0) {
        fprintf(stderr, "Qwen3-ASR text_backbone extract failed\n");
    }
    return out;
}

ncnn::Mat ncnn_qwen3_asr::run_text_prefill_kv(const ncnn::Mat& input_embeds,
                                               const std::vector<int>& attention_mask,
                                               Qwen3ASRKVDecodeState& state) const {
    ncnn::Mat out;
    if (!text_prefill_kv_net_ || num_hidden_layers_ <= 0) {
        return out;
    }

    ncnn::Extractor ex = text_prefill_kv_net_->create_extractor();
    if (ex.input("in0", input_embeds) != 0) {
        fprintf(stderr, "Qwen3-ASR text_prefill_kv input failed\n");
        return out;
    }

    state.kv_cache.clear();
    for (int i = 0; i < num_hidden_layers_; i++) {
        ncnn::Mat k_cache;
        ncnn::Mat v_cache;
        if (extract_prefill_cache_blob(ex, i, true, k_cache) != 0 ||
            extract_prefill_cache_blob(ex, i, false, v_cache) != 0) {
            fprintf(stderr, "Qwen3-ASR text_prefill_kv cache extract failed at layer %d\n", i);
            state.kv_cache.clear();
            return {};
        }
        state.kv_cache.emplace_back(std::move(k_cache), std::move(v_cache));
    }

    if (ex.extract("out0", out) != 0) {
        fprintf(stderr, "Qwen3-ASR text_prefill_kv hidden extract failed\n");
        state.kv_cache.clear();
        return {};
    }

    state.position = (int)attention_mask.size();
    state.ready = true;
    return out;
}

ncnn::Mat ncnn_qwen3_asr::run_text_decode_kv(const ncnn::Mat& input_embed,
                                             Qwen3ASRKVDecodeState& state) const {
    ncnn::Mat out;
    if (!text_decode_kv_net_ || !state.ready || (int)state.kv_cache.size() != num_hidden_layers_) {
        return out;
    }

    ncnn::Extractor ex = text_decode_kv_net_->create_extractor();
    ncnn::Mat cos_cache;
    ncnn::Mat sin_cache;
    generate_rope_embed_cache_full(1, rope_head_dim_ > 0 ? rope_head_dim_ : 128, state.position, cos_cache, sin_cache, rope_theta_);
    if (ex.input("in0", input_embed) != 0 || ex.input("in1", cos_cache) != 0 || ex.input("in2", sin_cache) != 0) {
        fprintf(stderr, "Qwen3-ASR text_decode_kv input failed\n");
        return out;
    }

    for (int i = 0; i < num_hidden_layers_; i++) {
        if (input_cache_blob(ex, i, true, state.kv_cache[i].first) != 0 ||
            input_cache_blob(ex, i, false, state.kv_cache[i].second) != 0) {
            fprintf(stderr, "Qwen3-ASR text_decode_kv cache input failed at layer %d\n", i);
            return {};
        }
    }

    for (int i = 0; i < num_hidden_layers_; i++) {
        ncnn::Mat k_cache;
        ncnn::Mat v_cache;
        if (extract_decode_cache_blob(ex, i, true, k_cache) != 0 ||
            extract_decode_cache_blob(ex, i, false, v_cache) != 0) {
            fprintf(stderr, "Qwen3-ASR text_decode_kv cache extract failed at layer %d\n", i);
            return {};
        }
        state.kv_cache[i] = std::make_pair(std::move(k_cache), std::move(v_cache));
    }

    if (ex.extract("out0", out) != 0) {
        fprintf(stderr, "Qwen3-ASR text_decode_kv hidden extract failed\n");
        return {};
    }

    state.position += 1;
    return out;
}

ncnn::Mat ncnn_qwen3_asr::run_lm_head(const ncnn::Mat& hidden_states) const {
    ncnn::Mat out;
    ncnn::Extractor ex = lm_head_net_->create_extractor();
    if (ex.input("in0", hidden_states) != 0) {
        fprintf(stderr, "Qwen3-ASR lm_head input failed\n");
        return out;
    }
    if (ex.extract("out0", out) != 0) {
        fprintf(stderr, "Qwen3-ASR lm_head extract failed\n");
    }
    return out;
}

int ncnn_qwen3_asr::select_next_token_from_logits(const ncnn::Mat& logits) const {
    if (logits.total() <= 0) {
        return -1;
    }

    const int width = logits.w;
    const float* row = logits.h > 1 ? logits.row(logits.h - 1) : (const float*)logits.data;
    const int limit = vocab_size_ > 0 ? std::min(vocab_size_, width) : width;
    int best = 0;
    float best_score = row[0];
    for (int i = 1; i < limit; i++) {
        if (row[i] > best_score) {
            best_score = row[i];
            best = i;
        }
    }
    return best;
}

bool ncnn_qwen3_asr::should_stop_token(int token_id) const {
    const int im_end = 151645;
    return token_id == im_end;
}

std::string ncnn_qwen3_asr::decode(const std::vector<int>& ids, bool skip_special_tokens) const {
    if (!tokenizer_) {
        return {};
    }
    return tokenizer_->decode(ids, skip_special_tokens);
}

Qwen3ASRResult ncnn_qwen3_asr::parse_output(const std::vector<int>& generated_ids) const {
    Qwen3ASRResult result;
    result.raw_text = decode(generated_ids, true);
    result.text = result.raw_text;

    const std::string prefix = "language ";
    if (result.text.rfind(prefix, 0) == 0) {
        const size_t after_prefix = prefix.size();
        size_t marker = result.text.find("<asr_text>", after_prefix);
        size_t text_start = std::string::npos;
        if (marker != std::string::npos) {
            result.language = result.text.substr(after_prefix, marker - after_prefix);
            text_start = marker + std::string("<asr_text>").size();
        } else {
            static const std::array<const char*, 12> languages = {
                "English", "Chinese", "German", "French", "Spanish", "Italian",
                "Portuguese", "Russian", "Japanese", "Korean", "Arabic", "Hindi"
            };
            for (const char* language : languages) {
                const std::string lang(language);
                if (result.text.compare(after_prefix, lang.size(), lang) == 0) {
                    result.language = lang;
                    text_start = after_prefix + lang.size();
                    break;
                }
            }
            if (result.language.empty()) {
                size_t split = result.text.find_first_of("\n\r\t ", after_prefix);
                if (split != std::string::npos) {
                    result.language = result.text.substr(after_prefix, split - after_prefix);
                    text_start = result.text.find_first_not_of(" \t\r\n", split);
                } else {
                    result.language = result.text.substr(after_prefix);
                }
            }
        }
        if (text_start != std::string::npos) {
            result.text = result.text.substr(text_start);
        } else {
            result.text.clear();
        }
    }

    while (!result.text.empty() && (result.text.back() == '\n' || result.text.back() == '\r')) {
        result.text.pop_back();
    }
    size_t first_text = result.text.find_first_not_of(" \t\r\n");
    if (first_text == std::string::npos) {
        result.text.clear();
    } else if (first_text > 0) {
        result.text = result.text.substr(first_text);
    }
    return result;
}

std::vector<int> ncnn_qwen3_asr::build_prompt_ids(int audio_token_count,
                                                  const std::string& context,
                                                  const std::string& language) const {
    if (audio_token_count <= 0) {
        return {};
    }

    const int im_start = 151644;
    const int im_end = 151645;
    const int newline = 198;
    const int system = 8948;
    const int assistant = 77091;

    std::vector<int> ids = {im_start, system, newline};
    if (!context.empty() && tokenizer_) {
        std::vector<int> context_ids = tokenizer_->encode(context, false, false, false, false);
        ids.insert(ids.end(), context_ids.begin(), context_ids.end());
    }
    ids.push_back(im_end);
    ids.push_back(newline);
    ids.push_back(im_start);
    ids.push_back(user_token_id_);
    ids.push_back(newline);
    ids.push_back(audio_start_token_id_);
    for (int i = 0; i < audio_token_count; i++) {
        ids.push_back(audio_token_id_);
    }
    ids.push_back(audio_end_token_id_);
    ids.push_back(im_end);
    ids.push_back(newline);
    ids.push_back(im_start);
    ids.push_back(assistant);
    ids.push_back(newline);
    if (!language.empty() && tokenizer_) {
        std::vector<int> language_ids = tokenizer_->encode("language " + language,
                                                           false, false, false, false);
        ids.insert(ids.end(), language_ids.begin(), language_ids.end());
        ids.push_back(asr_text_token_id_);
    }
    return ids;
}

ncnn::Mat ncnn_qwen3_asr::merge_audio_embeddings(const ncnn::Mat& text_embeds,
                                                 const std::vector<int>& input_ids,
                                                 const ncnn::Mat& audio_embeds) const {
    ncnn::Mat merged = text_embeds.clone();
    if (hidden_size_ <= 0 || merged.w != hidden_size_ || audio_embeds.w != hidden_size_) {
        fprintf(stderr, "Qwen3-ASR merge shape mismatch: text w=%d audio w=%d hidden=%d\n",
                merged.w, audio_embeds.w, hidden_size_);
        return {};
    }

    int audio_row = 0;
    for (size_t i = 0; i < input_ids.size(); i++) {
        if (input_ids[i] != audio_token_id_) {
            continue;
        }
        if (audio_row >= audio_embeds.h || (int)i >= merged.h) {
            fprintf(stderr, "Qwen3-ASR audio token/audio embedding count mismatch\n");
            return {};
        }
        float* dst = merged.row((int)i);
        const float* src = audio_embeds.row(audio_row);
        std::memcpy(dst, src, sizeof(float) * (size_t)hidden_size_);
        audio_row++;
    }

    if (audio_row != audio_embeds.h) {
        fprintf(stderr, "Qwen3-ASR unused audio embeddings: used=%d total=%d\n", audio_row, audio_embeds.h);
        return {};
    }
    return merged;
}

Qwen3ASRFirstStepDebug ncnn_qwen3_asr::debug_first_step(const std::vector<int>& input_ids,
                                                        const ncnn::Mat& audio_embeds) const {
    Qwen3ASRFirstStepDebug debug;
    if ((int)input_ids.size() > text_seq_len_) {
        fprintf(stderr, "Qwen3-ASR debug input length %zu exceeds static text_seq_len %d\n", input_ids.size(), text_seq_len_);
        return debug;
    }

    debug.prompt_len = (int)input_ids.size();
    std::vector<int> padded_ids = input_ids;
    padded_ids.resize((size_t)text_seq_len_, 0);
    debug.text_embeds = run_text_embed(padded_ids);
    if (debug.text_embeds.total() == 0) {
        return debug;
    }

    debug.merged_embeds = merge_audio_embeddings(debug.text_embeds, input_ids, audio_embeds);
    if (debug.merged_embeds.total() == 0) {
        return debug;
    }

    std::vector<int> mask((size_t)text_seq_len_, 0);
    std::fill(mask.begin(), mask.begin() + (std::vector<int>::difference_type)input_ids.size(), 1);
    debug.hidden = run_text_backbone(debug.merged_embeds, mask);
    if (debug.hidden.total() == 0) {
        return debug;
    }

    debug.logits = run_lm_head(debug.hidden);
    if (debug.logits.total() == 0) {
        return debug;
    }

    if (debug.logits.h <= (int)input_ids.size() - 1) {
        debug.selected_logits = debug.logits.clone();
    } else {
        debug.selected_logits = ncnn::Mat(debug.logits.w, (void*)debug.logits.row((int)input_ids.size() - 1), sizeof(float), 1);
        debug.selected_logits = debug.selected_logits.clone();
    }
    debug.next_token = select_next_token_from_logits(debug.selected_logits);
    return debug;
}

int ncnn_qwen3_asr::decode_next_token(const std::vector<int>& input_ids, const ncnn::Mat& audio_embeds) const {
    if ((int)input_ids.size() > text_seq_len_) {
        fprintf(stderr, "Qwen3-ASR input length %zu exceeds static text_seq_len %d\n", input_ids.size(), text_seq_len_);
        return -1;
    }

    std::vector<int> padded_ids = input_ids;
    padded_ids.resize((size_t)text_seq_len_, 0);
    ncnn::Mat text_embeds = run_text_embed(padded_ids);
    if (text_embeds.total() == 0) {
        return -1;
    }

    ncnn::Mat merged = merge_audio_embeddings(text_embeds, input_ids, audio_embeds);
    if (merged.total() == 0) {
        return -1;
    }

    std::vector<int> mask((size_t)text_seq_len_, 0);
    std::fill(mask.begin(), mask.begin() + (std::vector<int>::difference_type)input_ids.size(), 1);
    ncnn::Mat hidden = run_text_backbone(merged, mask);
    if (hidden.total() == 0) {
        return -1;
    }
    ncnn::Mat logits = run_lm_head(hidden);
    if (logits.total() == 0) {
        return -1;
    }

    if (logits.h <= (int)input_ids.size() - 1) {
        return select_next_token_from_logits(logits);
    }
    ncnn::Mat row(logits.w, (void*)logits.row((int)input_ids.size() - 1), sizeof(float), 1);
    row = row.clone();
    return select_next_token_from_logits(row);
}

int ncnn_qwen3_asr::prefill_kv(const std::vector<int>& input_ids,
                               const ncnn::Mat& audio_embeds,
                               Qwen3ASRKVDecodeState& state) const {
    if (!has_kv_decoder_) {
        return -1;
    }
    if ((int)input_ids.size() > text_seq_len_) {
        fprintf(stderr, "Qwen3-ASR KV prefill length %zu exceeds static text_seq_len %d\n",
                input_ids.size(), text_seq_len_);
        return -1;
    }

    ncnn::Mat text_embeds = run_text_embed(input_ids);
    if (text_embeds.total() == 0) {
        return -1;
    }

    ncnn::Mat merged = merge_audio_embeddings(text_embeds, input_ids, audio_embeds);
    if (merged.total() == 0) {
        return -1;
    }

    std::vector<int> mask(input_ids.size(), 1);
    ncnn::Mat hidden = run_text_prefill_kv(merged, mask, state);
    if (hidden.total() == 0) {
        return -1;
    }

    ncnn::Mat logits = run_lm_head(hidden);
    if (logits.total() == 0) {
        return -1;
    }
    ncnn::Mat row(logits.w, (void*)logits.row((int)input_ids.size() - 1), sizeof(float), 1);
    row = row.clone();
    state.position = (int)input_ids.size();
    return select_next_token_from_logits(row);
}

int ncnn_qwen3_asr::decode_next_token_kv(int token_id, Qwen3ASRKVDecodeState& state) const {
    if (!has_kv_decoder_ || !state.ready) {
        return -1;
    }
    ncnn::Mat token_embed = run_text_embed({token_id});
    if (token_embed.total() == 0) {
        return -1;
    }
    ncnn::Mat hidden = run_text_decode_kv(token_embed, state);
    if (hidden.total() == 0) {
        return -1;
    }
    ncnn::Mat lm_hidden = hidden;
    if (hidden.h == 1 && text_seq_len_ > 1 && hidden_size_ > 0) {
        lm_hidden = ncnn::Mat(hidden_size_, text_seq_len_);
        lm_hidden.fill(0.0f);
        std::memcpy(lm_hidden.row(0), hidden.row(0), sizeof(float) * (size_t)hidden_size_);
    }
    ncnn::Mat logits = run_lm_head(lm_hidden);
    if (logits.total() == 0) {
        return -1;
    }
    if (lm_hidden.h > 1 && logits.h > 1) {
        ncnn::Mat row(logits.w, (void*)logits.row(0), sizeof(float), 1);
        row = row.clone();
        return select_next_token_from_logits(row);
    }
    return select_next_token_from_logits(logits);
}
