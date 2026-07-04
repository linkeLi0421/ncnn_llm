#include "ncnn_qwen3_asr.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <stdexcept>

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
        const std::string lm_head_param = model_path + "/" + params["lm_head_param"].get<std::string>();
        const std::string lm_head_bin = model_path + "/" + params["lm_head_bin"].get<std::string>();

        printf("Loading Qwen3-ASR model from %s\n", model_path.c_str());
        printf("  audio_encoder: %s / %s\n", audio_param.c_str(), audio_bin.c_str());
        printf("  text_embed:    %s / %s\n", text_embed_param.c_str(), text_embed_bin.c_str());
        printf("  text_backbone: %s / %s\n", text_backbone_param.c_str(), text_backbone_bin.c_str());
        printf("  lm_head:       %s / %s\n", lm_head_param.c_str(), lm_head_bin.c_str());

        text_backbone_has_attention_mask_ = param_has_input_blob(text_backbone_param, "in1");

        if (!load_one_net(*audio_encoder_net_, audio_param, audio_bin)) return false;
        if (!load_one_net(*text_embed_net_, text_embed_param, text_embed_bin)) return false;
        if (!load_one_net(*text_backbone_net_, text_backbone_param, text_backbone_bin)) return false;
        if (!load_one_net(*lm_head_net_, lm_head_param, lm_head_bin)) return false;

        if (config.contains("setting")) {
            const auto& setting = config["setting"];
            if (setting.contains("audio_token_id")) {
                audio_token_id_ = setting["audio_token_id"].get<int>();
            }
            if (setting.contains("audio_start_token_id")) {
                audio_start_token_id_ = setting["audio_start_token_id"].get<int>();
            }
            if (setting.contains("user_token_id")) {
                user_token_id_ = setting["user_token_id"].get<int>();
            }
            if (setting.contains("text_config") && setting["text_config"].contains("hidden_size")) {
                hidden_size_ = setting["text_config"]["hidden_size"].get<int>();
            }
            if (setting.contains("text_config") && setting["text_config"].contains("vocab_size")) {
                vocab_size_ = setting["text_config"]["vocab_size"].get<int>();
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
            if (vocab_size_ <= 0) {
                vocab_size_ = (int)tokenizer_->vocab_size();
            }
        }

        printf("  hidden_size=%d vocab_size=%d audio_token_id=%d audio_start_token_id=%d user_token_id=%d\n",
               hidden_size_, vocab_size_, audio_token_id_, audio_start_token_id_, user_token_id_);
        printf("  text_backbone_attention_mask=%s\n", text_backbone_has_attention_mask_ ? "true" : "false");
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

std::string ncnn_qwen3_asr::decode(const std::vector<int>& ids, bool skip_special_tokens) const {
    if (!tokenizer_) {
        return {};
    }
    return tokenizer_->decode(ids, skip_special_tokens);
}
