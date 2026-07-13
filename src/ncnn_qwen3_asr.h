#pragma once

#include <memory>
#include <string>
#include <vector>

#include <mat.h>
#include <net.h>
#include <nlohmann/json.hpp>

#include "ncnn_llm_base.h"
#include "utils/tokenizer/bpe_tokenizer.h"

using nlohmann::json;

struct Qwen3ASRResult {
    std::string language;
    std::string text;
    std::string raw_text;
};

struct Qwen3ASRKVDecodeState {
    KVCache kv_cache;
    int position = 0;
    bool ready = false;
};

struct Qwen3ASRFirstStepDebug {
    ncnn::Mat text_embeds;
    ncnn::Mat merged_embeds;
    ncnn::Mat hidden;
    ncnn::Mat logits;
    ncnn::Mat selected_logits;
    int prompt_len = 0;
    int next_token = -1;
};

class ncnn_qwen3_asr : public ncnn_llm_base {
public:
    ncnn_qwen3_asr(const std::string& model_path, bool use_vulkan = false, int num_threads = 0);

    bool ok() const { return ok_; }
    int hidden_size() const { return hidden_size_; }
    int vocab_size() const { return vocab_size_; }
    int audio_token_id() const { return audio_token_id_; }
    int audio_start_token_id() const { return audio_start_token_id_; }
    int text_seq_len() const { return text_seq_len_; }
    bool has_kv_decoder() const { return has_kv_decoder_; }

    // Expects log-mel features as an ncnn Mat with w=frames, h=num_mel_bins.
    ncnn::Mat run_audio_encoder(const ncnn::Mat& mel_features) const;
    ncnn::Mat run_text_embed(const std::vector<int>& input_ids) const;
    ncnn::Mat run_text_backbone(const ncnn::Mat& input_embeds, const std::vector<int>& attention_mask) const;
    ncnn::Mat run_text_prefill_kv(const ncnn::Mat& input_embeds,
                                  const std::vector<int>& attention_mask,
                                  Qwen3ASRKVDecodeState& state) const;
    ncnn::Mat run_text_decode_kv(const ncnn::Mat& input_embed,
                                 Qwen3ASRKVDecodeState& state) const;
    ncnn::Mat run_lm_head(const ncnn::Mat& hidden_states) const;

    int select_next_token_from_logits(const ncnn::Mat& logits) const;
    bool should_stop_token(int token_id) const;
    std::string decode(const std::vector<int>& ids, bool skip_special_tokens = true) const;
    Qwen3ASRResult parse_output(const std::vector<int>& generated_ids) const;
    std::vector<int> build_prompt_ids(int audio_token_count,
                                      const std::string& context = "",
                                      const std::string& language = "") const;
    ncnn::Mat merge_audio_embeddings(const ncnn::Mat& text_embeds,
                                     const std::vector<int>& input_ids,
                                     const ncnn::Mat& audio_embeds) const;
    Qwen3ASRFirstStepDebug debug_first_step(const std::vector<int>& input_ids,
                                            const ncnn::Mat& audio_embeds) const;
    int decode_next_token(const std::vector<int>& input_ids, const ncnn::Mat& audio_embeds) const;
    int prefill_kv(const std::vector<int>& input_ids,
                   const ncnn::Mat& audio_embeds,
                   Qwen3ASRKVDecodeState& state) const;
    int decode_next_token_kv(int token_id, Qwen3ASRKVDecodeState& state) const;

private:
    bool load_model_file(const std::string& model_path);
    bool load_one_net(ncnn::Net& net, const std::string& param_path, const std::string& bin_path);
    void configure_net(ncnn::Net& net);
    static bool param_has_input_blob(const std::string& param_path, const std::string& blob_name);

private:
    std::shared_ptr<ncnn::Net> audio_encoder_net_;
    std::shared_ptr<ncnn::Net> text_embed_net_;
    std::shared_ptr<ncnn::Net> text_backbone_net_;
    std::shared_ptr<ncnn::Net> text_prefill_kv_net_;
    std::shared_ptr<ncnn::Net> text_decode_kv_net_;
    std::shared_ptr<ncnn::Net> lm_head_net_;
    std::shared_ptr<BpeTokenizer> tokenizer_;

    int hidden_size_ = 0;
    int vocab_size_ = 0;
    int audio_token_id_ = -1;
    int audio_start_token_id_ = -1;
    int user_token_id_ = -1;
    int audio_end_token_id_ = -1;
    int asr_text_token_id_ = 151704;
    int text_seq_len_ = 8;
    int kv_cache_len_ = 0;
    int num_hidden_layers_ = 0;
    int rope_head_dim_ = 0;
    float rope_theta_ = 1000000.0f;
    bool text_backbone_has_attention_mask_ = false;
    bool has_kv_decoder_ = false;
};
