#pragma once

#include <functional>
#include <memory>
#include <string>

struct NllbConfig {
    float temperature = 1.0f;
    int top_k = 0;
    float top_p = 1.0f;
    bool do_sample = false;
    int max_steps = 512;
};

class nllb_600m {
public:
    nllb_600m(std::string embed_param,
              std::string embed_bin,
              std::string encoder_param,
              std::string encoder_bin,
              std::string decoder_param,
              std::string decoder_bin,
              std::string vocab_file,
              std::string merges_file);

    nllb_600m(std::string embed_param,
              std::string embed_bin,
              std::string encoder_param,
              std::string encoder_bin,
              std::string decoder_param,
              std::string decoder_bin,
              std::string vocab_file,
              std::string merges_file,
              bool use_vulkan);

    ~nllb_600m();

    std::string translate(const std::string& input_text,
                          const std::string& source_lang,
                          const std::string& target_lang);

    std::string translate(const std::string& input_text,
                          const std::string& source_lang,
                          const std::string& target_lang,
                          const NllbConfig& config);

    bool translate(const std::string& input_text,
                   const std::string& source_lang,
                   const std::string& target_lang,
                   std::function<void(const std::string&)> callback);

    bool translate(const std::string& input_text,
                   const std::string& source_lang,
                   const std::string& target_lang,
                   const NllbConfig& config,
                   std::function<void(const std::string&)> callback);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};