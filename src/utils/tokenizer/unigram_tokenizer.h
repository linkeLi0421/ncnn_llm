#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <mutex>
#include "tokenizer_types.h"

class UnigramTokenizer {
public:
    static UnigramTokenizer LoadFromFile(const std::string& model_path,
                                         const SpecialTokensConfig& spec,
                                         bool add_special_if_missing = true,
                                         bool fallback_to_chars = true,
                                         double unk_penalty = -10.0);

    std::vector<int> encode(const std::string& text,
                            bool add_bos = false,
                            bool add_eos = false,
                            bool add_cls = false,
                            bool add_sep = false) const;

    std::string decode(const std::vector<int>& ids, bool skip_special_tokens = true) const;

    size_t vocab_size() const { return id_to_token_.size(); }
    const std::vector<std::string>& id_to_token() const { return id_to_token_; }
    const std::unordered_map<std::string, int>& token_to_id() const { return token_to_id_; }
    const SpecialTokenIds& special_ids() const { return special_ids_; }
    bool fallback_to_chars() const { return fallback_to_chars_; }
    double unk_penalty() const { return unk_penalty_; }

    UnigramTokenizer(const UnigramTokenizer&) = delete;
    UnigramTokenizer& operator=(const UnigramTokenizer&) = delete;
    UnigramTokenizer(UnigramTokenizer&& other) noexcept;
    UnigramTokenizer& operator=(UnigramTokenizer&& other) noexcept;

private:
    UnigramTokenizer() = default;

    static void LoadModel(const std::string& model_path,
                          std::vector<std::string>& tokens,
                          std::vector<double>& scores);
    static std::unordered_map<std::string, int> BuildTokenToId(const std::vector<std::string>& id_to_token);

    void EnsureSpecialTokens(const SpecialTokensConfig& spec, bool add_if_missing);

    static std::vector<std::string> PretokenizeSentencePiece(const std::string& text);

    static bool IsAsciiSpace(unsigned char c);
    static bool IsUnicodeSpace(uint32_t cp);
    static bool NextUtf8(const std::string& s, size_t& i, uint32_t& cp, size_t& cp_len);

    void BuildTrie();
    void AddToTrie(const std::string& token, int token_id);
    void MatchAt(const std::string& s, size_t pos, std::vector<std::pair<int,int>>& out_matches) const;

    const std::vector<std::string>& SegmentPieceCached(const std::string& piece) const;
    std::vector<std::string> SegmentPiece(const std::string& piece) const;

    void TokensToIds(const std::vector<std::string>& tokens, std::vector<int>& out) const;

private:
    struct TrieNode {
        int next[256];
        int token_id;
        TrieNode() : token_id(-1) {
            for (int i = 0; i < 256; ++i) next[i] = -1;
        }
    };

    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> token_to_id_;
    std::vector<double> token_logprob_;

    std::vector<TrieNode> trie_;

    SpecialTokenIds special_ids_;
    bool fallback_to_chars_ = true;
    double unk_penalty_ = -10.0;

    mutable std::unordered_map<std::string, std::vector<std::string>> piece_cache_;
    mutable std::mutex cache_mu_;
};