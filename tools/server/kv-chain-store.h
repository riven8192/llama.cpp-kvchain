#pragma once

#include "llama.h"
#include "common.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// disk-backed, content-addressed KV state store
// layout: <root_dir>/<root_hash>/<chain_hash>.kvchunk
// chunk file: u32 magic, u32 version, u32 root_hash, u32 parent_hash,
//             u32 n_tokens, llama_token[n_tokens], then the seq state blob
// (u32 io_magic, u32 src_seq, module data), then u64 fnv1a checksum of everything before it
struct kv_chain_store {
    kv_chain_store(std::string root_dir, uint64_t limit_bytes);

    // saves the full state of seq_id covering the given prompt tokens
    // returns false on failure (store disabled, io error, ...)
    bool save(llama_context * ctx, llama_seq_id seq_id, const llama_tokens & tokens);

    // finds the longest saved prefix matching tokens[0..lcp)
    // returns the loaded state blob (empty on miss), and *n_tokens = prefix length
    std::vector<uint8_t> load_prefix(const llama_tokens & tokens, size_t * n_tokens) const;

    size_t total_bytes() const { return total_bytes_cur; }
    bool   enabled() const { return !root_dir.empty(); }

private:
    static uint64_t fnv1a64(const uint8_t * data, size_t len);
    static uint32_t hash_tokens(const llama_tokens & tokens, uint32_t prev);
    static std::string hash_str(uint64_t h);

    std::string root_dir;
    uint64_t    limit_bytes;
    uint64_t    total_bytes_cur = 0;
    bool        index_loaded    = false;
};
