#pragma once

#include "llama.h"
#include "common.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// disk-backed, content-addressed KV state store
// layout: <root_dir>/<root_hash>/<chain_hash>.kvchunk
// chunk file: u32 magic, u32 version, u32 root_hash, u32 parent_hash,
//             u32 n_tokens, llama_token[n_tokens], then the seq state blob
// (u32 io_magic, u32 src_seq, module data), then u64 fnv1a checksum of everything before it
struct kv_chain_store {
    kv_chain_store(std::string root_dir, uint64_t limit_bytes, int32_t batch_size);

    // saves a chunk: the full state of seq_id covering prefix_tokens[0..pos)
    // where pos = (chunk_n+1)*batch_size. content-addressed by chunk_hash.
    // chunk_tokens are this chunk's own tokens (stored verbatim in the header).
    // returns false on failure (store disabled, io error, ...)
    bool save(llama_context * ctx, llama_seq_id seq_id, const llama_tokens & prefix_tokens,
              uint32_t chunk_hash, const llama_tokens & chunk_tokens);

    // hash for one chunk: FNV-1a over the chunk's token ids, seeded by prev_hash
    // (prev_hash = 0 for the root chunk). this is the hash-chain step.
    uint32_t hash_chunk(const llama_tokens & chunk_tokens, uint32_t prev_hash) const;

    // finds the longest saved prefix matching tokens[0..lcp)
    // returns the loaded state blob (empty on miss), and *n_tokens = prefix length
    std::vector<uint8_t> load_prefix(const llama_tokens & tokens, size_t * n_tokens) const;

    size_t total_bytes() const { return total_bytes_cur; }
    bool   enabled() const { return !root_dir.empty(); }
    int32_t batch_size() const { return batch_size_; }

private:
    static uint64_t fnv1a64(const uint8_t * data, size_t len);
    static std::string hash_str(uint64_t h);

    bool write_chunk(const fs::path & dir, uint32_t chunk_hash, const llama_tokens & chunk_tokens,
                     const std::vector<uint8_t> & state, size_t pos);
    bool write_chunk_file(const fs::path & tmp, const fs::path & file, uint32_t chain_hash,
                          const llama_tokens & tokens, const std::vector<uint8_t> & state);
    void evict_oldest(uint64_t need_bytes);

    // reads the state blob from a chunk file; returns empty if missing/corrupt
    static std::vector<uint8_t> read_chunk_blob(const fs::path & file);

    std::string root_dir;
    uint64_t    limit_bytes;
    int32_t     batch_size_; // dump granularity (= n_batch = n_ubatch)
    uint64_t    total_bytes_cur = 0;
    bool        index_loaded    = false;
};
