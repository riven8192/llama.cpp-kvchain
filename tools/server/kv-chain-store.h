#pragma once

#include "llama.h"
#include "common.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// one matched chunk's payload, in chain order. each chunk holds the attn rows
// and the recurrent rows for ITS OWN window [k*bs, (k+1)*bs) - no duplication
// across chunks. on restore the caller loads each chunk's attn (appending,
// first chunk wipes) and each chunk's recurrent (overwriting; the last one wins).
struct kv_chain_chunk {
    std::vector<uint8_t> attn_blob; // seq state blob for the attn (per-token KV) part
    std::vector<uint8_t> recr_blob; // seq state blob for the recurrent (R/S) part
};

// disk-backed, content-addressed KV state store
// layout: <root_dir>/<chunk_hash>.kvchunk
// chunk file: u32 magic, u32 version, u32 chunk_hash, u32 n_tokens,
//             llama_token[n_tokens], u32 attn_size, attn_blob[attn_size],
//             u32 recr_size, recr_blob[recr_size], u64 fnv1a checksum of everything before it
// both blobs are self-contained seq-state blobs (each carries its own io_magic,
// src_seq, module header) so they can be fed straight to the state_seq_set API.
struct kv_chain_store {
    kv_chain_store(std::string root_dir, uint64_t limit_bytes, int32_t batch_size);

    // saves one chunk covering the window [pos_lo, pos_hi) of seq_id's state.
    // content-addressed by chunk_hash. chunk_tokens are this chunk's own tokens
    // (pos_hi-pos_lo of them), stored verbatim in the header for validation.
    // returns false on failure (store disabled, io error, ...)
    bool save(llama_context * ctx, llama_seq_id seq_id, llama_pos pos_lo, llama_pos pos_hi,
              uint32_t chunk_hash, const llama_tokens & chunk_tokens);

    // hash for one chunk: FNV-1a over the chunk's token ids, seeded by prev_hash
    // (prev_hash = 0 for the root chunk). this is the hash-chain step.
    uint32_t hash_chunk(const llama_tokens & chunk_tokens, uint32_t prev_hash) const;

    // finds the longest saved prefix matching tokens[0..lcp). walks the chain,
    // stopping at the first missing/corrupt file. returns the matched chunks in
    // order (empty on miss) and sets *n_tokens = prefix length (= n_chunks*bs).
    std::vector<kv_chain_chunk> load_prefix(const llama_tokens & tokens, size_t * n_tokens) const;

    size_t total_bytes() const { return total_bytes_cur; }
    bool   enabled() const { return !root_dir.empty(); }
    int32_t batch_size() const { return batch_size_; }

private:
    static uint64_t fnv1a64(const uint8_t * data, size_t len);
    static std::string hash_str(uint64_t h);

    bool write_chunk(const fs::path & dir, uint32_t chunk_hash, const llama_tokens & chunk_tokens,
                     const std::vector<uint8_t> & attn, const std::vector<uint8_t> & recr);
    bool write_chunk_file(const fs::path & tmp, const fs::path & file, uint32_t chunk_hash,
                          const llama_tokens & tokens, const std::vector<uint8_t> & attn,
                          const std::vector<uint8_t> & recr);
    void evict_oldest(uint64_t need_bytes);

    // reads both blobs from a chunk file; returns false if missing/corrupt
    static bool read_chunk(const fs::path & file, kv_chain_chunk & out);

    std::string root_dir;
    uint64_t    limit_bytes;
    int32_t     batch_size_; // chunk stride (boundary grid), not the runtime ubatch size
    uint64_t    total_bytes_cur = 0;
    bool        index_loaded    = false;
};
