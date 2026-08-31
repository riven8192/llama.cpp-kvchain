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
    llama_tokens        tokens;    // the chunk's token IDs, verbatim from the file header
};

// disk-backed, content-addressed KV state store
// layout (v3): <cache_dir>/<chunk_hash_hex>.kvcache  (attn only)
//              <cache_dir>/<chunk_hash_hex>.rscache  (recr only)
// flat directory (no per-model subdirs). the root hash is still computed and
// used to validate the metadata; files from different models/configs simply
// never match because the chunk hashes differ (they are seeded by the root).
//
// each file: u32 magic KVC1, u32 version=3, u32 hash32, u32 n_tokens,
//            llama_token[n_tokens], u32 blob_size, blob[blob_size],
//            u64 fnv1a checksum of everything before it.
// the blob is a self-contained seq-state blob (carries its own io_magic,
// src_seq, module header) so it can be fed straight to the state_seq_set API.
//
// root_hash = FNV-1a64 over a canonical metadata blob. everything in the blob
// must affect the numeric content or LAYOUT of cached KV values, so that any
// change produces a fresh (clean-miss) directory instead of a silent garbage
// read. gathering is deliberately cheap: one stat() on the model file, no
// model-weight reads, no file parsing.
struct kv_chain_metadata {
    int32_t  format_version;              // KV_CHAIN_VERSION (single version: file layout + this blob)
    uint32_t chunk_size;                  // == llama_n_batch(ctx) at store construction
    int64_t  model_file_size;             // stat() of the model file (-1 if stat fails)
    int64_t  model_file_mtime;            // stat() mtime seconds
    std::string arch;                     // llama_model_arch_name(model) (architecture string)
    std::string ftype;                    // llama_model_ftype_name(model) (quantization string)
    uint32_t type_k;                      // ggml_type of the K cache
    uint32_t type_v;                      // ggml_type of the V cache
    int32_t  rope_scaling_type;           // llama_rope_scaling_type
    uint32_t rope_freq_base_bits;         // float bits (0.0f = "from model")
    uint32_t rope_freq_scale_bits;        // float bits (0.0f = "from model")
};

struct kv_chain_store {
    // root_dir: base cache dir (empty = feature disabled).
    // params:   common_params of the loaded model (rope config, kv dtypes).
    // model:    the loaded model (arch + quant string + file stat).
    kv_chain_store(std::string root_dir, uint64_t limit_bytes, int32_t batch_size,
                   const common_params & params, const llama_model * model);

    // saves one chunk covering the window [pos_lo, pos_hi) of seq_id's state.
    // content-addressed by chunk_hash (the low 32 bits are stored in the header).
    // chunk_tokens are this chunk's own tokens (pos_hi-pos_lo of them), stored
    // verbatim in the header for validation.
    // returns false on failure (store disabled, io error, ...)
    bool save(llama_context * ctx, llama_seq_id seq_id, llama_pos pos_lo, llama_pos pos_hi,
              uint64_t chunk_hash, const llama_tokens & chunk_tokens,
              uint64_t parent_hash = 0); // parent_hash: [DEBUG] logging only

    // hash for one chunk: FNV-1a over the chunk's token ids, seeded by prev_hash
    // (prev_hash = 0 for the root chunk). this is the hash-chain step.
    // uint64 to match the root hash (consistency); the header stores it as u32.
    uint64_t hash_chunk(const llama_tokens & chunk_tokens, uint64_t prev_hash) const;

    // the FULL hash chain for a prompt, computed ONCE at prompt arrival:
    // hashes[k] = hash_chain step for chunk k (tokens [k*bs, (k+1)*bs)).
    // hashes.size() = n_complete_chunks(tokens). the restore walk and the
    // per-ubatch save hook both INDEX this vector - neither recomputes hashes,
    // so save and restore can never disagree about a chunk's name.
    std::vector<uint64_t> hash_chain(const llama_tokens & tokens) const;

    // finds the longest saved prefix matching tokens[0..lcp). walks the chain,
    // stopping at the first missing/corrupt file. returns the matched chunks in
    // order (empty on miss) and sets *n_tokens = prefix length (= n_chunks*bs).
    std::vector<kv_chain_chunk> load_prefix(const llama_tokens & tokens, size_t * n_tokens) const;

    // touches (utimensat) the files of the given chunk hashes so their mtime is
    // "now". required because the default relatime mount does not update mtime
    // on read; without this, LRU eviction would evict the hottest chains first.
    // load_prefix() already touches on hit; this is for callers that restore
    // via a different path and want the same LRU semantics.
    void touch_chunks(const std::vector<uint64_t> & chunk_hashes) const;

    size_t total_bytes() const { return total_bytes_cur; }
    bool   enabled() const { return !root_dir.empty(); }
    int32_t batch_size() const { return batch_size_; }
    uint64_t root_hash() const { return root_hash_; }

    static uint64_t fnv1a64(const uint8_t * data, size_t len);
    static uint64_t fnv1a64(uint64_t h, const uint8_t * data, size_t len);

private:
    static std::string hash_str(uint64_t h);

    // computes the root hash from the metadata blob (see struct above).
    // fills root_hash_. the model file is stat()ed only - never opened.
    void compute_root_hash(const common_params & params, const llama_model * model);

    bool write_chunk(const fs::path & dir, uint64_t chunk_hash, const llama_tokens & chunk_tokens,
                     const std::vector<uint8_t> & attn, const std::vector<uint8_t> & recr);
    bool write_chunk_file(const fs::path & tmp, const fs::path & file, uint64_t chunk_hash,
                          const llama_tokens & tokens, const std::vector<uint8_t> & blob);
    void evict_oldest(uint64_t need_bytes);

    // reads a single-blob chunk file (.kvcache or .rscache); returns false if missing/corrupt
    static bool read_chunk_file(const fs::path & file, std::vector<uint8_t> & out_blob, llama_tokens & out_tokens);

    std::string root_dir;
    uint64_t    root_hash_ = 0; // identity of this model/config; the chain's parent for chunk 0
    uint64_t    limit_bytes;
    int32_t     batch_size_; // chunk stride (boundary grid), not the runtime ubatch size
    uint64_t    total_bytes_cur = 0;
};
