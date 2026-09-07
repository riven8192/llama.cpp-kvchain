#pragma once

#include "llama.h"
#include "common.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// one matched chunk, in chain order. holds only file paths, not blob data:
// the caller replays the chunks one at a time (read -> set_data -> free),
// so the whole chain is never in RAM at once.
struct kv_chain_chunk {
    fs::path     attn_file; // this chunk's .kvcache (ATTN_ONLY rows for [k*bs,(k+1)*bs))
    fs::path     recr_file; // this chunk's .rscache; set for the TAIL chunk only
    llama_tokens tokens;    // the chunk's token IDs, verbatim from the file header
};

// disk-backed, content-addressed KV state store.
// layout: <cache_dir>/<chunk_hash_hex>.kvcache  (attn only)
//         <cache_dir>/<chunk_hash_hex>.rscache  (recr tail)
// each file: u32 magic KVC1, u32 version, u32 hash32, u32 n_tokens,
//            llama_token[n_tokens], u32 blob_size, blob[blob_size].
// no trailing checksum: verifying one costs a full pass over every (multi-MiB)
// recr file, which dominated restore time.
// the blob is a self-contained seq-state blob (its own io_magic + module
// header, no source seq_id), feedable straight to the state_seq_set API.
// root_hash = FNV-1a64 over a canonical metadata blob; everything in the blob
// must affect the numeric content or layout of cached KV values, so any change
// is a clean miss, never a garbage read.
struct kv_chain_metadata {
    int32_t  format_version;              // KV_CHAIN_VERSION (single version: file layout + this blob)
    uint32_t chunk_size;                  // == n_ubatch at store construction
    int64_t  model_file_size;             // stat() of the model file (-1 if stat fails)
    int64_t  model_file_mtime;            // stat() mtime seconds
    std::string arch;                     // model architecture string
    std::string ftype;                    // model quantization string
    uint32_t type_k;                      // ggml_type of the K cache
    uint32_t type_v;                      // ggml_type of the V cache
    int32_t  rope_scaling_type;           // llama_rope_scaling_type
    uint32_t rope_freq_base_bits;         // float bits (0.0f = "from model")
    uint32_t rope_freq_scale_bits;        // float bits (0.0f = "from model")
    uint32_t n_seq_max;                   // --parallel: the attn blob's n_stream scales with it, so a
                                          // different --parallel changes the root hash (clean miss)
};

class kv_chain_store {
public:
    // ubatch_size: the chunk stride == n_ubatch (-ub). chunk boundaries are the
    // ubatch boundaries (where the state snapshots fire), not the n_batch size.
    kv_chain_store(std::string root_dir, uint64_t limit_bytes, int32_t ubatch_size,
                   const common_params & params, const llama_model * model);

    // saves one chunk covering the window [pos_lo, pos_hi) of seq_id's state.
    // chunk_tokens (pos_hi-pos_lo of them) are stored verbatim in the header for
    // validation. returns false on failure (store disabled, io error, ...).
    bool save(llama_context * ctx, llama_seq_id seq_id, llama_pos pos_lo, llama_pos pos_hi,
              uint64_t chunk_hash, const llama_tokens & chunk_tokens,
              uint64_t parent_hash = 0); // parent_hash: logging only

    // hash for one chunk: FNV-1a over the chunk's token ids, seeded by prev_hash.
    uint64_t hash_chunk(const llama_tokens & chunk_tokens, uint64_t prev_hash) const;

    // the full hash chain for a prompt, computed once at prompt arrival:
    // hashes[k] = hash for chunk k (tokens [k*bs, (k+1)*bs)). the restore walk
    // and the per-ubatch save hook both index this vector, so save and restore
    // can never disagree about a chunk's name.
    std::vector<uint64_t> hash_chain(const llama_tokens & tokens) const;

    // finds the longest saved prefix matching the prompt. walks the chain,
    // stopping at the first missing file. returns the matched chunks in order
    // (empty on miss) and sets *n_tokens = prefix length (= n_chunks*bs).
    std::vector<kv_chain_chunk> load_prefix(const llama_tokens & tokens, size_t * n_tokens) const;

    // reads + validates one chunk file (header magic/version + the token IDs
    // must match expected_tokens) and returns the blob. returns false if
    // missing/corrupt/token-mismatch.
    bool read_chunk_file(const fs::path & file, std::vector<uint8_t> & out_blob,
                         const llama_tokens & expected_tokens) const;

    // per-chunk "rs file present" map from the last load_prefix() call (chain
    // order; 1 = present). used by the replay loop to truncate at the deepest
    // loaded chunk whose rs file is present, if a .kvcache read fails mid-replay.
    const std::vector<uint8_t> & last_rs_present() const { return last_rs_present_; }

    // utimensat the files of the given chunk hashes to "now". the default
    // relatime mount does not update mtime on read, so without this LRU would
    // evict the hottest chains first. load_prefix() already touches on hit;
    // this is for other restore paths.
    void touch_chunks(const std::vector<uint64_t> & chunk_hashes) const;

    size_t total_bytes() const { return total_bytes_cur; }
    bool   enabled() const { return !root_dir.empty(); }
    int32_t ubatch_size() const { return ubatch_size_; }
    uint64_t root_hash() const { return root_hash_; }

    static uint64_t fnv1a64(const uint8_t * data, size_t len);
    static uint64_t fnv1a64(uint64_t h, const uint8_t * data, size_t len);

private:
    static std::string hash_str(uint64_t h);

    // computes the root hash from the metadata blob (see struct above).
    // the model file is stat()ed only, never opened.
    void compute_root_hash(const common_params & params, const llama_model * model);

    bool write_chunk(const fs::path & dir, uint64_t chunk_hash, const llama_tokens & chunk_tokens,
                     const std::vector<uint8_t> & attn, const std::vector<uint8_t> & recr);
    bool write_chunk_file(const fs::path & tmp, const fs::path & file, uint64_t chunk_hash,
                          const llama_tokens & tokens, const std::vector<uint8_t> & blob);
    void evict_oldest(uint64_t need_bytes);

    std::string root_dir;
    uint64_t    root_hash_ = 0; // identity of this model/config; parent for chunk 0's hash
    uint64_t    limit_bytes;
    int32_t     ubatch_size_; // chunk stride == n_ubatch
    uint64_t    total_bytes_cur = 0;
    mutable std::vector<uint8_t> last_rs_present_; // set by load_prefix() (const: caches the last call's result)
};
