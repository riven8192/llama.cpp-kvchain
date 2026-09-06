#pragma once

#include "llama.h"
#include "common.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// one matched chunk, in chain order. holds ONLY the file paths (and the
// validated header tokens) - NOT the blob data. the blobs are large (~16 MiB
// attn per chunk; the tail rs is ~150 MiB), and materializing the WHOLE chain
// in RAM before the restore (the old behavior) peaked at ~2x the chain size on
// resource-constrained boxes. the restore therefore replays chunk by chunk:
// open the file, feed the blob to set_data, close, next. peak = one file.
// attn_file:  this chunk's .kvcache (ATTN_ONLY rows for [k*bs,(k+1)*bs)),
//             always set. additive across chunks (chunk 0 wipes, rest APPEND).
// recr_file:  this chunk's .rscache. set for the TAIL chunk only (the
//             recurrent tail is a single fixed-size "last write wins" object;
//             the earlier rs files are superseded and never read). empty for
//             all other chunks.
struct kv_chain_chunk {
    fs::path     attn_file; // .kvcache file for this chunk's window
    fs::path     recr_file; // .rscache file, TAIL chunk only
    llama_tokens tokens;    // the chunk's token IDs, verbatim from the file header
};

// disk-backed, content-addressed KV state store
// layout (v3): <cache_dir>/<chunk_hash_hex>.kvcache  (attn only)
//              <cache_dir>/<chunk_hash_hex>.rscache  (recr only)
// flat directory (no per-model subdirs). the root hash is still computed and
// used to validate the metadata; files from different models/configs simply
// never match because the chunk hashes differ (they are seeded by the root).
//
// each file: u32 magic KVC1, u32 version=3, u32 hash32, u32 n_tokens,
//            llama_token[n_tokens], u32 blob_size, blob[blob_size].
// there is NO trailing checksum: verifying one costs a full pass over every
// (multi-hundred-MiB) recr file in the chain, which dominated restore time.
// we trust the storage device (see read_chunk_file in the .cpp).
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
    uint32_t n_seq_max;                   // --parallel: the attn blob's n_stream scales with it, so
                                          // a different --parallel gets a different root hash ->
                                          // a clean miss instead of a blob/live n_stream mismatch
};

class kv_chain_store {
public:
    // root_dir:    base cache dir (empty = feature disabled).
    // params:      common_params of the loaded model (rope config, kv dtypes).
    // model:       the loaded model (arch + quant string + file stat).
    // ubatch_size: the chunk stride == llama's n_ubatch (--ubatch). the chunk
    //              boundaries are the UBatch boundaries (where state snapshots
    //              fire), NOT the n_batch (--batch) size. see the server-context
    //              construction for why -b != -ub desyncs if this is wrong.
    kv_chain_store(std::string root_dir, uint64_t limit_bytes, int32_t ubatch_size,
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
    // the returned chunks carry file PATHS, not blob data (see struct comment):
    // the caller replays them one at a time with read_chunk_file(), so the
    // whole chain is never materialized in RAM at once.
    std::vector<kv_chain_chunk> load_prefix(const llama_tokens & tokens, size_t * n_tokens) const;

    // reads + validates ONE chunk file (header: magic/version/n_tokens + token
    // IDs must match `expected_tokens`) and returns the blob. used by the
    // restore to stream the chain file-by-file (peak RAM = one file).
    // returns false if missing/corrupt/token-mismatch.
    bool read_chunk_file(const fs::path & file, std::vector<uint8_t> & out_blob,
                         const llama_tokens & expected_tokens) const;

    // per-chunk "rs file present" map from the LAST load_prefix() call (in
    // chain order, only the matched prefix; 1 = present). the replay loop needs
    // it for the mid-replay .kvcache-failure fallback: when a chunk's attn file
    // fails to read, truncate the restore at the deepest loaded chunk whose rs
    // file is present (its rs IS the valid recurrent tail).
    const std::vector<uint8_t> & last_rs_present() const { return last_rs_present_; }


    // touches (utimensat) the files of the given chunk hashes so their mtime is
    // "now". required because the default relatime mount does not update mtime
    // on read; without this, LRU eviction would evict the hottest chains first.
    // load_prefix() already touches on hit; this is for callers that restore
    // via a different path and want the same LRU semantics.
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
    // fills root_hash_. the model file is stat()ed only - never opened.
    void compute_root_hash(const common_params & params, const llama_model * model);

    bool write_chunk(const fs::path & dir, uint64_t chunk_hash, const llama_tokens & chunk_tokens,
                     const std::vector<uint8_t> & attn, const std::vector<uint8_t> & recr);
    bool write_chunk_file(const fs::path & tmp, const fs::path & file, uint64_t chunk_hash,
                          const llama_tokens & tokens, const std::vector<uint8_t> & blob);
    void evict_oldest(uint64_t need_bytes);

    std::string root_dir;
    uint64_t    root_hash_ = 0; // identity of this model/config; the chain's parent for chunk 0
    uint64_t    limit_bytes;
    int32_t     ubatch_size_; // chunk stride == n_ubatch (the ubatch boundary grid)
    uint64_t    total_bytes_cur = 0;
    mutable std::vector<uint8_t> last_rs_present_; // set by load_prefix() (const: it caches the last call's result); see last_rs_present()
};
