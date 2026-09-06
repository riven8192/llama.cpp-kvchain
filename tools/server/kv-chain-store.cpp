#include "kv-chain-store.h"

#include "server-common.h"

#include "../../src/llama-ext.h" // llama_model_arch_name

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>

#ifdef __linux__
#include <fcntl.h>    // AT_FDCWD
#include <sys/stat.h> // utimensat
#endif

namespace fs = std::filesystem;

static constexpr uint32_t KV_CHAIN_MAGIC   = 0x4b564331; // "KVC1"
// the single version number for the kv-chain format: the chunk FILE layout AND
// the root-hash metadata blob (both are fed into / checked against it).
// v3: split into .kvcache + .rscache (one blob per file)
// v4: .kvcache now holds an ATTN_ONLY blob (per-token KV rows only, no ring
//     states) and .rscache now holds a TAIL_ONLY blob dumped via the plain
//     llama_state_seq_get_data_ext (the full tail object: DSV4F compressed K
//     caches + compressor rings, or Qwen Gated DeltaNet R/S). the .kvcache blob
//     type changes for DSV4F (FULL_ONLY -> ATTN_ONLY), the .rscache call site
//     moves from the windowed-ext to the plain-ext, and the tail flag changes
//     (PARTIAL_ONLY -> TAIL_ONLY), so old files are a clean miss. bump whenever
//     EITHER changes: an old file with a stale version is never read (version
//     check), and the root hash changes, so old chunks are a clean miss.
// v5: n_seq_max (--parallel) added to the root-hash metadata blob. the attn
//     blob's n_stream scales with --parallel (with kv-unified off the cache has
//     n_stream == n_seq_max parallel cell arrays), so a chunk file written by a
//     server with a different --parallel used to fail restore with
//     "state_read: n_stream mismatch" and fall back to a 100% prefill. baking
//     it into the root hash turns that into a CLEAN MISS (different chain
//     names, no files match). the version is ALSO hashed into the root, so a
//     chain computed under an older version can never hit a file written by
//     this one.
static constexpr uint32_t KV_CHAIN_VERSION = 5;

kv_chain_store::kv_chain_store(std::string root_dir, uint64_t limit_bytes, int32_t ubatch_size,
                               const common_params & params, const llama_model * model) :
    root_dir(std::move(root_dir)), limit_bytes(limit_bytes), ubatch_size_(ubatch_size > 0 ? ubatch_size : 512) {
    if (!enabled()) {
        return;
    }

    compute_root_hash(params, model);

    std::error_code ec;
    const fs::path cache_dir = fs::path(this->root_dir);
    fs::create_directories(cache_dir, ec);
    if (ec) {
        SRV_ERR("kv-chain[storage]: failed to create cache dir '%s': %s (ec=%d)\n",
                cache_dir.string().c_str(), ec.message().c_str(), ec.value());
        this->root_dir.clear();
        this->root_hash_ = 0;
        return;
    }
    // remove stray .tmp files from an aborted previous run, then index existing chunks.
    // flat dir: files from all models/configs live here; total_bytes_cur sums
    // everything (eviction is global, LRU by mtime).
    for (const auto & entry : fs::directory_iterator(cache_dir, fs::directory_options::skip_permission_denied)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (entry.path().extension() == ".tmp") {
            fs::remove(entry.path(), ec);
            continue;
        }
        const auto size = static_cast<uint64_t>(entry.file_size());
        total_bytes_cur += size;
    }
    SRV_INF("kv-chain[storage]: cache dir '%s', root=%s, %zu bytes on disk, %.1f GiB (limit %.1f GiB), chunk ub=%d\n",
            cache_dir.string().c_str(), hash_str(root_hash_).c_str(), (size_t) total_bytes_cur,
            (double) total_bytes_cur / (1024.0*1024.0*1024.0),
            (double) limit_bytes / (1024.0*1024.0*1024.0),
            ubatch_size);
}

uint64_t kv_chain_store::fnv1a64(const uint8_t * data, size_t len) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < len; ++i) {
        h ^= data[i];
        h *= 1099511628211ULL;
    }
    return h;
}

uint64_t kv_chain_store::fnv1a64(uint64_t h, const uint8_t * data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        h ^= data[i];
        h *= 1099511628211ULL;
    }
    return h;
}

uint64_t kv_chain_store::hash_chunk(const llama_tokens & chunk_tokens, uint64_t prev_hash) const {
    // FNV-1a over this chunk's token ids, seeded by the parent hash (chain step).
    // the full 64-bit result is kept (the on-disk header stores its low 32 bits).
    uint64_t h = 1469598103934665603ULL ^ prev_hash;
    for (auto t : chunk_tokens) {
        const uint8_t * p = reinterpret_cast<const uint8_t *>(&t);
        for (int i = 0; i < (int) sizeof(t); ++i) {
            h ^= p[i];
            h *= 1099511628211ULL;
        }
    }
    return h;
}

std::vector<uint64_t> kv_chain_store::hash_chain(const llama_tokens & tokens) const {
    // the whole chain in one pass: chunk k = tokens[k*bs, (k+1)*bs), chained
    // off the previous chunk's hash (0 for the root). computed once at prompt
    // arrival; restore and save both index the result.
    std::vector<uint64_t> hashes;
    const size_t bs = (size_t) ubatch_size_;
    if (bs == 0 || tokens.empty()) {
        return hashes;
    }
    const size_t n_chunks = tokens.size() / bs;
    hashes.reserve(n_chunks);
    // the root chunk chains off the ROOT hash (the metadata identity: model,
    // config, version). this keeps different models/configs in disjoint hash
    // namespaces even for identical leading tokens.
    uint64_t prev = root_hash_;
    for (size_t k = 0; k < n_chunks; ++k) {
        const llama_tokens block(tokens.begin() + k * bs, tokens.begin() + (k + 1) * bs);
        prev = hash_chunk(block, prev);
        hashes.push_back(prev);
    }
    return hashes;
}

std::string kv_chain_store::hash_str(uint64_t h) {
    static const char * hex = "0123456789abcdef";
    std::string s(16, '0');
    for (int i = 15; i >= 0; --i) {
        s[i] = hex[h & 0xf];
        h >>= 4;
    }
    return s;
}

// feeds a field into a running FNV-1a hash. the field is length-prefixed (u64 LE)
// so field boundaries are unambiguous (no "ab"+"cd" vs "a"+"bcd" collisions).
static uint64_t hash_field(uint64_t h, const void * data, size_t len) {
    uint8_t lbuf[8];
    for (int i = 0; i < 8; ++i) {
        lbuf[i] = static_cast<uint8_t>((len >> (8 * i)) & 0xff);
    }
    h = kv_chain_store::fnv1a64(h, lbuf, 8);
    h = kv_chain_store::fnv1a64(h, reinterpret_cast<const uint8_t *>(data), len);
    return h;
}
static uint64_t hash_le(uint64_t h, uint64_t v, size_t nbytes) {
    uint8_t b[8] = {0};
    for (size_t i = 0; i < nbytes && i < 8; ++i) {
        b[i] = static_cast<uint8_t>((v >> (8 * i)) & 0xff);
    }
    return hash_field(h, b, nbytes);
}
static uint64_t hash_str_field(uint64_t h, const std::string & s) {
    return hash_field(h, s.data(), s.size());
}

void kv_chain_store::compute_root_hash(const common_params & params, const llama_model * model) {
    kv_chain_metadata md;
    md.format_version    = KV_CHAIN_VERSION;
    md.chunk_size        = ubatch_size_;
    md.model_file_size   = -1;
    md.model_file_mtime  = -1;
    md.arch              = "";
    md.ftype             = "";
    md.type_k            = params.cache_type_k;
    md.type_v            = params.cache_type_v;
    md.rope_scaling_type = params.rope_scaling_type;
    // --parallel becomes the ctx's n_seq_max (common/common.cpp: cparams.n_seq_max
    // = params.n_parallel). with kv-unified off the attn cache has n_stream ==
    // n_seq_max parallel cell arrays, so a different --parallel changes the attn
    // blob LAYOUT -> it must change the root hash (clean miss, not a mismatch).
    // the .rscache blob never carries n_stream (a single slot's R/S), but a
    // clean miss is still the right semantics: cross-parallel reuse is not
    // meaningful anyway.
    md.n_seq_max           = static_cast<uint32_t>(params.n_parallel > 0 ? params.n_parallel : 1);
    {
        // copy the float bits portably (memcpy, not a type-punned pointer cast)
        const float rope_freq_base  = params.rope_freq_base;  // 0.0f = "from model"
        const float rope_freq_scale = params.rope_freq_scale; // 0.0f = "from model"
        std::memcpy(&md.rope_freq_base_bits,  &rope_freq_base,  sizeof(uint32_t));
        std::memcpy(&md.rope_freq_scale_bits, &rope_freq_scale, sizeof(uint32_t));
    }

    std::string model_path;
    if (model != nullptr) {
        md.arch  = llama_model_arch_name(model);
        md.ftype = llama_ftype_name(llama_model_ftype(model));
        model_path = params.model.path;
        if (!model_path.empty()) {
            // stat only - never open the (multi-GB) model file
            std::error_code ec;
            const auto st = fs::status(model_path, ec);
            if (!ec && st.type() == fs::file_type::regular) {
                md.model_file_size  = static_cast<int64_t>(fs::file_size(model_path, ec));
                if (!ec) {
                    md.model_file_mtime = std::chrono::duration_cast<std::chrono::seconds>(
                        fs::last_write_time(model_path, ec).time_since_epoch()).count();
                    if (ec) {
                        md.model_file_mtime = -1;
                    }
                }
            }
        }
    }
    if (model == nullptr) {
        SRV_WRN("%s", "kv-chain[storage]: no model handle, root hash will not include arch/ftype/file identity");
    }

    // canonical serialization: every field, length-prefixed, in struct order.
    // the model path is hashed as a string (path+size+mtime identifies the file
    // content; a weight hash would take too long).
    uint64_t h = 1469598103934665603ULL;
    h = hash_le(h, static_cast<uint64_t>(md.format_version), 4);
    h = hash_le(h, static_cast<uint64_t>(md.chunk_size),       4);
    h = hash_le(h, static_cast<uint64_t>(md.model_file_size),  8);
    h = hash_le(h, static_cast<uint64_t>(md.model_file_mtime), 8);
    h = hash_str_field(h, md.arch);
    h = hash_str_field(h, md.ftype);
    h = hash_le(h, static_cast<uint64_t>(md.type_k),           4);
    h = hash_le(h, static_cast<uint64_t>(md.type_v),           4);
    h = hash_le(h, static_cast<uint64_t>(md.rope_scaling_type),4);
    h = hash_le(h, md.rope_freq_base_bits,  4);
    h = hash_le(h, md.rope_freq_scale_bits, 4);
    h = hash_le(h, md.n_seq_max,            4);
    h = hash_str_field(h, model_path);

    root_hash_ = h;
    {
        float rope_freq_base  = 0.0f;
        float rope_freq_scale = 0.0f;
        std::memcpy(&rope_freq_base,  &md.rope_freq_base_bits,  sizeof(float));
        std::memcpy(&rope_freq_scale, &md.rope_freq_scale_bits, sizeof(float));
        SRV_INF("kv-chain[storage]: metadata: version=%d chunk_size=%d model='%s' size=%lld mtime=%lld arch='%s' ftype='%s' type_k=%d type_v=%d rope=(%d,%.6g,%.6g) n_seq_max=%d\n",
                md.format_version, md.chunk_size, model_path.c_str(),
                (long long) md.model_file_size, (long long) md.model_file_mtime,
                md.arch.c_str(), md.ftype.c_str(),
                (int) md.type_k, (int) md.type_v,
                md.rope_scaling_type, rope_freq_base, rope_freq_scale,
                (int) md.n_seq_max);
    }
}

// dumps one seq-state blob for the window [pos_lo, pos_hi) with the given
// part-selection flags. used for the per-chunk .kvcache file (ATTN_ONLY =
// per-token KV rows only). returns an empty vector on failure.
static std::vector<uint8_t> dump_window(llama_context * ctx, llama_seq_id seq_id,
                                        llama_pos pos_lo, llama_pos pos_hi, llama_state_seq_flags flags) {
    const size_t size = llama_state_seq_get_size_window_ext(ctx, seq_id, flags, pos_lo, pos_hi);
    if (size == 0) {
        return {};
    }
    std::vector<uint8_t> blob(size);
    const size_t got = llama_state_seq_get_data_window_ext(ctx, blob.data(), size, seq_id, flags, pos_lo, pos_hi);
    if (got == 0 || got != size) {
        SRV_WRN("kv-chain[storage]: failed to get window state (%zu of %zu bytes)\n", got, size);
        return {};
    }
    return blob;
}

// dumps the TAIL object via the PLAIN llama_state_seq_get_data_ext with
// TAIL_ONLY (NOT the windowed variant, and NOT flags=0). it is a "last write
// wins" state, only valid at exact chunk boundaries, and the restore only ever
// loads the LAST chunk's copy.
//
// WHY TAIL_ONLY and NOT flags=0 (FULL): on llama_kv_cache_dsv4, a FULL-mode
// blob STARTS with kv_raw (the per-token rows), and its state_read clears
// kv_raw before loading it. the .rscache is loaded AFTER the per-chunk
// .kvcache replay, so a FULL blob would WIPE the just-restored per-token rows
// (garbled output). PARTIAL_ONLY is wrong the other way: it saves kv_raw +
// rings but NOT the compressed K caches, while the tail prefill does NOT
// recompute them (it only processes tokens >= n_saved) - so a PARTIAL_ONLY
// tail also garbled DSV4F. TAIL_ONLY is the complement of the .kvcache files:
// on dsv4 it is the compressed K caches + the compressor rings (no kv_raw);
// on Qwen / pure-attn caches it behaves exactly like PARTIAL_ONLY.
static std::vector<uint8_t> dump_tail(llama_context * ctx, llama_seq_id seq_id) {
    const size_t size = llama_state_seq_get_size_ext(ctx, seq_id, LLAMA_STATE_SEQ_FLAGS_TAIL_ONLY);
    if (size == 0) {
        return {};
    }
    std::vector<uint8_t> blob(size);
    const size_t got = llama_state_seq_get_data_ext(ctx, blob.data(), size, seq_id, LLAMA_STATE_SEQ_FLAGS_TAIL_ONLY);
    if (got == 0 || got != size) {
        SRV_WRN("kv-chain[storage]: failed to get tail state (%zu of %zu bytes)\n", got, size);
        return {};
    }
    return blob;
}

// save one chunk covering the window [pos_lo, pos_hi). the .kvcache file holds
// the ATTN_ONLY blob (per-token KV rows for exactly this window [pos_lo,
// pos_hi), additive across chunks). the .rscache file holds the FULL tail
// snapshot (kv_raw + the DSV4 compressed K caches + the compressor rings; on
// Qwen, attn + recr), dumped via the plain _ext at this chunk boundary - a
// "last write wins" state that is only valid at exact chunk boundaries, and the
// restore loads only the LAST chunk's copy. chunk_tokens are the window's
// tokens, stored verbatim in both file headers.
bool kv_chain_store::save(llama_context * ctx, llama_seq_id seq_id, llama_pos pos_lo, llama_pos pos_hi,
                          uint64_t chunk_hash, const llama_tokens & chunk_tokens,
                          uint64_t parent_hash /* = 0, [DEBUG] logging only */) {
    if (!enabled() || ctx == nullptr || chunk_tokens.empty() || pos_hi <= pos_lo) {
        return false;
    }

    // ATTN_ONLY (not FULL_ONLY): on a DSV4 cache FULL_ONLY would also emit the
    // compressor ring states, which must NOT be in the per-chunk file (they are
    // the tail object stored separately in .rscache). on Qwen / pure-attn caches
    // ATTN_ONLY behaves identically to FULL_ONLY.
    std::vector<uint8_t> attn = dump_window(ctx, seq_id, pos_lo, pos_hi, LLAMA_STATE_SEQ_FLAGS_ATTN_ONLY);
    std::vector<uint8_t> recr = dump_tail(ctx, seq_id);
    if (attn.empty() || recr.empty()) {
        SRV_WRN("kv-chain[storage]: failed to dump chunk window [%d, %d)\n", (int) pos_lo, (int) pos_hi);
        return false;
    }

    const fs::path dir = fs::path(root_dir);

    // [DEBUG] every save attempt: the hash the file will be named by (and the
    // parent it was computed from), plus whether both files already existed
    // (idempotent skip) or were written.
    SRV_INF("kv-chain[storage]: saving chunk hash=%s parent=%s tokens=[%d..%d) existing=(kv=%d rs=%d)\n",
            hash_str(chunk_hash).c_str(), hash_str(parent_hash).c_str(),
            (int) pos_lo, (int) pos_hi,
            (int) fs::exists(dir / (hash_str(chunk_hash) + ".kvcache")),
            (int) fs::exists(dir / (hash_str(chunk_hash) + ".rscache")));

    return write_chunk(dir, chunk_hash, chunk_tokens, attn, recr);
}

// writes both chunk files (.kvcache + .rscache) if not present.
// idempotent: if one file already exists, only the missing one is written
// (handles crash-between-the-two-writes). returns true if anything was written.
bool kv_chain_store::write_chunk(const fs::path & dir, uint64_t chunk_hash, const llama_tokens & chunk_tokens,
                                  const std::vector<uint8_t> & attn, const std::vector<uint8_t> & recr) {
    const std::string stem = hash_str(chunk_hash);
    const fs::path kv_file = dir / (stem + ".kvcache");
    const fs::path rs_file = dir / (stem + ".rscache");

    const bool kv_exists = fs::exists(kv_file);
    const bool rs_exists = fs::exists(rs_file);
    if (kv_exists && rs_exists) {
        return false;
    }

    // per-file header overhead: magic + version + hash32 + n_tokens + tokens + blob_size
    // (no trailing checksum - see read_chunk_file)
    const uint64_t hdr_bytes = 4 * sizeof(uint32_t) + sizeof(llama_token) * chunk_tokens.size()
                              + sizeof(uint32_t);
    const uint64_t kv_entry = attn.size() + hdr_bytes;
    const uint64_t rs_entry = recr.size() + hdr_bytes;
    const uint64_t need_bytes = (kv_exists ? 0 : kv_entry) + (rs_exists ? 0 : rs_entry);

    if (limit_bytes > 0 && total_bytes_cur + need_bytes > limit_bytes) {
        evict_oldest(need_bytes);
    }

    bool any_written = false;
    if (!kv_exists) {
        const fs::path tmp = dir / (stem + ".kvcache.tmp");
        if (write_chunk_file(tmp, kv_file, chunk_hash, chunk_tokens, attn)) {
            total_bytes_cur += kv_entry;
            any_written = true;
        }
    }
    if (!rs_exists) {
        const fs::path tmp = dir / (stem + ".rscache.tmp");
        if (write_chunk_file(tmp, rs_file, chunk_hash, chunk_tokens, recr)) {
            total_bytes_cur += rs_entry;
            any_written = true;
        }
    }
    if (any_written) {
        SRV_INF("kv-chain[storage]: saved chunk hash=%s (attn %.1f MiB, recr %.1f MiB)\n",
                stem.c_str(),
                (double) attn.size() / (1024.0*1024.0), (double) recr.size() / (1024.0*1024.0));
    }
    return any_written;
}

// writes one chunk file (header + single blob) to tmp, then renames.
bool kv_chain_store::write_chunk_file(const fs::path & tmp, const fs::path & file, uint64_t chunk_hash,
                                      const llama_tokens & tokens, const std::vector<uint8_t> & blob) {
    std::error_code ec;
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) {
            SRV_ERR("kv-chain[storage]: failed to open '%s' for writing\n", tmp.string().c_str());
            return false;
        }
        const uint32_t magic   = KV_CHAIN_MAGIC;
        const uint32_t version = KV_CHAIN_VERSION;
        const uint32_t hash32  = static_cast<uint32_t>(chunk_hash);
        const uint32_t n_tok   = static_cast<uint32_t>(tokens.size());
        const uint32_t blob_sz = static_cast<uint32_t>(blob.size());
        f.write(reinterpret_cast<const char *>(&magic),  sizeof(magic));
        f.write(reinterpret_cast<const char *>(&version),sizeof(version));
        f.write(reinterpret_cast<const char *>(&hash32), sizeof(hash32));
        f.write(reinterpret_cast<const char *>(&n_tok),  sizeof(n_tok));
        f.write(reinterpret_cast<const char *>(tokens.data()), sizeof(llama_token) * tokens.size());
        f.write(reinterpret_cast<const char *>(&blob_sz),  sizeof(blob_sz));
        f.write(reinterpret_cast<const char *>(blob.data()), blob.size());
        if (!f) {
            SRV_ERR("kv-chain[storage]: write failed for '%s'\n", tmp.string().c_str());
            fs::remove(tmp, ec);
            return false;
        }
        // no trailing checksum: read_chunk_file does not verify one (we trust
        // the storage device; verifying costs a full pass over every recr file).
        f.close();
    }

    fs::rename(tmp, file, ec);
    if (ec) {
        SRV_ERR("kv-chain[storage]: rename failed for '%s': %s\n", file.string().c_str(), ec.message().c_str());
        fs::remove(tmp, ec);
        return false;
    }
    return true;
}

// reads + validates ONE chunk file (.kvcache or .rscache) and returns the blob.
// the header's token IDs must match expected_tokens verbatim (collision guard:
// a hash collision or a stale file is a clean miss, not a garbage restore).
// returns false if missing/corrupt/token-mismatch.
//
// MEMORY: this is the ONLY place a chunk file's contents are materialized. the
// restore calls it once per file and frees the blob before the next (see
// load_prefix / the server-context replay loop), so peak RAM = one file (~16
// MiB attn, ~150 MiB for the tail rs) - never the whole chain. the header is
// parsed from a small fixed buffer; the blob is streamed straight into
// out_blob (no second copy).
bool kv_chain_store::read_chunk_file(const fs::path & file, std::vector<uint8_t> & out_blob,
                                     const llama_tokens & expected_tokens) const {
    std::ifstream f(file, std::ios::binary);
    if (!f) {
        return false;
    }
    f.seekg(0, std::ios::end);
    const uint64_t file_size = static_cast<uint64_t>(f.tellg());
    const uint64_t min_size = sizeof(uint32_t) * 4 + sizeof(uint32_t); // magic/version/hash32/n_tokens + blob_size
    if (file_size < min_size) {
        return false;
    }
    f.seekg(0, std::ios::beg);

    // header is fixed-size up to n_tokens; the token array follows. read the
    // 4 u32s first, then the tokens, then the blob_size u32.
    uint32_t hdr[4];
    f.read(reinterpret_cast<char *>(hdr), sizeof(hdr));
    if (!f) {
        return false;
    }
    const uint32_t magic   = hdr[0];
    const uint32_t version = hdr[1];
    // hdr[2] = hash32 (the file name; not re-verified here - the name IS the hash)
    const uint32_t n_tokens = hdr[3];
    // NOTE: there is NO trailing checksum in the file. it was removed from the
    // writer because verifying it costs a full pass over every (multi-hundred-
    // MiB) recr file in the chain, which dominated the restore time. we trust
    // the storage device; a silent bit-flip would surface as garbage model
    // output, not as a clean cache miss - accepted trade-off for performance
    // (the size checks below still catch layout corruption).
    if (magic != KV_CHAIN_MAGIC || version != KV_CHAIN_VERSION) {
        SRV_WRN("kv-chain[storage]: bad magic/version in %s (magic=%08x version=%u), ignoring\n",
                file.string().c_str(), magic, version);
        return false;
    }
    // bound the token count before allocating (n_tokens * 4 bytes must fit the file)
    if (n_tokens == 0 || (uint64_t) n_tokens > (file_size - min_size) / sizeof(llama_token)) {
        return false;
    }
    std::vector<llama_token> file_tokens(n_tokens);
    f.read(reinterpret_cast<char *>(file_tokens.data()), sizeof(llama_token) * n_tokens);
    if (!f) {
        return false;
    }
    uint32_t blob_size;
    f.read(reinterpret_cast<char *>(&blob_size), sizeof(blob_size));
    if (!f) {
        return false;
    }
    const uint64_t hdr_len = sizeof(hdr) + sizeof(llama_token) * n_tokens + sizeof(blob_size);
    if (hdr_len + blob_size != file_size) {
        SRV_WRN("kv-chain[storage]: size mismatch in %s (header %llu + blob %u != file %llu), ignoring\n",
                file.string().c_str(), (unsigned long long) hdr_len, blob_size, (unsigned long long) file_size);
        return false;
    }
    // token IDs must match the prompt verbatim (self-describing files)
    if (file_tokens.size() != expected_tokens.size() ||
        std::memcmp(file_tokens.data(), expected_tokens.data(), sizeof(llama_token) * n_tokens) != 0) {
        SRV_WRN("kv-chain[storage]: token mismatch in %s (n=%u vs %zu), treating as miss\n",
                file.string().c_str(), n_tokens, expected_tokens.size());
        return false;
    }
    // stream the blob straight out (one copy: disk -> out_blob)
    out_blob.resize(blob_size);
    if (blob_size > 0) {
        f.read(reinterpret_cast<char *>(out_blob.data()), blob_size);
        if (!f) {
            return false;
        }
    }
    return true;
}

void kv_chain_store::evict_oldest(uint64_t need_bytes) {
    std::error_code ec;
    // flat scan of the cache dir over BOTH .kvcache and .rscache files.
    // sorted by mtime ascending; delete from the front until we have room.
    // no tree-integrity checks: deleting an old rs file just lowers `usable`
    // for chains that would have used it; deleting a kv file shortens the kv
    // chain. both are benign "cache ends here" on the next restore.
    const fs::path cache_dir = fs::path(root_dir);
    struct ent { fs::path p; uint64_t size; std::filesystem::file_time_type mtime; };
    std::vector<ent> all;
    for (const auto & e : fs::directory_iterator(cache_dir, ec)) {
        if (!e.is_regular_file()) {
            continue;
        }
        const auto ext = e.path().extension();
        if (ext == ".kvcache" || ext == ".rscache") {
            all.push_back({ e.path(), (uint64_t) e.file_size(), e.last_write_time(ec) });
        }
    }
    std::sort(all.begin(), all.end(), [](const ent & a, const ent & b) {
        return a.mtime < b.mtime;
    });
    for (auto & e : all) {
        if (limit_bytes == 0 || total_bytes_cur + need_bytes <= limit_bytes) {
            break;
        }
        if (fs::remove(e.p, ec)) {
            total_bytes_cur -= e.size;
            SRV_INF("kv-chain[storage]: evicted %s (%.1f MiB)\n", e.p.filename().string().c_str(), e.size / (1024.0*1024.0));
        }
    }
}

// sets the mtime+atime of every file in `files` to "now" (utimensat).
// a failure on any single file is non-fatal (the file may have been evicted).
static void touch_chain_files(const std::vector<fs::path> & files) {
    if (files.empty()) {
        return;
    }
#ifdef __linux__
    const struct timespec now[2] = { { 0, UTIME_NOW }, { 0, UTIME_NOW } };
    for (const auto & p : files) {
        (void) utimensat(AT_FDCWD, p.c_str(), now, 0);
    }
#else
    (void) files; // no-op on non-linux
#endif
}

std::vector<kv_chain_chunk> kv_chain_store::load_prefix(const llama_tokens & tokens, size_t * n_tokens) const {
    std::vector<kv_chain_chunk> chunks;
    *n_tokens = 0;
    if (!enabled() || tokens.empty()) {
        return chunks;
    }

    const fs::path dir = fs::path(root_dir);
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        return chunks;
    }

    // TWO-PHASE restore. the chunk files are large (~2 MiB attn + ~150 MiB
    // recr per 32-token chunk), so the expensive part is reading them - not
    // finding them. therefore:
    //   phase 1 (find): walk the hash chain with fs::exists() ONLY - no file
    //     contents are read. this locates the chain break (first missing kv
    //     file) and `usable` = the last chunk whose rs file is present.
    //   phase 2 (read): read file contents ONLY for the chunks we will replay:
    //     the .kvcache of every chunk in [0, usable), and the .rscache of
    //     chunk (usable-1) ALONE - the recurrent state is a tail object (last
    //     write wins), so the earlier rs files are superseded and never read.
    // a failed read mid-way is a benign "cache ends here" (eviction race).
    const size_t bs = (size_t) ubatch_size_;
    const size_t n_chunks = tokens.size() / bs;

    // the chain was computed once at prompt arrival; this walk only LOOKS files
    // up by the precomputed names. (recomputing here would be a second source of
    // truth - the save hook indexes the same vector, so the two can never drift)
    const std::vector<uint64_t> hashes = hash_chain(tokens);

    // ---- phase 1: exists-only walk ----------------------------------------
    std::vector<std::string> stems;          // per found chunk: file name stem
    std::vector<bool> rs_present;            // per found chunk: rs file exists
    size_t n_kv_found = 0;
    size_t usable = 0; // 1-based: last chunk with an rs file
    for (size_t k = 0; k < n_chunks; ++k) {
        const std::string stem = hash_str(hashes[k]);
        const bool kv_exists = fs::exists(dir / (stem + ".kvcache"));
        const bool rs_exists = kv_exists && fs::exists(dir / (stem + ".rscache"));
        // [DEBUG] every find attempt: the precomputed name for chunk k and what
        // is on disk for it. a 'kv=0' here means the walk stops.
        SRV_DBG("kv-chain[storage]: chunk %zu hash=%s kv=%d rs=%d\n", k, stem.c_str(),
                (int) kv_exists, (int) rs_exists);
        if (!kv_exists) {
            break;
        }
        stems.push_back(std::move(stem));
        rs_present.push_back(rs_exists);
        n_kv_found++;
        if (rs_exists) {
            usable = k + 1;
        }
    }

    // NEVER restore the whole prompt: always leave at least one token to prefill.
    //
    // Logits are NOT state - they are the OUTPUT of a forward pass and are not in
    // the cache files. A restore that covers all N prompt tokens leaves nothing to
    // run a forward pass over, so no logits are ever produced for the sampler:
    // the slot skips sampling entirely (i_batch == -1) or trips
    // "corrupt output buffer (n_outputs=0)" -> GGML_ABORT in llama_get_logits_ith.
    // That only happens when the prompt length is an exact multiple of bs (so the
    // final chunk ends exactly at the last prompt token).
    //
    // Fix: drop the last chunk, re-prefilling bs tokens. Rejected alternatives:
    // rolling back one token needs an invertible recurrent state (Gated DeltaNet
    // is not; llama_memory_recurrent::seq_rm's bounded rollback needs n_rs_seq > 0,
    // which is speculative-decoding-only and 0 here), and re-running the last token
    // on top of the restored state + reloading the rs blob to repair the corrupted
    // recurrent state works but is a lot of machinery for a rare case.
    //
    // The cost is negligible: this triggers only when len % bs == 0, i.e. 1 prompt
    // in bs. Amortized it adds ~1 token of prefill per request - it just shifts the
    // reuse threshold by one, and the len % bs == bs-1 case already prefills bs-1.
    if (usable == n_chunks && n_chunks > 0 && tokens.size() % bs == 0) {
        usable--;
        SRV_INF("kv-chain[storage]: prompt length %zu is an exact multiple of the chunk size %zu; "
                "capping the restore at %zu chunks so the last chunk is re-prefilled "
                "(a forward pass is required to produce logits)\n",
                tokens.size(), bs, usable);
    }

    *n_tokens = 0;
    if (usable == 0) {
        SRV_INF("kv-chain[storage]: %zu prompt chunks, no usable chain (no kv+rs pair on disk)\n", n_chunks);
        return chunks;
    }

    // ---- phase 2: resolve the replay plan (NO blob data is read here) ------
    // the matched chunks are returned as file PATHS only; the caller replays
    // them one at a time (read file -> set_data -> free, next file), so the
    // whole chain is never materialized in RAM at once (on a 128 GB box running
    // a 110 GB model, the old "read everything first" behavior peaked at ~2x
    // the chain size in RAM).
    //
    // validation happens in TWO places, with different failure semantics:
    //   * the TAIL .rscache is validated HERE (cheap: one 150 MiB read, and its
    //     failure mode is special - the recurrent tail is a single fixed-size
    //     object with no earlier fallback, so a bad tail discards the ENTIRE
    //     restore and the corrupt file is deleted).
    //   * the per-chunk .kvcache files are validated by the CALLER during the
    //     replay (one read each, streamed). a .kvcache that fails to open/parse
    //     mid-replay is a benign "cache ends here" (eviction race, manual
    //     deletion): the caller truncates the restore at the last chunk with a
    //     valid rs file (rs_present[] is exposed for that) and prefills the rest.
    stems.resize(usable);
    rs_present.resize(usable);
    // expose for the caller's mid-replay fallback (vector<uint8_t>, not
    // vector<bool>: the latter's proxy references are non-const-assignable)
    last_rs_present_.resize(usable);
    for (size_t k = 0; k < usable; ++k) {
        last_rs_present_[k] = rs_present[k] ? 1 : 0;
    }

    const fs::path rs_file_tail = dir / (stems[usable - 1] + ".rscache");
    const llama_tokens tail_block(tokens.begin() + (usable - 1) * bs, tokens.begin() + usable * bs);
    std::vector<uint8_t> tail_blob; // validated here, then freed; the CALLER re-reads the file at replay time
    if (!read_chunk_file(rs_file_tail, tail_blob, tail_block)) {
        SRV_WRN("kv-chain[storage]: load_prefix: rs read failed/mismatch at tail chunk %zu, discarding entire restore\n", usable - 1);
        // if the file is still on disk it is corrupt/stale: delete it so
        // it is not re-read (and re-deleted) on the next restore.
        std::error_code dec;
        if (fs::exists(rs_file_tail, dec) && fs::remove(rs_file_tail, dec)) {
            SRV_WRN("kv-chain[storage]: load_prefix: deleted corrupt rs file %s\n", rs_file_tail.filename().string().c_str());
        }
        *n_tokens = 0;
        return chunks;
    }
    tail_blob.clear();
    tail_blob.shrink_to_fit();

    for (size_t k = 0; k < usable; ++k) {
        kv_chain_chunk chunk;
        chunk.attn_file = dir / (stems[k] + ".kvcache");
        chunk.tokens.assign(tokens.begin() + k * bs, tokens.begin() + (k + 1) * bs);
        if (k + 1 == usable) {
            chunk.recr_file = rs_file_tail; // TAIL chunk only (validated above)
        }
        chunks.push_back(std::move(chunk));
    }
    *n_tokens = usable * bs;

    // touch exactly the files that will be read and replayed:
    // chunks[0..usable-1]'s .kvcache + the .rscache of the TAIL chunk
    // (usable-1), PLUS the .rscache of every chunk index that is a multiple
    // of KV_CHAIN_RS_TOUCH_STRIDE below the tail (0, 8, 16, ... < usable-1,
    // plus the tail itself if it happens to land on the grid).
    // WHY: a future prompt may fork off this chain at one of those
    // intermediate boundaries (a shorter shared prefix). if we only touched
    // the tail rs file, LRU eviction would prune the intermediate rs files
    // even though a fork at that boundary still needs its recurrent state -
    // the fork would silently degrade to a full re-prefill of the shared
    // trunk. touching the grid-stride rs files keeps the most useful fork
    // points alive without paying for every chunk's rs file.
    {
        static constexpr size_t KV_CHAIN_RS_TOUCH_STRIDE = 8;
        std::vector<fs::path> to_touch;
        to_touch.reserve(usable * 2);
        for (size_t k = 0; k < usable; ++k) {
            to_touch.push_back(chunks[k].attn_file);
        }
        const size_t tail = usable - 1;
        for (size_t k = 0; k < tail; k += KV_CHAIN_RS_TOUCH_STRIDE) {
            if (rs_present[k]) {
                to_touch.push_back(dir / (stems[k] + ".rscache"));
            }
        }
        to_touch.push_back(rs_file_tail);
        touch_chain_files(to_touch);
    }

    SRV_INF("kv-chain[storage]: %zu prompt chunks, first %zu kv-files found on disk, last rs-file found for chunk %zu, replay plan: %zu chunks\n",
            n_chunks, n_kv_found, usable, chunks.size());
    return chunks;
}

void kv_chain_store::touch_chunks(const std::vector<uint64_t> & chunk_hashes) const {
    if (!enabled() || chunk_hashes.empty()) {
        return;
    }
    const fs::path dir = fs::path(root_dir);
    std::vector<fs::path> files;
    files.reserve(chunk_hashes.size() * 2);
    for (uint64_t h : chunk_hashes) {
        files.push_back(dir / (hash_str(h) + ".kvcache"));
        files.push_back(dir / (hash_str(h) + ".rscache"));
    }
    touch_chain_files(files);
}
