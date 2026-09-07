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
// single version number for the kv-chain format: the chunk file layout AND the
// root-hash metadata blob. bump on any change to either: a stale file is never
// read (version check) and the root hash changes, so old chunks are a clean miss.
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
    // remove stray .tmp files from an aborted previous run, then index existing
    // chunks (eviction is global over the flat dir, LRU by mtime).
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
    std::vector<uint64_t> hashes;
    const size_t bs = (size_t) ubatch_size_;
    if (bs == 0 || tokens.empty()) {
        return hashes;
    }
    const size_t n_chunks = tokens.size() / bs;
    hashes.reserve(n_chunks);
    // chain chunk 0 off the root hash: keeps different models/configs in disjoint
    // hash namespaces even for identical leading tokens.
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

// feeds a field into a running FNV-1a hash, length-prefixed (u64 LE) so field
// boundaries are unambiguous.
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
    // the attn blob's n_stream scales with --parallel, so a different value
    // changes the blob layout -> it must change the root hash (clean miss).
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

// dumps one seq-state blob for the window [pos_lo, pos_hi). returns {} on failure.
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

// dumps the TAIL object (the recurrent/fixed-size state, "last write wins",
// valid only at chunk boundaries) via the plain get_data_ext with TAIL_ONLY.
// TAIL_ONLY, not FULL (flags=0): on a DSV4 cache a FULL-mode blob starts with
// kv_raw and its state_read clears kv_raw first, which would wipe the
// per-token rows restored from the .kvcache files (loaded after this). it is
// not PARTIAL_ONLY either: that omits the compressed K caches, which the tail
// prefill does not recompute. TAIL_ONLY is the exact complement of the
// per-chunk .kvcache files.
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

// save one chunk covering the window [pos_lo, pos_hi): the .kvcache file gets
// the ATTN_ONLY rows for exactly this window (additive across chunks), the
// .rscache file gets the tail snapshot (only the last chunk's copy is ever
// read). chunk_tokens are stored verbatim in both headers.
bool kv_chain_store::save(llama_context * ctx, llama_seq_id seq_id, llama_pos pos_lo, llama_pos pos_hi,
                          uint64_t chunk_hash, const llama_tokens & chunk_tokens,
                          uint64_t parent_hash /* = 0, logging only */) {
    if (!enabled() || ctx == nullptr || chunk_tokens.empty() || pos_hi <= pos_lo) {
        return false;
    }

    // ATTN_ONLY, not FULL_ONLY: on a DSV4 cache FULL_ONLY would also emit the
    // compressor ring states, which belong in the .rscache tail, not here.
    std::vector<uint8_t> attn = dump_window(ctx, seq_id, pos_lo, pos_hi, LLAMA_STATE_SEQ_FLAGS_ATTN_ONLY);
    std::vector<uint8_t> recr = dump_tail(ctx, seq_id);
    if (attn.empty() || recr.empty()) {
        SRV_WRN("kv-chain[storage]: failed to dump chunk window [%d, %d)\n", (int) pos_lo, (int) pos_hi);
        return false;
    }

    const fs::path dir = fs::path(root_dir);

    SRV_INF("kv-chain[storage]: saving chunk hash=%s parent=%s tokens=[%d..%d) existing=(kv=%d rs=%d)\n",
            hash_str(chunk_hash).c_str(), hash_str(parent_hash).c_str(),
            (int) pos_lo, (int) pos_hi,
            (int) fs::exists(dir / (hash_str(chunk_hash) + ".kvcache")),
            (int) fs::exists(dir / (hash_str(chunk_hash) + ".rscache")));

    return write_chunk(dir, chunk_hash, chunk_tokens, attn, recr);
}

// writes both chunk files if not present; idempotent (a crash between the two
// writes only re-writes the missing one). returns true if anything was written.
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

    // per-file header: magic + version + hash32 + n_tokens + tokens + blob_size
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

// reads + validates one chunk file and returns the blob. the header's token
// IDs must match expected_tokens verbatim (a collision/stale file is a clean
// miss, not a garbage restore). returns false if missing/corrupt/mismatched.
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

    uint32_t hdr[4];
    f.read(reinterpret_cast<char *>(hdr), sizeof(hdr));
    if (!f) {
        return false;
    }
    const uint32_t magic   = hdr[0];
    const uint32_t version = hdr[1];
    // hdr[2] = hash32 (the file name; not re-verified - the name IS the hash)
    const uint32_t n_tokens = hdr[3];
    // no trailing checksum: verifying one costs a full pass over every recr
    // file, which dominated restore time; we trust the storage device.
    if (magic != KV_CHAIN_MAGIC || version != KV_CHAIN_VERSION) {
        SRV_WRN("kv-chain[storage]: bad magic/version in %s (magic=%08x version=%u), ignoring\n",
                file.string().c_str(), magic, version);
        return false;
    }
    // bound n_tokens before allocating
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
    // the token IDs must match the prompt verbatim
    if (file_tokens.size() != expected_tokens.size() ||
        std::memcmp(file_tokens.data(), expected_tokens.data(), sizeof(llama_token) * n_tokens) != 0) {
        SRV_WRN("kv-chain[storage]: token mismatch in %s (n=%u vs %zu), treating as miss\n",
                file.string().c_str(), n_tokens, expected_tokens.size());
        return false;
    }
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
    // delete oldest (by mtime) until we have room. no tree-integrity checks:
    // an evicted file just shortens the chain on the next restore.
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

// sets atime+mtime of every file to "now"; a per-file failure is non-fatal.
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

    // two phases, because the files are large and reading dominates:
    //   phase 1 (find): walk the chain with fs::exists() only - locate the break
    //     (first missing kv file) and usable = last chunk with an rs file.
    //   phase 2 (plan): resolve file paths for the replay; only the TAIL rs file
    //     is read here (it is validated up front). the caller streams the
    //     .kvcache files one at a time; a read failure mid-replay discards the
    //     whole restore (100% re-prefill), since the loaded rows would be orphaned.
    const size_t ubs = (size_t) ubatch_size_;

    // the last token is never restored: it must be prefilled to produce logits.
    const size_t n_search = (tokens.size() > 0) ? tokens.size() - 1 : 0;
    const std::vector<uint64_t> hashes = hash_chain(std::vector<llama_token>(
            tokens.begin(), tokens.begin() + n_search));
    const size_t n_chunks = n_search / ubs;

    // phase 1: exists-only walk
    std::vector<std::string> stems;          // per found chunk: file name stem
    std::vector<bool> rs_present;            // per found chunk: rs file exists
    size_t n_kv_found = 0;
    size_t usable = 0; // 1-based: last chunk with an rs file
    for (size_t k = 0; k < n_chunks; ++k) {
        const std::string stem = hash_str(hashes[k]);
        const bool kv_exists = fs::exists(dir / (stem + ".kvcache"));
        const bool rs_exists = kv_exists && fs::exists(dir / (stem + ".rscache"));
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

    *n_tokens = 0;
    if (usable == 0) {
        SRV_INF("kv-chain[storage]: %zu prompt chunks, no usable chain (no kv+rs pair on disk)\n", n_chunks);
        return chunks;
    }

    // phase 2: resolve the replay plan (no blob data is read here except the
    // tail rs, validated up front: it is a single fixed-size object with no
    // earlier fallback, so a bad tail discards the whole restore, and the
    // corrupt file is deleted). the per-chunk .kvcache files are validated by
    // the caller during the replay; a read failure there discards the whole
    // restore too (the loaded rows would be orphaned - no valid recurrent
    // state to resume from), i.e. 100% re-prefill.
    stems.resize(usable);
    rs_present.resize(usable);
    // (vector<uint8_t>, not vector<bool>: the latter's proxies are not const-assignable)
    last_rs_present_.resize(usable);
    for (size_t k = 0; k < usable; ++k) {
        last_rs_present_[k] = rs_present[k] ? 1 : 0;
    }

    const fs::path rs_file_tail = dir / (stems[usable - 1] + ".rscache");
    const llama_tokens tail_block(tokens.begin() + (usable - 1) * ubs, tokens.begin() + usable * ubs);
    std::vector<uint8_t> tail_blob; // validated here, then freed; the caller re-reads at replay time
    if (!read_chunk_file(rs_file_tail, tail_blob, tail_block)) {
        SRV_WRN("kv-chain[storage]: load_prefix: rs read failed/mismatch at tail chunk %zu, discarding entire restore\n", usable - 1);
        // if still on disk it is corrupt/stale: delete it so it is not re-read
        // (and re-deleted) on every restore.
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
        chunk.tokens.assign(tokens.begin() + k * ubs, tokens.begin() + (k + 1) * ubs);
        if (k + 1 == usable) {
            chunk.recr_file = rs_file_tail; // TAIL chunk only (validated above)
        }
        chunks.push_back(std::move(chunk));
    }
    *n_tokens = usable * ubs;

    // touch the replayed .kvcache files + the tail rs + every 8th rs file below
    // the tail: a future prompt may fork at one of those intermediate
    // boundaries, and only its rs file is needed to restore the recurrent
    // state there.
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
