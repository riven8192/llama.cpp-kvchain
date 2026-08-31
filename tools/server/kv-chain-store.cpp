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
// bump whenever EITHER changes: an old file with a stale version is never read
// (version check), and the root hash changes, so old chunks are a clean miss.
static constexpr uint32_t KV_CHAIN_VERSION = 3;

kv_chain_store::kv_chain_store(std::string root_dir, uint64_t limit_bytes, int32_t batch_size,
                               const common_params & params, const llama_model * model) :
    root_dir(std::move(root_dir)), limit_bytes(limit_bytes), batch_size_(batch_size > 0 ? batch_size : 512) {
    if (!enabled()) {
        return;
    }

    compute_root_hash(params, model);

    std::error_code ec;
    const fs::path cache_dir = fs::path(this->root_dir);
    fs::create_directories(cache_dir, ec);
    if (ec) {
        SRV_ERR("kv-chain: failed to create cache dir '%s': %s (ec=%d)\n",
                cache_dir.string().c_str(), ec.message().c_str(), ec.value());
        this->root_dir.clear();
        this->root_hash_hex.clear();
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
    SRV_INF("kv-chain: cache dir '%s', root=%s, %zu bytes on disk, %.3f GiB (limit %.3f GiB), chunk bs=%d\n",
            cache_dir.string().c_str(), root_hash_hex.c_str(), (size_t) total_bytes_cur,
            (double) total_bytes_cur / (1024.0*1024.0*1024.0),
            (double) limit_bytes / (1024.0*1024.0*1024.0),
            batch_size);
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
    md.chunk_size        = batch_size_;
    md.model_file_size   = -1;
    md.model_file_mtime  = -1;
    md.arch              = "";
    md.ftype             = "";
    md.type_k            = params.cache_type_k;
    md.type_v            = params.cache_type_v;
    md.rope_scaling_type = params.rope_scaling_type;
    {
        const float rope_freq_base  = params.rope_freq_base;  // 0.0f = "from model"
        const float rope_freq_scale = params.rope_freq_scale; // 0.0f = "from model"
        md.rope_freq_base_bits   = *reinterpret_cast<const uint32_t *>(&rope_freq_base);
        md.rope_freq_scale_bits  = *reinterpret_cast<const uint32_t *>(&rope_freq_scale);
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
        SRV_WRN("kv-chain: no model handle, root hash will not include arch/ftype/file identity", (const char *) "");
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
    h = hash_str_field(h, model_path);

    root_hash_hex = hash_str(h);
    {
        const float rope_freq_base  = *reinterpret_cast<const float *>(&md.rope_freq_base_bits);
        const float rope_freq_scale = *reinterpret_cast<const float *>(&md.rope_freq_scale_bits);
        SRV_INF("kv-chain: metadata: version=%d chunk_size=%d model='%s' size=%lld mtime=%lld arch='%s' ftype='%s' type_k=%d type_v=%d rope=(%d,%.6g,%.6g)\n",
                md.format_version, md.chunk_size, model_path.c_str(),
                (long long) md.model_file_size, (long long) md.model_file_mtime,
                md.arch.c_str(), md.ftype.c_str(),
                (int) md.type_k, (int) md.type_v,
                md.rope_scaling_type, rope_freq_base, rope_freq_scale);
    }
}

// dumps one seq-state blob for the window [pos_lo, pos_hi) with the given
// part-selection flags (FULL_ONLY = attn rows, PARTIAL_ONLY = recurrent rows).
// returns an empty vector on failure.
static std::vector<uint8_t> dump_window(llama_context * ctx, llama_seq_id seq_id,
                                        llama_pos pos_lo, llama_pos pos_hi, llama_state_seq_flags flags) {
    const size_t size = llama_state_seq_get_size_window_ext(ctx, seq_id, flags, pos_lo, pos_hi);
    if (size == 0) {
        return {};
    }
    std::vector<uint8_t> blob(size);
    const size_t got = llama_state_seq_get_data_window_ext(ctx, blob.data(), size, seq_id, flags, pos_lo, pos_hi);
    if (got == 0 || got != size) {
        SRV_WRN("kv-chain: failed to get window state (%zu of %zu bytes)\n", got, size);
        return {};
    }
    return blob;
}

// save one chunk covering the window [pos_lo, pos_hi). dumps the attn rows and
// the recurrent rows for exactly that window, and stores them in two separate
// files (.kvcache + .rscache). chunk_tokens are the window's tokens.
bool kv_chain_store::save(llama_context * ctx, llama_seq_id seq_id, llama_pos pos_lo, llama_pos pos_hi,
                          uint64_t chunk_hash, const llama_tokens & chunk_tokens) {
    if (!enabled() || ctx == nullptr || chunk_tokens.empty() || pos_hi <= pos_lo) {
        return false;
    }

    std::vector<uint8_t> attn = dump_window(ctx, seq_id, pos_lo, pos_hi, LLAMA_STATE_SEQ_FLAGS_FULL_ONLY);
    std::vector<uint8_t> recr = dump_window(ctx, seq_id, pos_lo, pos_hi, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (attn.empty() || recr.empty()) {
        SRV_WRN("kv-chain: failed to dump chunk window [%d, %d)\n", (int) pos_lo, (int) pos_hi);
        return false;
    }

    const fs::path dir = fs::path(root_dir);

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

    // per-file header overhead: magic + version + hash32 + n_tokens + tokens + blob_size + checksum
    const uint64_t hdr_bytes = 4 * sizeof(uint32_t) + sizeof(llama_token) * chunk_tokens.size()
                              + sizeof(uint32_t) + sizeof(uint64_t);
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
        SRV_INF("kv-chain: saved chunk hash=%s (attn %.3f MiB, recr %.3f MiB)\n",
                stem.c_str(),
                (double) attn.size() / (1024.0*1024.0), (double) recr.size() / (1024.0*1024.0));
    }
    return any_written;
}

// writes one chunk file (header + single blob + checksum) to tmp, then renames.
bool kv_chain_store::write_chunk_file(const fs::path & tmp, const fs::path & file, uint64_t chunk_hash,
                                      const llama_tokens & tokens, const std::vector<uint8_t> & blob) {
    std::error_code ec;
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) {
            SRV_ERR("kv-chain: failed to open '%s' for writing\n", tmp.string().c_str());
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
            SRV_ERR("kv-chain: write failed for '%s'\n", tmp.string().c_str());
            fs::remove(tmp, ec);
            return false;
        }
        // checksum over everything written so far
        f.seekp(0, std::ios::end);
        const uint64_t file_size = static_cast<uint64_t>(f.tellp());
        std::ifstream rf(tmp, std::ios::binary);
        std::vector<uint8_t> buf(file_size);
        rf.read(reinterpret_cast<char *>(buf.data()), file_size);
        const uint64_t checksum = fnv1a64(buf.data(), buf.size());
        rf.close();
        f.write(reinterpret_cast<const char *>(&checksum), sizeof(checksum));
        f.close();
    }

    fs::rename(tmp, file, ec);
    if (ec) {
        SRV_ERR("kv-chain: rename failed for '%s': %s\n", file.string().c_str(), ec.message().c_str());
        fs::remove(tmp, ec);
        return false;
    }
    return true;
}

// reads a single-blob chunk file (.kvcache or .rscache); returns false if missing/corrupt
bool kv_chain_store::read_chunk_file(const fs::path & file, std::vector<uint8_t> & out_blob, llama_tokens & out_tokens) {
    std::ifstream f(file, std::ios::binary);
    if (!f) {
        return false;
    }
    f.seekg(0, std::ios::end);
    const uint64_t file_size = static_cast<uint64_t>(f.tellg());
    const uint64_t min_size = sizeof(uint32_t) * 4 + sizeof(uint32_t) + sizeof(uint64_t);
    if (file_size < min_size) {
        return false;
    }
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(file_size);
    f.read(reinterpret_cast<char *>(buf.data()), file_size);
    f.close();
    if (!f) {
        return false;
    }
    const uint64_t stored_sum = *reinterpret_cast<const uint64_t *>(buf.data() + file_size - sizeof(uint64_t));
    if (fnv1a64(buf.data(), file_size - sizeof(uint64_t)) != stored_sum) {
        SRV_WRN("kv-chain: checksum mismatch for %s, ignoring\n", file.string().c_str());
        return false;
    }
    // header: magic, version, chunk_hash, n_tokens
    const uint32_t magic = *reinterpret_cast<const uint32_t *>(buf.data());
    const uint32_t version = *reinterpret_cast<const uint32_t *>(buf.data() + sizeof(uint32_t));
    if (magic != KV_CHAIN_MAGIC || version != KV_CHAIN_VERSION) {
        SRV_WRN("kv-chain: bad magic/version in %s (magic=%08x version=%u), ignoring\n",
                file.string().c_str(), magic, version);
        return false;
    }
    const uint32_t n_tokens = *reinterpret_cast<const uint32_t *>(buf.data() + sizeof(uint32_t) * 3);
    // bound the token count before the size arithmetic
    if (n_tokens == 0 || (uint64_t) n_tokens > (file_size - (sizeof(uint32_t) * 4 + sizeof(uint32_t) + sizeof(uint64_t))) / sizeof(llama_token)) {
        return false;
    }
    const size_t hdr_len = sizeof(uint32_t) * 4 + sizeof(llama_token) * n_tokens;
    const uint32_t blob_size = *reinterpret_cast<const uint32_t *>(buf.data() + hdr_len);
    const uint8_t * blob_ptr = buf.data() + hdr_len + sizeof(uint32_t);
    const size_t expected = hdr_len + sizeof(uint32_t) + blob_size + sizeof(uint64_t);
    if (expected != file_size) {
        SRV_WRN("kv-chain: size mismatch in %s (expected %zu, got %llu), ignoring\n",
                file.string().c_str(), expected, (unsigned long long) file_size);
        return false;
    }
    out_tokens.assign(reinterpret_cast<const llama_token *>(buf.data() + sizeof(uint32_t) * 4),
                      reinterpret_cast<const llama_token *>(buf.data() + sizeof(uint32_t) * 4) + n_tokens);
    out_blob.assign(blob_ptr, blob_ptr + blob_size);
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
            SRV_INF("kv-chain: evicted %s (%.3f MiB)\n", e.p.filename().string().c_str(), e.size / (1024.0*1024.0));
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

    // single walk of the hash chain. chunk k covers tokens[k*bs,(k+1)*bs).
    // the .kvcache file is REQUIRED for every chunk (attn rows are additive);
    // the .rscache file is OPTIONAL (recr is a tail object, last one wins).
    // usable = last chunk index (1-based) that has BOTH kv + rs files.
    // chunks beyond `usable` are truncated (their attn rows are useless without
    // the recurrent tail). a missing rs file for chunks < usable is simply
    // skipped (empty recr_blob, no set_data) - later chunks' real rs overwrite it.
    const size_t bs = (size_t) batch_size_;
    const size_t n_chunks = tokens.size() / bs;

    uint64_t prev_hash = 0;
    size_t usable = 0; // 1-based: last chunk with an rs file
    size_t n_kv_found = 0;
    uint64_t total_loaded = 0;
    std::vector<fs::path> kv_files_touched;
    fs::path rs_file_touched;

    for (size_t k = 0; k < n_chunks; ++k) {
        const llama_tokens block(tokens.begin() + k * bs, tokens.begin() + (k + 1) * bs);
        const uint64_t chunk_hash = hash_chunk(block, prev_hash);
        const std::string stem = hash_str(chunk_hash);

        const fs::path kv_file = dir / (stem + ".kvcache");
        if (!fs::exists(kv_file)) {
            break;
        }
        std::vector<uint8_t> attn_blob;
        llama_tokens file_tokens;
        if (!read_chunk_file(kv_file, attn_blob, file_tokens)) {
            break;
        }
        if (file_tokens != block) {
            SRV_WRN("kv-chain: load_prefix: chain stops at chunk %zu (token mismatch)\n", k);
            break;
        }
        n_kv_found++;

        const fs::path rs_file = dir / (stem + ".rscache");
        std::vector<uint8_t> recr_blob;
        bool rs_ok = false;
        if (fs::exists(rs_file)) {
            llama_tokens rs_tokens;
            if (read_chunk_file(rs_file, recr_blob, rs_tokens) && rs_tokens == block) {
                rs_ok = true;
            }
        }
        if (rs_ok) {
            usable = k + 1;
            rs_file_touched = rs_file;
        } else {
            // empty recr_blob = "skip set_data for this chunk". safe because the
            // recurrent state is a tail object (last write wins): skipping a
            // middle chunk's recr has no effect on the final state (the next
            // chunk's real rs overwrites it). the only chunk that matters is the
            // last one, and if ITS rs file is missing, `usable` drops to the
            // previous rs file's chunk.
        }

        kv_chain_chunk chunk;
        chunk.attn_blob = std::move(attn_blob);
        chunk.recr_blob = std::move(recr_blob);
        chunk.tokens = std::move(file_tokens);
        total_loaded += chunk.attn_blob.size() + chunk.recr_blob.size();
        chunks.push_back(std::move(chunk));
        kv_files_touched.push_back(kv_file);
        prev_hash = chunk_hash;
    }

    // truncate to usable: chunks beyond the last rs file have no recurrent tail
    // and their attn rows cannot be used without it.
    if (usable < chunks.size()) {
        chunks.resize(usable);
    }
    *n_tokens = usable * bs;

    if (usable > 0) {
        // touch exactly the files that were read and replayed:
        // chunks[0..usable-1]'s .kvcache + the ONE .rscache of chunk (usable-1).
        std::vector<fs::path> to_touch(kv_files_touched.begin(), kv_files_touched.begin() + usable);
        to_touch.push_back(rs_file_touched);
        touch_chain_files(to_touch);
    }

    if (usable > 0) {
        SRV_INF("kv-chain: %zu prompt chunks, first %zu kv-files found on disk, last rs-file found for chunk %zu\n",
                n_chunks, n_kv_found, usable);
    } else {
        SRV_INF("kv-chain: %zu prompt chunks, no usable chain (no kv+rs pair on disk)\n", n_chunks);
    }
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
