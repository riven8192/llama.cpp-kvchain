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
static constexpr uint32_t KV_CHAIN_VERSION = 2; // v2: per-chunk window (attn + recr), no full-prefix duplication
// bump whenever the chunk file layout OR the root-hash metadata blob changes.
// an old file under a stale root dir is never read (different dir), a new file
// under a stale layout is never produced - "code changed" becomes a clean miss.
static constexpr int32_t KV_CHAIN_FORMAT_VERSION = 1;

kv_chain_store::kv_chain_store(std::string root_dir, uint64_t limit_bytes, int32_t batch_size,
                               const common_params & params, const llama_model * model) :
    root_dir(std::move(root_dir)), limit_bytes(limit_bytes), batch_size_(batch_size > 0 ? batch_size : 512) {
    if (!enabled()) {
        return;
    }

    compute_root_hash(params, model);

    std::error_code ec;
    const fs::path root = fs::path(this->root_dir) / root_hash_hex;
    fs::create_directories(root, ec);
    if (ec) {
        SRV_ERR("kv-chain: failed to create cache dir '%s': %s (ec=%d)\n",
                root.string().c_str(), ec.message().c_str(), ec.value());
        this->root_dir.clear();
        this->root_hash_hex.clear();
        return;
    }
    // remove stray .tmp files from an aborted previous run, then index existing chunks
    for (const auto & chunk_entry : fs::recursive_directory_iterator(root, fs::directory_options::skip_permission_denied)) {
        if (!chunk_entry.is_regular_file()) {
            continue;
        }
        if (chunk_entry.path().extension() == ".tmp") {
            fs::remove(chunk_entry.path(), ec);
            continue;
        }
        const auto size = static_cast<uint64_t>(chunk_entry.file_size());
        total_bytes_cur += size;
    }
    SRV_INF("kv-chain: cache dir '%s', root=%s, %zu bytes on disk, %.3f GiB (limit %.3f GiB), chunk bs=%d\n",
            root.string().c_str(), root_hash_hex.c_str(), (size_t) total_bytes_cur,
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
    md.format_version    = KV_CHAIN_FORMAT_VERSION;
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
// the recurrent rows for exactly that window (no full-prefix duplication), and
// stores both in the chunk file. chunk_tokens are the window's tokens.
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

    std::error_code ec;
    const fs::path dir = fs::path(root_dir) / root_hash_hex;
    fs::create_directories(dir, ec);
    if (ec) {
        SRV_ERR("kv-chain: failed to create dir '%s': %s\n", dir.string().c_str(), ec.message().c_str());
        return false;
    }

    return write_chunk(dir, chunk_hash, chunk_tokens, attn, recr);
}

// writes the chunk file if not present; returns true if written
// (the on-disk header stores the low 32 bits of chunk_hash)
bool kv_chain_store::write_chunk(const fs::path & dir, uint64_t chunk_hash, const llama_tokens & chunk_tokens,
                                  const std::vector<uint8_t> & attn, const std::vector<uint8_t> & recr) {
    const fs::path file = dir / (hash_str(chunk_hash) + ".kvchunk");
    if (fs::exists(file)) {
        return false;
    }
    const uint64_t entry_bytes = attn.size() + recr.size()
                               + 2 * sizeof(uint32_t)              // attn_size, recr_size
                               + 4 * sizeof(uint32_t)              // magic, version, chunk_hash, n_tokens
                               + sizeof(llama_token) * chunk_tokens.size()
                               + sizeof(uint64_t);                 // checksum
    if (limit_bytes > 0 && total_bytes_cur + entry_bytes > limit_bytes) {
        evict_oldest(entry_bytes);
    }
    const fs::path tmp = dir / (hash_str(chunk_hash) + ".kvchunk.tmp");
    if (!write_chunk_file(tmp, file, chunk_hash, chunk_tokens, attn, recr)) {
        return false;
    }
    total_bytes_cur += entry_bytes;
    SRV_INF("kv-chain: saved chunk (window %zu tokens) hash=%s (attn %.3f MiB, recr %.3f MiB)\n",
            chunk_tokens.size(), hash_str(chunk_hash).c_str(),
            (double) attn.size() / (1024.0*1024.0), (double) recr.size() / (1024.0*1024.0));
    return true;
}

bool kv_chain_store::write_chunk_file(const fs::path & tmp, const fs::path & file, uint64_t chunk_hash,
                                      const llama_tokens & tokens, const std::vector<uint8_t> & attn,
                                      const std::vector<uint8_t> & recr) {
    std::error_code ec;
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) {
            SRV_ERR("kv-chain: failed to open '%s' for writing\n", tmp.string().c_str());
            return false;
        }
        const uint32_t magic      = KV_CHAIN_MAGIC;
        const uint32_t version    = KV_CHAIN_VERSION;
        const uint32_t hash32     = static_cast<uint32_t>(chunk_hash); // header field is 32-bit
        const uint32_t n_tok      = static_cast<uint32_t>(tokens.size());
        const uint32_t attn_size  = static_cast<uint32_t>(attn.size());
        const uint32_t recr_size  = static_cast<uint32_t>(recr.size());
        f.write(reinterpret_cast<const char *>(&magic),  sizeof(magic));
        f.write(reinterpret_cast<const char *>(&version),sizeof(version));
        f.write(reinterpret_cast<const char *>(&hash32), sizeof(hash32));
        f.write(reinterpret_cast<const char *>(&n_tok),  sizeof(n_tok));
        f.write(reinterpret_cast<const char *>(tokens.data()), sizeof(llama_token) * tokens.size());
        f.write(reinterpret_cast<const char *>(&attn_size),  sizeof(attn_size));
        f.write(reinterpret_cast<const char *>(attn.data()),  attn.size());
        f.write(reinterpret_cast<const char *>(&recr_size),  sizeof(recr_size));
        f.write(reinterpret_cast<const char *>(recr.data()),  recr.size());
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

// reads both blobs from a chunk file; returns false if missing/corrupt
bool kv_chain_store::read_chunk(const fs::path & file, kv_chain_chunk & out) {
    std::ifstream f(file, std::ios::binary);
    if (!f) {
        return false;
    }
    f.seekg(0, std::ios::end);
    const uint64_t file_size = static_cast<uint64_t>(f.tellg());
    const uint64_t min_size = sizeof(uint32_t) * 4 + sizeof(uint64_t);
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
    const uint32_t n_tokens  = *reinterpret_cast<const uint32_t *>(buf.data() + sizeof(uint32_t) * 3);
    // bound the token count before the size arithmetic: a corrupt/truncated header
    // can make n_tokens huge, which would overflow the size checks below
    if (n_tokens == 0 || (uint64_t) n_tokens > (file_size - (sizeof(uint32_t) * 4 + sizeof(uint32_t) * 2 + sizeof(uint64_t))) / sizeof(llama_token)) {
        return false;
    }
    const size_t   hdr_len   = sizeof(uint32_t) * 4 + sizeof(llama_token) * n_tokens;
    const uint32_t * p      = reinterpret_cast<const uint32_t *>(buf.data() + hdr_len);
    const uint32_t   attn_size = p[0];
    const uint8_t *  attn_ptr  = buf.data() + hdr_len + sizeof(uint32_t);
    const uint32_t   recr_size = *reinterpret_cast<const uint32_t *>(attn_ptr + attn_size);
    const uint8_t *  recr_ptr  = attn_ptr + attn_size + sizeof(uint32_t);
    const size_t   expected = hdr_len + sizeof(uint32_t) + attn_size
                             + sizeof(uint32_t) + recr_size + sizeof(uint64_t);
    if (expected != file_size) {
        SRV_WRN("kv-chain: size mismatch in %s (expected %zu, got %llu), ignoring\n",
                file.string().c_str(), expected, (unsigned long long) file_size);
        return false;
    }
    out.tokens.assign(reinterpret_cast<const llama_token *>(buf.data() + sizeof(uint32_t) * 4),
                      reinterpret_cast<const llama_token *>(buf.data() + sizeof(uint32_t) * 4) + n_tokens);
    out.attn_blob.assign(attn_ptr, attn_ptr + attn_size);
    out.recr_blob.assign(recr_ptr, recr_ptr + recr_size);
    return true;
}

void kv_chain_store::evict_oldest(uint64_t need_bytes) {
    std::error_code ec;
    // remove oldest-by-mtime chunks until we have room for need_bytes.
    // the root dir holds only this model/config's chain, so a flat scan of it
    // is the whole universe (no other root dirs are evicted here).
    const fs::path root = fs::path(root_dir) / root_hash_hex;
    struct ent { fs::path p; uint64_t size; std::filesystem::file_time_type mtime; };
    std::vector<ent> all;
    for (const auto & e : fs::directory_iterator(root, ec)) {
        if (e.is_regular_file() && e.path().extension() == ".kvchunk") {
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
            SRV_INF("kv-chain: evicted %s (%.3f MiB)\n", e.p.stem().c_str(), e.size / (1024.0*1024.0));
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

    const fs::path dir = fs::path(root_dir) / root_hash_hex;
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        return chunks;
    }

    // walk the hash chain from the root. chunk k covers tokens[k*bs,(k+1)*bs)
    // and hash_k = H(hash_{k-1} + chunk_k_tokens). stop at the first missing or
    // corrupt file - that is where the cached prefix ends. the leftover tokens
    // [best_n, n) are re-prefilled by the caller. only complete bs-blocks count
    // (a trailing partial block is never cached).
    const size_t bs = (size_t) batch_size_;
    const size_t n_chunks = tokens.size() / bs;

    uint64_t prev_hash = 0;
    uint64_t total_loaded = 0;
    std::vector<fs::path> matched_files;
    for (size_t k = 0; k < n_chunks; ++k) {
        const llama_tokens block(tokens.begin() + k * bs, tokens.begin() + (k + 1) * bs);
        const uint64_t chunk_hash = hash_chunk(block, prev_hash);

        const fs::path file = dir / (hash_str(chunk_hash) + ".kvchunk");
        if (!fs::exists(file)) {
            SRV_INF("kv-chain: load_prefix: chain stops at chunk %zu (no file)\n", k);
            break;
        }
        kv_chain_chunk chunk;
        if (!read_chunk(file, chunk)) {
            SRV_INF("kv-chain: load_prefix: chain stops at chunk %zu (corrupt)\n", k);
            break;
        }
        // the header token IDs make a hash collision a clean miss instead of
        // a silent garbage read: verify them against the prompt, stop if they
        // disagree (e.g. the file was written by a different build or the hash
        // collided)
        if (chunk.tokens != block) {
            SRV_WRN("kv-chain: load_prefix: chain stops at chunk %zu (header token IDs do not match the prompt)\n", k);
            break;
        }
        total_loaded += chunk.attn_blob.size() + chunk.recr_blob.size();
        chunks.push_back(std::move(chunk));
        matched_files.push_back(file);
        prev_hash = chunk_hash;
    }

    if (!chunks.empty()) {
        // touch all matched files so LRU eviction keeps the hottest chains alive.
        // (relatime mounts do not update mtime on read, so an explicit utimensat
        // is required for the LRU to be meaningful.)
        touch_chain_files(matched_files);
        *n_tokens = chunks.size() * bs;
        SRV_INF("kv-chain: loaded %zu chunks (%zu tokens) from %s (%.3f MiB)\n",
                chunks.size(), *n_tokens, dir.string().c_str(), (double) total_loaded / (1024.0*1024.0));
    }
    return chunks;
}

void kv_chain_store::touch_chunks(const std::vector<uint64_t> & chunk_hashes) const {
    if (!enabled() || chunk_hashes.empty()) {
        return;
    }
    const fs::path dir = fs::path(root_dir) / root_hash_hex;
    std::vector<fs::path> files;
    files.reserve(chunk_hashes.size());
    for (uint64_t h : chunk_hashes) {
        files.push_back(dir / (hash_str(h) + ".kvchunk"));
    }
    touch_chain_files(files);
}
