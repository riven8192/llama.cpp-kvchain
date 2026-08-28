#include "kv-chain-store.h"

#include "server-common.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

static constexpr uint32_t KV_CHAIN_MAGIC   = 0x4b564331; // "KVC1"
static constexpr uint32_t KV_CHAIN_VERSION = 1;

kv_chain_store::kv_chain_store(std::string root_dir, uint64_t limit_bytes, int32_t batch_size) :
    root_dir(std::move(root_dir)), limit_bytes(limit_bytes), batch_size(batch_size > 0 ? batch_size : 512) {
    if (!enabled()) {
        return;
    }
    std::error_code ec;
    fs::create_directories(this->root_dir, ec);
    if (ec) {
        SRV_ERR("kv-chain: failed to create cache dir '%s': %s (ec=%d)\n",
                this->root_dir.c_str(), ec.message().c_str(), ec.value());
        this->root_dir.clear();
        return;
    }
    // index existing chunks
    for (const auto & chunk_entry : fs::recursive_directory_iterator(this->root_dir, fs::directory_options::skip_permission_denied)) {
        if (chunk_entry.is_regular_file()) {
            const auto size = static_cast<uint64_t>(chunk_entry.file_size());
            total_bytes_cur += size;
        }
    }
    index_loaded = true;
    SRV_INF("kv-chain: cache dir '%s', %zu bytes on disk, %.3f GiB (limit %.3f GiB), chunk bs=%d\n",
            this->root_dir.c_str(), (size_t) total_bytes_cur,
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

uint32_t kv_chain_store::hash_tokens(const llama_tokens & tokens, uint32_t prev) {
    // FNV-1a over the token ids, chained with the parent hash
    uint64_t h = 1469598103934665603ULL ^ prev;
    for (auto t : tokens) {
        const uint8_t * p = reinterpret_cast<const uint8_t *>(&t);
        for (int i = 0; i < (int) sizeof(t); ++i) {
            h ^= p[i];
            h *= 1099511628211ULL;
        }
    }
    return static_cast<uint32_t>(h);
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

// save a single chunk covering the prefix tokens[0..n). the state is dumped
// truncated to pos < n (via the prefix API), so the chunk holds a
// self-consistent (rows, recurrent) prefix that can be restored and resumed
// from position n.
bool kv_chain_store::save(llama_context * ctx, llama_seq_id seq_id, const llama_tokens & tokens) {
    if (!enabled() || ctx == nullptr || tokens.empty()) {
        return false;
    }

    const size_t n = tokens.size();
    const size_t state_size = llama_state_seq_get_size_ext(ctx, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
    if (state_size == 0) {
        return false;
    }

    std::vector<uint8_t> state(state_size);
    const size_t got = llama_state_seq_get_data_prefix_ext(ctx, state.data(), state_size, seq_id,
            LLAMA_STATE_SEQ_FLAGS_NONE, (llama_pos) n);
    if (got == 0 || got != state_size) {
        SRV_WRN("kv-chain: failed to get state data (%zu of %zu bytes)\n", got, state_size);
        return false;
    }

    std::error_code ec;
    const fs::path dir = fs::path(root_dir);
    fs::create_directories(dir, ec);
    if (ec) {
        SRV_ERR("kv-chain: failed to create dir '%s': %s\n", dir.string().c_str(), ec.message().c_str());
        return false;
    }

    // hash over the cumulative prefix tokens (position-aware, self-describing)
    const uint32_t chain_hash = hash_tokens(tokens, 0);
    return write_chunk(dir, chain_hash, tokens, state, 0, n);
}

// writes the chunk file for [lo, hi) if not present; returns true if written
bool kv_chain_store::write_chunk(const fs::path & dir, uint32_t chain_hash, const llama_tokens & chunk_tokens,
                                 const std::vector<uint8_t> & state, size_t lo, size_t hi) {
    const fs::path file = dir / (hash_str(chain_hash) + ".kvchunk");
    if (fs::exists(file)) {
        return false;
    }
    const uint64_t entry_bytes = state.size() + 32 + sizeof(llama_token) * chunk_tokens.size() + 8;
    if (limit_bytes > 0 && total_bytes_cur + entry_bytes > limit_bytes) {
        evict_oldest(entry_bytes);
    }
    const fs::path tmp = dir / (hash_str(chain_hash) + ".kvchunk.tmp");
    if (!write_chunk_file(tmp, file, chain_hash, chunk_tokens, state)) {
        return false;
    }
    total_bytes_cur += entry_bytes;
    SRV_INF("kv-chain: saved chunk [%zu, %zu) hash=%s (%.3f MiB)\n",
            lo, hi, hash_str(chain_hash).c_str(), (double) state.size() / (1024.0*1024.0));
    return true;
}

bool kv_chain_store::write_chunk_file(const fs::path & tmp, const fs::path & file, uint32_t chain_hash,
                                      const llama_tokens & tokens, const std::vector<uint8_t> & state) {
    std::error_code ec;
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) {
            SRV_ERR("kv-chain: failed to open '%s' for writing\n", tmp.string().c_str());
            return false;
        }
        const uint32_t magic   = KV_CHAIN_MAGIC;
        const uint32_t version = KV_CHAIN_VERSION;
        const uint32_t n_tok   = static_cast<uint32_t>(tokens.size());
        f.write(reinterpret_cast<const char *>(&magic),      sizeof(magic));
        f.write(reinterpret_cast<const char *>(&version),    sizeof(version));
        f.write(reinterpret_cast<const char *>(&chain_hash), sizeof(chain_hash));
        f.write(reinterpret_cast<const char *>(&n_tok),      sizeof(n_tok));
        f.write(reinterpret_cast<const char *>(tokens.data()), sizeof(llama_token) * tokens.size());
        f.write(reinterpret_cast<const char *>(state.data()), state.size());
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

std::vector<uint8_t> kv_chain_store::read_chunk_blob(const fs::path & file, size_t n_tokens) {
    std::ifstream f(file, std::ios::binary);
    if (!f) {
        return {};
    }
    f.seekg(0, std::ios::end);
    const uint64_t file_size = static_cast<uint64_t>(f.tellg());
    const uint64_t min_size = sizeof(uint32_t) * 4 + sizeof(uint64_t);
    if (file_size < min_size) {
        return {};
    }
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(file_size);
    f.read(reinterpret_cast<char *>(buf.data()), file_size);
    f.close();
    if (!f) {
        return {};
    }
    const uint64_t stored_sum = *reinterpret_cast<const uint64_t *>(buf.data() + file_size - sizeof(uint64_t));
    if (fnv1a64(buf.data(), file_size - sizeof(uint64_t)) != stored_sum) {
        SRV_WRN("kv-chain: checksum mismatch for %s, ignoring\n", file.string().c_str());
        return {};
    }
    const size_t hdr_len = sizeof(uint32_t) * 4 + sizeof(llama_token) * n_tokens;
    if (file_size < hdr_len + sizeof(uint64_t)) {
        return {};
    }
    return std::vector<uint8_t>(buf.begin() + hdr_len, buf.end() - sizeof(uint64_t));
}

void kv_chain_store::evict_oldest(uint64_t need_bytes) {
    std::error_code ec;
    // remove oldest-by-mtime chunks until we have room for need_bytes
    struct ent { fs::path p; uint64_t size; std::filesystem::file_time_type mtime; };
    std::vector<ent> all;
    for (const auto & e : fs::directory_iterator(this->root_dir, ec)) {
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

std::vector<uint8_t> kv_chain_store::load_prefix(const llama_tokens & tokens, size_t * n_tokens) const {
    std::vector<uint8_t> empty;
    *n_tokens = 0;
    if (!enabled() || tokens.empty()) {
        return empty;
    }

    const fs::path dir = fs::path(root_dir);
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        return empty;
    }

    // enumerate the same chunk boundaries that save() used (full ubatches plus
    // the final, possibly short chunk). each chunk's hash is over the cumulative
    // tokens [0, hi), so a chunk exists iff that exact prefix was saved. the
    // best restore is the largest hi whose chunk file is intact; the leftover
    // tokens [hi, n) are re-prefilled by the caller.
    const size_t bs = (size_t) batch_size;
    size_t best_n = 0;
    std::vector<uint8_t> best_blob;

    size_t lo = 0;
    while (lo < tokens.size()) {
        const size_t hi = std::min(lo + bs, tokens.size());

        llama_tokens cum_tokens(tokens.begin(), tokens.begin() + hi);
        const uint32_t chain_hash = hash_tokens(cum_tokens, 0);

        const fs::path file = dir / (hash_str(chain_hash) + ".kvchunk");
        const bool exists = fs::exists(file);
        std::vector<uint8_t> blob = read_chunk_blob(file, hi);
        SRV_INF("kv-chain: load_prefix: boundary hi=%zu hash=%s exists=%d ok=%d\n",
                hi, hash_str(chain_hash).c_str(), (int) exists, (int) !blob.empty());
        if (!blob.empty()) {
            best_blob = std::move(blob);
            best_n = hi;
        }
        lo = hi;
    }

    if (best_n > 0) {
        *n_tokens = best_n;
        SRV_INF("kv-chain: loaded %zu tokens from %s (%.3f MiB)\n",
                best_n, dir.string().c_str(), best_blob.size() / (1024.0*1024.0));
        return best_blob;
    }
    return empty;
}
