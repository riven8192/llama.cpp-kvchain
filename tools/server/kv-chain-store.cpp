#include "kv-chain-store.h"

#include "server-common.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

static constexpr uint32_t KV_CHAIN_MAGIC   = 0x4b564331; // "KVC1"
static constexpr uint32_t KV_CHAIN_VERSION = 1;

kv_chain_store::kv_chain_store(std::string root_dir, uint64_t limit_bytes) :
    root_dir(std::move(root_dir)), limit_bytes(limit_bytes) {
    if (!enabled()) {
        return;
    }
    std::error_code ec;
    fs::create_directories(root_dir, ec);
    if (ec) {
        SRV_ERR("kv-chain: failed to create cache dir '%s': %s\n", root_dir.c_str(), ec.message().c_str());
        root_dir.clear();
        return;
    }
    // index existing chunks
    for (const auto & root_entry : fs::directory_iterator(root_dir)) {
        if (!root_entry.is_directory()) {
            continue;
        }
        for (const auto & chunk_entry : fs::directory_iterator(root_entry.path())) {
            if (!chunk_entry.is_regular_file()) {
                continue;
            }
            const auto size = static_cast<uint64_t>(chunk_entry.file_size());
            total_bytes_cur += size;
        }
    }
    index_loaded = true;
    SRV_INF("kv-chain: cache dir '%s', %zu bytes on disk, %.3f GiB (limit %.3f GiB)\n",
            root_dir.c_str(), (size_t) total_bytes_cur,
            (double) total_bytes_cur / (1024.0*1024.0*1024.0),
            (double) limit_bytes / (1024.0*1024.0*1024.0));
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

bool kv_chain_store::save(llama_context * ctx, llama_seq_id seq_id, const llama_tokens & tokens) {
    if (!enabled() || ctx == nullptr || tokens.empty()) {
        return false;
    }

    const size_t state_size = llama_state_seq_get_size_ext(ctx, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
    if (state_size == 0) {
        return false;
    }

    const uint32_t chain_hash = hash_tokens(tokens, 0);

    const fs::path dir  = fs::path(root_dir);
    const fs::path file = dir / (hash_str(chain_hash) + ".kvchunk");

    if (fs::exists(file)) {
        return true; // already saved
    }

    // evict oldest roots until we fit under the limit
    const uint64_t entry_bytes = state_size + 64 + sizeof(llama_token) * tokens.size();
    if (limit_bytes > 0) {
        struct root_info {
            fs::path path;
            uint64_t bytes;
            fs::file_time_type mtime;
        };
        std::vector<root_info> roots;
        for (const auto & e : fs::directory_iterator(root_dir)) {
            if (!e.is_directory()) {
                continue;
            }
            uint64_t b = 0;
            for (const auto & c : fs::directory_iterator(e.path())) {
                if (c.is_regular_file()) {
                    b += static_cast<uint64_t>(c.file_size());
                }
            }
            roots.push_back({ e.path(), b, e.last_write_time() });
        }
        std::sort(roots.begin(), roots.end(), [](const root_info & a, const root_info & b) {
            return a.mtime < b.mtime;
        });
        for (const auto & r : roots) {
            if (total_bytes_cur + entry_bytes <= limit_bytes) {
                break;
            }
            std::error_code ec;
            fs::remove_all(r.path, ec);
            if (!ec) {
                total_bytes_cur -= r.bytes;
            }
        }
    }

    std::vector<uint8_t> state(state_size);
    const size_t n = llama_state_seq_get_data_ext(ctx, state.data(), state_size, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
    if (n != state_size) {
        SRV_WRN("kv-chain: failed to get state data (%zu of %zu bytes)\n", n, state_size);
        return false;
    }

    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        SRV_ERR("kv-chain: failed to create dir '%s': %s\n", dir.string().c_str(), ec.message().c_str());
        return false;
    }

    // write to .tmp then rename for atomicity
    const fs::path tmp = file.parent_path() / (file.stem().string() + ".kvchunk.tmp");
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) {
            SRV_ERR("kv-chain: failed to open '%s' for writing\n", tmp.string().c_str());
            return false;
        }
        const uint32_t magic   = KV_CHAIN_MAGIC;
        const uint32_t version = KV_CHAIN_VERSION;
        const uint32_t n_tok   = static_cast<uint32_t>(tokens.size());
        f.write(reinterpret_cast<const char *>(&magic),   sizeof(magic));
        f.write(reinterpret_cast<const char *>(&version), sizeof(version));
        f.write(reinterpret_cast<const char *>(&chain_hash), sizeof(chain_hash));
        f.write(reinterpret_cast<const char *>(&n_tok),   sizeof(n_tok));
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
        // re-read to compute the checksum (cheap enough for v1)
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

    total_bytes_cur += entry_bytes;
    SRV_INF("kv-chain: saved %zu tokens (%.3f MiB) -> %s\n",
            tokens.size(), state_size / (1024.0*1024.0), file.string().c_str());

    return true;
}

std::vector<uint8_t> kv_chain_store::load_prefix(const llama_tokens & tokens, size_t * n_tokens) const {
    std::vector<uint8_t> empty;
    *n_tokens = 0;
    if (!enabled() || tokens.empty()) {
        return empty;
    }

    // walk the chain: for every prefix length that could have been saved,
    // check if the chunk file exists and is intact
    // v1: save() writes the full prompt only; a restore is a full-prefix hit
    // or a miss (v2 will store incremental chain chunks for partial hits)
    const fs::path dir = fs::path(root_dir);
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        return empty;
    }

    // collect candidate chunk files with their token counts
    struct candidate {
        fs::path file;
        uint32_t n_tok;
    };
    std::vector<candidate> cands;
    for (const auto & e : fs::directory_iterator(dir, ec)) {
        if (!e.is_regular_file() || e.path().extension() != ".kvchunk") {
            continue;
        }
        std::ifstream f(e.path(), std::ios::binary);
        uint32_t hdr[5];
        if (!f.read(reinterpret_cast<char *>(hdr), sizeof(hdr))) {
            continue;
        }
        if (hdr[0] != KV_CHAIN_MAGIC || hdr[1] != KV_CHAIN_VERSION) {
            continue;
        }
        cands.push_back({ e.path(), hdr[4] });
    }
    std::sort(cands.begin(), cands.end(), [](const candidate & a, const candidate & b) {
        return a.n_tok > b.n_tok;
    });

    for (const auto & c : cands) {
        if (c.n_tok > tokens.size()) {
            continue;
        }
        // verify the chain hash matches the incoming tokens
        llama_tokens prefix(tokens.begin(), tokens.begin() + c.n_tok);
        const uint32_t chain_hash = hash_tokens(prefix, 0);
        if (hash_str(chain_hash) != c.file.stem().string()) {
            continue;
        }
        // verify checksum
        std::ifstream f(c.file, std::ios::binary);
        if (!f) {
            continue;
        }
        f.seekg(0, std::ios::end);
        const uint64_t file_size = static_cast<uint64_t>(f.tellg());
        if (file_size < sizeof(uint32_t) * 4 + sizeof(uint64_t)) {
            continue;
        }
        std::vector<uint8_t> buf(file_size);
        f.read(reinterpret_cast<char *>(buf.data()), file_size);
        if (!f) {
            continue;
        }
        f.close();
        const uint64_t stored_sum = *reinterpret_cast<const uint64_t *>(buf.data() + file_size - sizeof(uint64_t));
        if (fnv1a64(buf.data(), file_size - sizeof(uint64_t)) != stored_sum) {
            SRV_WRN("kv-chain: checksum mismatch for %s, ignoring\n", c.file.string().c_str());
            continue;
        }
        // intact: return the state blob (everything after the token array)
        const size_t hdr_len = sizeof(uint32_t) * 4 + sizeof(llama_token) * c.n_tok;
        *n_tokens = c.n_tok;
        std::vector<uint8_t> out(buf.begin() + hdr_len, buf.end() - sizeof(uint64_t));
        SRV_INF("kv-chain: loaded %zu tokens from %s (%.3f MiB)\n",
                c.n_tok, c.file.string().c_str(), out.size() / (1024.0*1024.0));
        return out;
    }

    return empty;
}
