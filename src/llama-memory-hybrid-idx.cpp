#include "llama-memory-hybrid-idx.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"
#include "llama-ext.h" // llama_kvchain_sync (the idx-key fingerprint diagnostic)


#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <stdexcept>

//
// llama_memory_hybrid_idx
//

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
    const layer_filter_cb & filter_idx) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, offload, unified,
        filter_attn, filter_recr),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size;

        // the cached indexer keys are raw, rotation happens after pooling at read time, so a
        // K-shift must not rotate them while the stream copies in the same update still apply
        hparams_idx.rope_type = LLAMA_ROPE_TYPE_NONE;

        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);

        return new llama_kv_cache(
            model, hparams_idx, type_k, type_v, v_trans, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_");
    }()) {}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // note: repeats llama_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer uses the attention cache's slot layout; a separate one can drift from it
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

void llama_memory_hybrid_idx::clear(bool data) {
    llama_memory_hybrid::clear(data);

    if (mem_idx) {
        mem_idx->clear(data);
    }
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, p0, p1);
    }

    return get_mem_attn()->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);

    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    llama_memory_hybrid::seq_keep(seq_id);

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
    }
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);

    if (mem_idx) {
        mem_idx->seq_add(seq_id, p0, p1, shift);
    }
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);

    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
    }
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    return mb;
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags, llama_pos pos_lo, llama_pos pos_limit) const {
    // the QSA indexer (mem_idx) is a per-token llama_kv_cache (row i = token i, the
    // "blocks" are a runtime pooling, not a storage ratio). it is serialized:
    //   - for COMP_ONLY: as the WHOLE blob (the .cmcache file, per-chunk additive,
    //     identity window - mem_idx is a plain kv cache so state_write's pos filter
    //     does the cut);
    //   - for FULL (flags 0): as a pure suffix after attn+recr (the upstream layout,
    //     so a whole-context round-trip stays complete);
    //   - NOT for ATTN_ONLY/FULL_ONLY (the .kvcache holds only the main attn KV) nor
    //     TAIL_ONLY/PARTIAL_ONLY (the recurrent tail).
    // this mirrors llama_kv_cache_dsv4, whose COMP_ONLY flag selects the comp K
    // caches and excludes them from ATTN_ONLY. (an earlier version emitted the idx
    // under the attn gate, which put the whole growing indexer into every .kvcache
    // and desynced the read side.)
    const bool comp_only = (flags & LLAMA_STATE_SEQ_FLAGS_COMP_ONLY) != 0;
    if (comp_only) {
        if (mem_idx) {
            // mem_idx is a plain llama_kv_cache: COMP_ONLY there means "write an
            // EMPTY state" (it has no comp part of its own), so clear the flag -
            // the indexer IS the per-token cache, serialized in full for the
            // window [pos_lo, pos_limit) (state_write's pos filter does the cut).
            mem_idx->state_write(io, seq_id, (llama_state_seq_flags) (flags & ~LLAMA_STATE_SEQ_FLAGS_COMP_ONLY), pos_lo, pos_limit);
        }
        return;
    }

    llama_memory_hybrid::state_write(io, seq_id, flags, pos_lo, pos_limit);

    // the idx section is a pure suffix, present for a FULL blob (flags 0) only.
    // FULL_ONLY/ATTN_ONLY/TAIL_ONLY/PARTIAL_ONLY all exclude it.
    const bool full = (flags & (LLAMA_STATE_SEQ_FLAGS_FULL_ONLY | LLAMA_STATE_SEQ_FLAGS_ATTN_ONLY |
                                LLAMA_STATE_SEQ_FLAGS_TAIL_ONLY | LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) == 0;
    if (full && mem_idx) {
        mem_idx->state_write(io, seq_id, flags, pos_lo, pos_limit);
    }
}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags, llama_pos pos_lo, llama_pos pos_limit, const void * sinfos_in) {
    // note: repeats llama_memory_hybrid::state_read
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    // [TAG_HYBRID_IDX_SINFO]
    // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    // sinfos_in (kv-chain) is the attention cache's slot layout, captured while the
    // .kvcache (ATTN_ONLY) restore ran; the idx must land on those SAME cell indices
    // or the cell-for-cell invariant (qwen4exp.cpp: "the indexer cache must track
    // the attention cache cell for cell") breaks. it is a const slot_info_vec_t *;
    // the base declares it void * to avoid including llama-kv-cache.h.
    const llama_kv_cache::slot_info_vec_t * sinfos_attn_in =
            static_cast<const llama_kv_cache::slot_info_vec_t *>(sinfos_in);
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    if (llama_kvchain_diag_verbose()) {
        LLAMA_LOG_ERROR("KVCHAINDBG hybrid_idx::state_read flags=0x%x pos=[%d,%d) mem_idx=%d sinfos_in=%d\n",
                (unsigned) flags, (int) pos_lo, (int) pos_limit, (int) (mem_idx != nullptr), (int) (sinfos_in != nullptr));
    }

    // COMP_ONLY: the .cmcache blob is ONLY the indexer rows (written by the
    // comp_only branch of state_write). mem_idx is a plain llama_kv_cache, so clear
    // COMP_ONLY (it would otherwise read an EMPTY state). it must restore into the
    // attention cache's layout (sinfos_attn_in) - a plain state_read would find_slot
    // in the idx's OWN cells, and under APPEND its head counter drifts from the
    // attn's, landing cells at different indices (the cell-for-cell assert).
    // APPEND = do not seq_rm the dest seq first, so the replay appends this chunk's
    // rows on top of the earlier chunks' (chunk 0 does the wipe).
    const bool comp_only = (flags & LLAMA_STATE_SEQ_FLAGS_COMP_ONLY) != 0;
    if (comp_only) {
        if (mem_idx) {
            const bool append = (flags & LLAMA_STATE_SEQ_FLAGS_APPEND) != 0;
            if (llama_kvchain_diag_verbose()) {
                LLAMA_LOG_ERROR("KVCHAINDBG hybrid_idx::state_read COMP_ONLY -> mem_idx::state_read_sinfo append=%d sinfos_in=%d\n",
                        (int) append, (int) (sinfos_attn_in != nullptr));
            }
            mem_idx->state_read_sinfo(io, seq_id,
                    (llama_state_seq_flags) (flags & ~LLAMA_STATE_SEQ_FLAGS_COMP_ONLY),
                    pos_lo, pos_limit, nullptr, sinfos_attn_in, append);
        }
        return;
    }

    // section gates must mirror state_write exactly:
    //   attn part : written by llama_memory_hybrid::state_write under the
    //               FULL_ONLY/ATTN_ONLY gate, read under the !PARTIAL_ONLY gate
    //               (the base hybrid's own convention - the two agree for every
    //               flag combination this arch sees).
    //   recr part : written under the PARTIAL_ONLY/TAIL_ONLY gate. this function
    //               "repeats" llama_memory_hybrid::state_read (which gates the
    //               recr read the same way), but the repeat here called it
    //               unconditionally - a latent bug that only bites now that
    //               kv-chain restores a hybrid_idx arch (qwen4exp) with an
    //               ATTN_ONLY blob: with no recr section in the blob, the recr
    //               reader mis-parses the following bytes and runs past the end
    //               of buffer. gate it on the same flag the write side uses.
    //   idx part  : a pure suffix, written for FULL (flags 0) only (COMP_ONLY is
    //               handled above). it adopts the attention sinfos, so it is read
    //               only when the attn part was read too.
    const bool has_recr_section = (flags & (LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_TAIL_ONLY)) != 0;
    const bool full             = (flags & (LLAMA_STATE_SEQ_FLAGS_FULL_ONLY | LLAMA_STATE_SEQ_FLAGS_ATTN_ONLY |
                                            LLAMA_STATE_SEQ_FLAGS_TAIL_ONLY | LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) == 0;
    const bool has_idx_section  = full && (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0;
    if (llama_kvchain_diag_verbose()) {
        LLAMA_LOG_ERROR("KVCHAINDBG hybrid_idx::state_read has_recr_section=%d full=%d has_idx_section=%d\n",
                (int) has_recr_section, (int) full, (int) has_idx_section);
    }

    // the attn gate must mirror llama_memory_hybrid::state_write, which emits the
    // attn part under the FULL_ONLY/ATTN_ONLY gate. the original upstream repeat
    // here used a bare !PARTIAL_ONLY gate, which is WRONG for TAIL_ONLY (0x20):
    // PARTIAL_ONLY (0x1) is clear, so !PARTIAL_ONLY is true and the attn section
    // would be read out of a TAIL_ONLY blob that holds only the recr tail -
    // mis-parsing it ("invalid seq_id-agnostic kv cell"). this is the second
    // latent bug in this "repeat" that the kv-chain flags exposed.
    const bool has_attn_section = (flags & (LLAMA_STATE_SEQ_FLAGS_FULL_ONLY | LLAMA_STATE_SEQ_FLAGS_ATTN_ONLY)) != 0;

    // the attn (per-token KV) part is what the kv-chain restores chunk-by-chunk:
    // chunk 0 wipes (APPEND clear), chunks 1..N APPEND. the append mode must come
    // from the flags, NOT be hardcoded false - a hardcoded false made every chunk
    // after the first seq_rm the whole seq, so only the LAST chunk's cells survived
    // (the restore looked empty: a windowed read of [0,ubs) saw cell_count=0).
    const bool attn_append = (flags & LLAMA_STATE_SEQ_FLAGS_APPEND) != 0;

    try {
        if (has_attn_section) {
            get_mem_attn()->state_read_sinfo(io, seq_id, flags, pos_lo, pos_limit, has_idx_section ? &sinfos_attn : nullptr, nullptr, attn_append);
        }

        if (has_recr_section) {
            get_mem_recr()->state_read(io, seq_id, flags, pos_lo, pos_limit);
        }

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if (has_idx_section) {
            if (mem_idx) {
                mem_idx->state_read_sinfo(io, seq_id, flags, pos_lo, pos_limit, nullptr, &sinfos_attn, false);
            }
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->seq_rm(seq_id, -1, -1);
    get_mem_recr()->seq_rm(seq_id, -1, -1);

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, -1, -1);
    }
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}

void llama_memory_hybrid_idx::log_kvchain_dbg(llama_seq_id seq_id) const {
    if (!llama_kvchain_diag_verbose()) {
        return;
    }
    const auto * attn = get_mem_attn();
    const auto * idx  = mem_idx.get();
    if (!attn) {
        return;
    }
    if (idx) {
        LLAMA_LOG_ERROR("KVCHAINDBG hybrid_idx seq=%d attn(head=%u used=%u size=%u) idx(head=%u used=%u size=%u)\n",
                (int) seq_id,
                attn->get_head(seq_id), attn->get_used(seq_id), attn->get_size(),
                idx->get_head (seq_id), idx->get_used (seq_id), idx->get_size ());
    } else {
        LLAMA_LOG_ERROR("KVCHAINDBG hybrid_idx seq=%d attn(head=%u used=%u size=%u) idx=<none>\n",
                (int) seq_id, attn->get_head(seq_id), attn->get_used(seq_id), attn->get_size());
    }
}

void llama_memory_hybrid_idx::log_kvchain_dbg(const struct llama_context * ctx, llama_seq_id seq_id) const {
    if (!llama_kvchain_diag_verbose()) {
        return;
    }
    const auto * attn = get_mem_attn();
    const auto * idx  = mem_idx.get();
    if (!attn) {
        return;
    }
    if (idx) {
        LLAMA_LOG_ERROR("KVCHAINDBG hybrid_idx seq=%d attn(head=%u used=%u size=%u) idx(head=%u used=%u size=%u)\n",
                (int) seq_id,
                attn->get_head(seq_id), attn->get_used(seq_id), attn->get_size(),
                idx->get_head (seq_id), idx->get_used (seq_id), idx->get_size ());
    } else {
        LLAMA_LOG_ERROR("KVCHAINDBG hybrid_idx seq=%d attn(head=%u used=%u size=%u) idx=<none>\n",
                (int) seq_id, attn->get_head(seq_id), attn->get_used(seq_id), attn->get_size());
    }

    if (ctx == nullptr || idx == nullptr) {
        return;
    }

    // content fingerprint of the idx K tensor, so a RESTORED run can be compared
    // byte-for-byte against the NO-RESTORE run (same prompt -> same values):
    //   - cell 0 = the first restored cell (the restore's data)
    //   - the LAST non-empty cell = the most recent prefill/decode cell (fresh data)
    const auto & icells = idx->get_cells(seq_id);
    const int32_t il = idx->get_layer_ids().front();
    auto * kt = idx->get_k_storage(il);
    if (!kt) {
        return;
    }

    // the K buffer is a device tensor (unified memory, offloaded): sync, then fetch
    // the rows through the backend - never read the buffer raw.
    llama_kvchain_sync(ctx);

    const int64_t row_bytes = ggml_row_size(kt->type, kt->ne[0]);
    const size_t n_el = (size_t) (kt->ne[0] < 64 ? kt->ne[0] : 64);

    auto fp = [&](uint32_t cell) -> std::string {
        std::vector<uint8_t> buf((size_t) row_bytes);
        ggml_backend_tensor_get(kt, buf.data(), (size_t) cell*row_bytes, buf.size());
        uint64_t h = 1469598103934665603ULL;
        for (size_t i = 0; i < n_el; ++i) {
            const uint32_t bits = *(const uint32_t *) (buf.data() + i*4);
            h ^= bits; h *= 1099511628211ULL;
        }
        char out[64];
        const float f0 = *(const float *) buf.data();
        std::snprintf(out, sizeof(out), " %016llx (f0=%f)", (unsigned long long) h, f0);
        return out;
    };

    uint32_t last = 0;
    for (uint32_t j = 1; j < icells.size(); ++j) {
        if (!icells.is_empty(j)) {
            last = j;
        }
    }
    LLAMA_LOG_ERROR("KVCHAINDBG idx-k-fp seq=%d il=%d cell0[%s] last@%u pos=%d[%s]\n",
            (int) seq_id, il, fp(0).c_str(), last, (int) icells.pos_get(last), fp(last).c_str());
}

void llama_memory_hybrid_idx::dump_idx_k(const struct llama_context * ctx, llama_seq_id seq_id, const char * path, uint32_t n_cells) const {
    const auto * idx = mem_idx.get();
    if (!idx || ctx == nullptr || n_cells == 0) {
        return;
    }
    const int32_t il = idx->get_layer_ids().front();
    auto * kt = idx->get_k_storage(il);
    if (!kt) {
        return;
    }
    LLAMA_LOG_ERROR("KVCHAINDBG dump_idx_k: begin %u cells il=%d (ne0=%ld row_bytes=%ld)\n",
            n_cells, il, (long) kt->ne[0], (long) ggml_row_size(kt->type, kt->ne[0]));
    llama_kvchain_sync(ctx);
    LLAMA_LOG_ERROR("KVCHAINDBG dump_idx_k: synced\n");
    const int64_t row_bytes = ggml_row_size(kt->type, kt->ne[0]);
    std::vector<uint8_t> buf((size_t) n_cells*row_bytes);
    ggml_backend_tensor_get(kt, buf.data(), 0, buf.size());
    LLAMA_LOG_ERROR("KVCHAINDBG dump_idx_k: tensor_get done (%zu bytes)\n", buf.size());
    FILE * f = std::fopen(path, "wb");
    if (f) {
        std::fwrite(buf.data(), 1, buf.size(), f);
        std::fclose(f);
        LLAMA_LOG_ERROR("KVCHAINDBG dump_idx_k: wrote %u cells (%zu bytes) il=%d to %s\n", n_cells, buf.size(), il, path);
    } else {
        LLAMA_LOG_ERROR("KVCHAINDBG dump_idx_k: failed to open %s\n", path);
    }
}

llama_kv_cache::slot_info_vec_t llama_memory_hybrid_idx::kvchain_attn_sinfos(llama_seq_id seq_id, llama_pos pos_lo, llama_pos pos_limit) const {
    const auto * attn = get_mem_attn();
    if (attn == nullptr || seq_id < 0) {
        return {};
    }
    // the attn's current cells for seq_id, in cell-index order: this is the layout
    // the idx must adopt so the two caches stay cell-for-cell after a restore.
    // kv-chain runs with a single stream, so the stream index is 0.
    // a per-chunk idx restore restores exactly one chunk's window, so restrict the
    // layout to [pos_lo, pos_limit) when given: feeding the whole restored prefix
    // here makes the mirrored-layout size check reject the chunk-sized restore.
    const bool windowed = pos_limit > pos_lo;
    const auto & cells = attn->get_cells(seq_id);
    const uint32_t strm = 0;
    llama_kv_cache::slot_info sinfo;
    sinfo.s0 = strm;
    sinfo.s1 = strm;
    sinfo.strm = { strm };
    sinfo.idxs = { {} };
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.is_empty(i) && cells.seq_has(i, seq_id)) {
            if (windowed) {
                const llama_pos p = cells.pos_get(i);
                if (p < pos_lo || p >= pos_limit) {
                    continue;
                }
            }
            sinfo.idxs[0].push_back(i);
        }
    }
    llama_kv_cache::slot_info_vec_t res;
    res.push_back(std::move(sinfo));
    return res;
}

void llama_memory_hybrid_idx::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias) const {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(get_mem_idx() != nullptr);

    GGML_ASSERT(ggml_backend_buffer_is_host(cell_blk->buffer));

    const int64_t n_kv     = cell_blk->ne[0];
    const int64_t n_ns     = cell_blk->ne[1];        // streams in this ubatch
    const int64_t n_blocks = blk_pos->ne[0]/(4*n_ns);
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t r        = ratio;

    GGML_ASSERT(n_tokens % n_ns == 0);
    const int64_t n_tps = n_tokens/n_ns;             // tokens per stream

    int32_t * dst_cell_blk  = (int32_t *) cell_blk->data;
    int32_t * dst_blk_cells = (int32_t *) blk_cells->data;
    int32_t * dst_blk_pos   = (int32_t *) blk_pos->data;
    float   * dst_bias      = (float   *) bias->data;

    // a block is keyed on (sequence set, index bucket): a unified cache counts every sequence
    // from zero, so the bucket alone would pool two sequences into one block
    GGML_ASSERT(r <= 64);
    const uint64_t slots_full = r == 64 ? ~uint64_t(0) : ((uint64_t(1) << r) - 1);

    // TODO: this runs per ubatch and is O(n_kv) per stream, about 865 us at 33k context. the cost
    //       is the per-cell scan rather than these allocations, so hoisting them buys nothing
    std::vector<int32_t>  blk_of(n_kv);
    std::vector<int32_t>  cell_grp(n_kv);
    std::vector<int32_t>  grp_head(n_blocks);
    std::vector<int32_t>  grp_next;
    std::vector<int32_t>  grp_first;
    std::vector<int32_t>  grp_slot0;
    std::vector<uint64_t> grp_slots;
    std::vector<int32_t>  grp_bid;
    std::vector<int32_t>  bid_idx;
    std::vector<int32_t>  bid_cell;
    std::vector<int32_t>  bid_slot0;

    std::vector<int32_t> order;
    std::vector<int32_t> rank;

    std::fill(dst_blk_pos, dst_blk_pos + 4*n_blocks*n_ns, 0);

    for (int64_t s = 0; s < n_ns; ++s) {
        // ubatch index s*n_tps belongs to this stream; ask which cells array it uses
        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
        const auto & cells = get_mem_idx()->get_cells(seq_of_stream);

        int32_t * cur_cell_blk  = dst_cell_blk  + s*n_kv;
        int32_t * cur_blk_cells = dst_blk_cells + s*(r*n_blocks);

        std::fill(cur_blk_cells, cur_blk_cells + r*n_blocks, 0);

        bid_idx  .clear();
        bid_cell .clear();
        bid_slot0.clear();

        int n_seq_present = 0;

        for (int sq = 0; sq < LLAMA_MAX_SEQ && n_seq_present < 2; ++sq) {
            if (cells.seq_pos_min(sq) >= 0) {
                n_seq_present++;
            }
        }

        const bool one_seq = n_seq_present <= 1;

        // a cell no block covers needs its own -inf, which a per-block bias cannot carry
        // every cache path keeps the position below the cell window, so this stays false
        bool oor = false;

        bool dup = false;

        bool ranked = false;

        auto group_cells = [&]() {
            // -1 means no usable block: an incomplete or short group cannot be pooled
            std::fill(blk_of.begin(),   blk_of.end(),   -1);
            std::fill(cell_grp.begin(), cell_grp.end(), -1);
            std::fill(grp_head.begin(), grp_head.end(), -1);

            grp_next .clear();
            grp_first.clear();
            grp_slot0.clear();
            grp_slots.clear();
            grp_bid  .clear();

            oor = false;
            dup = false;

            for (int64_t j = 0; j < n_kv; ++j) {
                if (cells.is_empty(j)) {
                    continue;
                }

                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);
                const int64_t pb  = idx/r;

                if (pb >= n_blocks) {
                    oor = true;
                    continue;
                }

                int32_t g = -1;

                for (int32_t c = grp_head[pb]; c >= 0; c = grp_next[c]) {
                    if (one_seq || cells.seq_get_all((uint32_t) grp_first[c]) == cells.seq_get_all((uint32_t) j)) {
                        g = c;
                        break;
                    }
                }

                if (g < 0) {
                    g = (int32_t) grp_first.size();

                    grp_next .push_back(grp_head[pb]);
                    grp_first.push_back((int32_t) j);
                    grp_slot0.push_back(-1);
                    grp_slots.push_back(0);
                    grp_bid  .push_back(-1);

                    grp_head[pb] = g;
                }

                const uint64_t bit = uint64_t(1) << (idx%r);

                dup |= (grp_slots[g] & bit) != 0;

                cell_grp[j]   = g;
                grp_slots[g] |= bit;

                if (idx%r == 0) {
                    grp_slot0[g] = (int32_t) j;
                }
            }
        };

        group_cells();

        // mrope repeats one position across an image, so rank cells instead of using the position
        if (dup && ubatch->is_pos_2d() && one_seq) {
            order.clear();
            order.reserve(n_kv);

            for (int64_t j = 0; j < n_kv; ++j) {
                if (!cells.is_empty(j)) {
                    order.push_back((int32_t) j);
                }
            }

            // same total order the mrope causal mask uses: pos, then ext.y, then ext.x
            std::sort(order.begin(), order.end(), [&cells](int32_t a, int32_t b) {
                const llama_pos pa = cells.pos_get(a);
                const llama_pos pb = cells.pos_get(b);

                if (pa != pb) {
                    return pa < pb;
                }

                const auto & ea = cells.ext_get(a);

                return cells.ext_get(b).is_2d_gt(ea.x, ea.y);
            });

            rank.assign(n_kv, -1);

            for (int64_t k = 0; k < (int64_t) order.size(); ++k) {
                rank[order[k]] = (int32_t) k;
            }

            ranked = true;

            group_cells();
        }

        if (blk_bias && oor) {
            // diagnose which cell runs past the window: dump the max position, the
            // cache's own occupancy counters (to tell a stale n_kv from cells not
            // sitting at the low indices), and the offending cells.
            const auto * idxkv = get_mem_idx();
            int64_t pmax = -1;
            int64_t jmin = -1;
            int64_t n_used_seq = 0;
            for (int64_t j = 0; j < n_kv; ++j) {
                if (cells.is_empty(j) || !cells.seq_has((uint32_t) j, seq_of_stream)) {
                    continue;
                }
                if (jmin < 0) {
                    jmin = j;
                }
                n_used_seq++;
                pmax = std::max(pmax, (int64_t) cells.pos_get(j));
            }
            std::string oor_cells;
            for (int64_t j = 0; j < n_kv && oor_cells.size() < 200; ++j) {
                if (cells.is_empty(j)) {
                    continue;
                }
                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);
                if (idx/r >= n_blocks) {
                    char buf[80];
                    std::snprintf(buf, sizeof(buf), " j=%lld pos=%lld rank=%d",
                            (long long) j, (long long) cells.pos_get(j), ranked ? (int) rank[j] : -1);
                    oor_cells += buf;
                }
            }
            LLAMA_LOG_ERROR("KVCHAINDBG qsa OOR: s=%lld n_kv=%lld n_blocks=%lld r=%lld pmax=%lld jmin=%lld n_used_seq=%lld used_max_p1=%u get_used=%u cells_size=%u seq=%d n_tokens=%lld%s\n",
                    (long long) s, (long long) n_kv, (long long) n_blocks, (long long) r,
                    (long long) pmax, (long long) jmin, (long long) n_used_seq,
                    idxkv ? idxkv->get_cells(seq_of_stream).used_max_p1() : 0,
                    idxkv ? idxkv->get_used(seq_of_stream) : 0,
                    idxkv ? (uint32_t) idxkv->get_cells(seq_of_stream).size() : 0,
                    (int) seq_of_stream, (long long) n_tokens, oor_cells.c_str());
        }
        GGML_ASSERT((!blk_bias || !oor) && "qsa: cell position runs past the cell window");

        int32_t n_bid = 0;

        for (int64_t pb = 0; pb < n_blocks; ++pb) {
            for (int32_t g = grp_head[pb]; g >= 0; g = grp_next[g]) {
                if (grp_slots[g] != slots_full) {
                    continue;
                }

                grp_bid[g] = n_bid++;

                bid_idx  .push_back((int32_t) (pb*r));
                bid_cell .push_back(grp_first[g]);
                bid_slot0.push_back(grp_slot0[g]);
            }
        }

        GGML_ASSERT(n_bid <= n_blocks);

        for (int32_t b = 0; b < n_bid; ++b) {
            int32_t sec_pos[4] = { bid_idx[b], bid_idx[b], bid_idx[b], bid_idx[b] };

            if (ranked) {
                const int32_t   c = bid_slot0[b];
                const llama_pos p = cells.pos_get(c);
                const auto &    e = cells.ext_get(c);

                sec_pos[0] = p;
                sec_pos[1] = e.y;
                sec_pos[2] = e.x;
                sec_pos[3] = p;
            }

            for (int64_t sec = 0; sec < 4; ++sec) {
                dst_blk_pos[sec*(n_blocks*n_ns) + s*n_blocks + b] = sec_pos[sec];
            }
        }

        // unpooled cells all point at one spare block. a spare block exists only when some
        // cell is unpooled: n_bid == n_blocks means every cell sits in a full block.
        const bool     have_dead = n_bid < n_blocks;
        const int32_t  dead_bid  = have_dead ? n_bid : n_blocks - 1;

        for (int64_t j = 0; j < n_kv; ++j) {
            const int32_t g = cell_grp[j];

            blk_of[j] = g < 0 ? -1 : grp_bid[g];

            if (blk_of[j] >= 0) {
                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                cur_blk_cells[blk_of[j]*r + (idx%r)] = (int32_t) j;
            }

            cur_cell_blk[j] = blk_of[j] < 0 ? dead_bid : blk_of[j];
        }

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t      i      = s*n_tps + ii;
            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            int64_t q = ubatch->pos[i];

            if (ranked) {
                const llama_pos qt = ubatch->pos[i];
                const llama_pos qy = ubatch->pos[i + n_tokens];
                const llama_pos qx = ubatch->pos[i + n_tokens*2];

                int64_t lo = 0;
                int64_t hi = (int64_t) order.size();

                while (lo < hi) {
                    const int64_t   mid = (lo + hi)/2;
                    const int32_t   c   = order[mid];
                    const llama_pos pc  = cells.pos_get(c);

                    if (pc < qt || (pc == qt && !cells.ext_get(c).is_2d_gt(qx, qy))) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }

                q = lo - 1;
            }

            // the tail is an incomplete block and is always visible, as in the reference
            const int64_t tail_start = (q + 1)/r*r;

            if (blk_bias) {
                // a block sits wholly inside or outside the tail, so one value covers it
                // the caller adds the attention mask, which drops empty, foreign and future cells
                float * cur_blk_bias = dst_bias + i*n_blocks;

                for (int64_t b = 0; b < n_blocks; ++b) {
                    if (b >= n_bid || !cells.seq_has((uint32_t) bid_cell[b], seq_id)) {
                        cur_blk_bias[b] = -INFINITY;
                        continue;
                    }

                    // finite, so it can never meet a -inf and produce a nan
                    cur_blk_bias[b] = bid_idx[b] >= tail_start ? 1e9f : 0.0f;
                }

                // the spare block holds the unpooled cells, which are the incomplete tail, so
                // it gets the tail value. it must stay finite: a sequence with fewer than
                // `ratio` cells owns no full block, and a row of -inf only gives a nan.
                if (have_dead) {
                    cur_blk_bias[dead_bid] = 1e9f;
                }

                continue;
            }

            float * cur_bias = dst_bias + i*n_kv;

            for (int64_t j = 0; j < n_kv; ++j) {
                float v = -INFINITY;

                if (!cells.is_empty(j) && cells.seq_has(j, seq_id)) {
                    const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                    if (idx <= q) {
                        // finite, so it can never meet a -inf and produce a nan
                        v = idx >= tail_start ? 1e9f : (blk_of[j] < 0 ? -INFINITY : 0.0f);
                    }
                }

                cur_bias[j] = v;
            }
        }
    }
}

//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    mem(mem),
    // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
    // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
    ns_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ mem->get_mem_idx()->get_n_stream() }),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx())) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem),
    // update() applies a pending cross-stream seq_cp, else the copy keeps stale indexer keys
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        mem->get_mem_idx()->init_update(lctx, optimize)) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)) {}

bool llama_memory_hybrid_idx_context::next() {
    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    return res;
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(i_cur < ns_ubatch.size());

    return ns_ubatch[i_cur];
}

void llama_memory_hybrid_idx_context::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias) const {
    GGML_ASSERT(mem != nullptr);

    mem->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, ubatch, ratio, blk_bias);
}
