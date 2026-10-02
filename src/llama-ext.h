#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"

#include <cstdint>
#include <cstdlib>
#include <map>

// kv-chain: return true when the verbose diagnostics are enabled (KVCHAIN_VERBOSE=1).
// OFF by default, so the normal prefill/restore path pays no diagnostic I/O and no
// per-layer KVCHAINDBG logging. the /tmp/rt-* and idx-K tensor dumps are the
// expensive part (a GPU sync + tensor_get per call); gating them and the log noise
// behind this one env var keeps production clean while leaving the tooling available
// when a new architecture needs to be debugged. read once (cached) because the
// state-write/read paths call it in per-layer loops.
inline bool llama_kvchain_diag_verbose() {
    static const bool on = [] {
        const char * e = std::getenv("KVCHAIN_VERBOSE");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// kv-chain diagnostics: log the attn vs QSA-indexer cache head/used/size for a
// seq, so a cell-for-cell drift after a restore is visible (qwen4exp). no-op if
// the model has no hybrid_idx memory.
LLAMA_API void llama_kvchain_dbg_log(const struct llama_context * ctx, int32_t seq_id);

// kv-chain: restore the QSA-indexer (COMP_ONLY) blob for seq_id, placing its cells
// in the SAME indices the attention cache currently holds (instead of running its
// own find_slot, which drifts under APPEND). the blob is a COMP_ONLY seq-state
// window blob, same shape as the one fed to llama_state_seq_set_data_window_ext.
// append: when true, do not clear the seq's idx cells first (chunks 1..N). returns
// the number of bytes read, 0 on failure (or if the model has no indexer).
LLAMA_API size_t llama_kvchain_set_idx_window(const struct llama_context * ctx,
        const uint8_t * src, size_t size, int32_t seq_id,
        llama_state_seq_flags flags, int32_t pos_lo, int32_t pos_limit, bool append);

// kv-chain: invalidate the cached compute graph after an out-of-band state
// restore that bypasses the state_seq_set_data path (the QSA idx restore). the
// next decode must rebuild the graph, or it sizes its windows (qwen4exp QSA)
// from a stale n_kv.
LLAMA_API void llama_kvchain_invalidate_graph(const struct llama_context * ctx);

// kv-chain: synchronize the backend so device tensors are readable on host.
// needed by the idx-key fingerprint diagnostic (log_kvchain_dbg).
LLAMA_API void llama_kvchain_sync(const struct llama_context * ctx);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

LLAMA_API const char * llama_model_arch_name(const struct llama_model * model);

//
// model/context data extraction
//

LLAMA_API int32_t llama_model_dflash_selector_top_k(const struct llama_model * model);

// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);
