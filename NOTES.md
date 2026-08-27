# Hash-chain KV cache - findings

Base: llama.cpp b10520, branch `hash-chain-kv`.
Goal: disk-backed, content-addressed (hash-chained) KV cache for llama-server.
Model: Qwen3.8-27B, hybrid memory (16 full-attn layers + 48 Gated DeltaNet recurrent layers).

## 1. State save/load API (src/llama-context.cpp)

- `llama_state_seq_get_size_ext(ctx, seq_id, flags)` / `llama_state_seq_get_data_ext` /
  `llama_state_seq_set_data_ext` - the seq-level API we build on.
- Wire format of a seq state blob:
  - u32 magic `0xaf143cd8` (io_magic, not the file magic)
  - u32 seq_id (source seq)
  - then per-memory-module data (see below)
- `state_seq_write_data` / `state_seq_read_data` just dispatch to `memory->state_write/read`
  (llama-context.cpp:3220-3238). `seq_id` is ignored at that level; each memory module
  filters cells by seq_id itself.
- File wrappers: `LLAMA_STATE_SEQ_MAGIC` + version + n_tokens + tokens + state blob
  (`state_seq_save_file` / `state_seq_load_file`, llama-context.cpp:3100-3172).
- Flags: `LLAMA_STATE_SEQ_FLAGS_NONE`, `PARTIAL_ONLY` (SWA/recurrent only, skips full
  attn KV), `ON_DEVICE` (keeps data in device buffers; fast, but invalidates prior
  on-device grabs for that seq).

## 2. Hybrid memory state layout (src/llama-memory-hybrid.cpp:190-202)

- `state_write` = mem_attn->state_write (skipped if PARTIAL_ONLY) + mem_recr->state_write.
- Full-attn side (llama-kv-cache.cpp:1967-2035): n_stream, then per stream:
  cell_count, then per-cell meta (pos + seq_ids) and raw K/V row data.
  Cells are filtered by `cells.seq_has(i, seq_id)` and SWA masking.
  So a seq save contains only rows tagged with that seq (plus shared rows if seq_id == -1).
- Recurrent side (llama-memory-recurrent.cpp:733-845): cell_count, per-cell pos meta,
  then R/S tensor rows (r_l/s_l per layer). Has rollback support (`rs_idx`, n_rs_seq):
  cell.src points at a rollback row. `state_read` resets rs_idx to 0 after load.

## 3. Server: RAM prompt cache (the in-memory analogue of what we want on disk)

- `--cache-ram N` (common/arg.cpp:1706, default 8192 MiB, 0 disables, -1 unlimited).
  `params_base.cache_ram_mib` (common/common.h:615).
- `server_prompt_cache` (tools/server/server-task.h:598-631, server-task.cpp:1691-1903):
  - `std::list<server_prompt_cache_state> states`; each state = tokens + checkpoints +
    data.main (tgt state blob) + data.drft (draft state blob).
  - `alloc(prompt, size_tgt, size_dft)` (server-task.cpp:1713):
    - skip if an existing entry already fully contains the prompt (LCP check)
    - skip if state > limit_size (per-entry cap)
    - erase entries fully contained in the new prompt
    - pop_front (oldest) until under limit_size
  - `load(prompt, tokens_new, ctx_tgt, ctx_dft, id_slot)` (server-task.cpp:1795):
    - pick entry maximizing (f_keep = lcp/entry_size, f_sim = lcp/new_size)
      with f_keep >= 0.25 ("don't trash large prompts")
    - `llama_state_seq_set_data_ext(ctx_tgt, data.main, id_slot, 0)` (full state, no flags)
    - moves the entry's prompt (incl. checkpoints) into the slot, erases entry
  - `update()` (server-task.cpp:1872): evict oldest while over size/token limits.
- Slot integration (tools/server/server-context.cpp):
  - `prompt_save(prompt_cache)` (server-context.cpp:255):
    get_size_ext + get_data_ext for tgt (and dft) with flags NONE, then cache.alloc.
  - `prompt_load(prompt_cache, tokens)` (server-context.cpp:281): cache.load into slot.
  - On task dispatch, idle slot selection (server-context.cpp:1570-1609):
    pick LRU idle slot, `prompt_save` its current prompt into RAM cache, then
    `prompt_load` the best matching cached prompt into it; `prompt_cache->update()`.
  - `slot.release()` (server-context.cpp:500): SLOT_STATE_IDLE,
    `prompt_clear()` for child tasks only (`mem.seq_rm(id, -1, -1)` + prompt.clear(),
    server-context.cpp:290). Non-child slots keep their KV until reused/cleared.
    `callback_on_reset` runs before `reset()` - a natural hook point.
- Slot prefix reuse within one slot (server-context.cpp:3092-3328):
  - `n_past = slot.prompt.tokens.get_common_prefix(input_tokens)`
    (only when `cache_prompt`).
  - `n_cache_reuse` chunk shifting via seq_rm/seq_add (needs can_shift).
  - Context checkpoints (`slot.prompt.checkpoints`, `common_prompt_checkpoint`
    common/common.h:1136): saved with `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`
    (recurrent+SWA only - the part that cannot be rolled back), restored on mismatch.
  - If no usable checkpoint and pos_min below threshold: full re-prefill
    (n_past = 0) - the "likely due to SWA or hybrid/recurrent memory" path
    (server-context.cpp:3308-3313). This is the pain point: for hybrid models the
    recurrent state can only be restored from a checkpoint; with `--parallel 1` and
    a new slot there is none, hence full prefill.

## 4. Design for the disk hash-chain cache (open questions marked ?)

File layout per project plan:
  <cache_dir>/<root_hash>/<chain_hash>.kvchunk

Chunk = state blob of a token-prefix (root_hash = hash of first chunk's tokens,
chain_hash = H(parent_chain_hash + this chunk's tokens)).

- What to store per chunk:
  - `llama_state_seq_get_data_ext(ctx, seq, LLAMA_STATE_SEQ_FLAGS_NONE)` blob
    (full state: attn KV rows + recurrent R/S rows, tagged with that seq).
  - plus the token ids of the prefix (for re-tokenization checks / hashing).
  ? decide: store per-chunk (prefix grows) or incremental (delta rows only).
    Full-state-per-chunk is simplest and matches the checkpoint machinery;
    incremental would need the cell-range knowledge from state_write_meta.
- Save hook: on slot completion (callback_on_reset in slot.release(),
  server-context.cpp:319) or on dispatch-time prompt_save (server-context.cpp:1599).
  Save the longest prefix we keep (the whole prompt, or up to a chunk boundary).
- Restore hook: in task dispatch, before/alongside `prompt_load`
  (server-context.cpp:1601): look up root_hash + chain in the disk index,
  read the longest chain prefix that matches the incoming tokens,
  `llama_state_seq_set_data_ext` into the slot, set n_past accordingly.
  ? the dispatch-time path sets n_past via get_common_prefix on slot.prompt.tokens,
    which is replaced by prompt_load's moved prompt; for disk restore we must
    keep the token list consistent with the restored state (pos alignment).
- Index: <cache_dir>/index.json or a per-root dir listing (chain_hash -> file size,
  n_tokens). Atomic writes: .tmp + rename (per project plan).
- Limits: --kv-chain-limit-gb (total), evict oldest root dirs / longest-unused chains.
- ? whether to also serve the RAM cache from the same store (dedup) - v1: no.

## 5. Test plan (from docs/project-plan.md)

1. Build (done: build-vulkan/bin/llama-server, build 10520).
2. Smoke: serve Qwen3.8-27B UD-Q8_K_XL, /v1/completions with cache_prompt,
   check n_prompt_tokens_cache in metrics for repeated prefixes.
3. Disk: enable --kv-chain-dir, kill -9 server mid-session, restart, send same
   prefix, verify cached tokens > 0 and output matches pre-kill run.
4. Corruption: flip bytes in a .kvchunk, verify magic/size check rejects it
   and falls back to prefill (no crash).
