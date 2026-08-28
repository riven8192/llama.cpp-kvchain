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

## 4. Implementation status (as of this commit)

### v1 - WORKING (committed, tested)
- tools/server/kv-chain-store.{h,cpp}: disk store, one chunk = full-prompt state.
  - file: <dir>/<chain_hash>.kvchunk = u32 magic "KVC1", u32 version, u32
    chain_hash, u32 n_tokens, llama_token[n_tokens], state blob, u64 fnv1a64 sum.
    chain_hash = fnv1a64 over the prefix token ids. Atomic .tmp+rename.
  - save() called from post_decode() at SLOT_STATE_DONE_PROMPT (state = exactly
    the prompt cells, no decode tokens). Saving on release() is WRONG (would
    include generated tokens -> oversized state -> position mismatch).
  - load_prefix() walks candidate chunks, verifies checksum, returns longest
    intact prefix state blob.
- server-context.cpp restore (SLOT_STATE_STARTED, n_past==0, slot empty,
  cache_prompt): set_data, push prompt tokens, n_past=n_saved. Flags:
  - kv_chain_restored: skip truncating seq_rm (recurrent can't roll back).
  - kv_chain_full_restore: whole prompt restored -> go straight to GENERATING
    (avoids the empty-batch assert in the DONE_PROMPT path).
  - skip the [TAG_PROMPT_LOGITS] n_past-- when restored.
- common: --kv-chain-dir, --kv-chain-limit-gb.
- TESTED PASS (devops/llama_test.sh, Qwen3.8-27B): prime, 100% match (restart,
  zero prefill), partial (restart, +4 -> restore 15 + prefill 4), crash-restart
  (kill -9, chunk intact, restore), corruption (flipped byte -> checksum reject
  -> prefill fallback, no crash). All coherent output.

### v2 - IN PROGRESS (not yet working; build currently BROKEN)
Goal: per-ubatch chunks so an aborted long prefill (e.g. 50K of 80K) leaves the
completed chunks on disk; on reprompt, restore the longest COMPLETE chunk
boundary (if the match lands mid-chunk N, discard chunk N, resume from end of
N-1, prefill the rest). Never need to resume mid-chunk.

What is in the tree (uncommitted, partial):
1. Per-ubatch hook: llama.h llama_context_params gained cb_ubatch(cb_ubatch_data)
   + cb_ubatch_data, copied into cparams (llama-cparams.h, llama-context.cpp:142,
   common.h/common.cpp). Fired in llama_decode's ubatch loop (llama-context.cpp
   ~line 1969) after each ubatch commits. Server registers it (server-context.cpp
   load_model) and arms kv_chain_prefill_slot in SLOT_STATE_STARTED, clears it at
   the DONE_PROMPT->GENERATING transition (so nothing is saved after prefill -
   decode-generated state is worthless for caching: tokenizer round-trip is not
   stable and clients rarely re-send generated text).
2. Server snapshot (kv_chain_save_prefill_ubatch): reads pos_max from
   llama_memory_seq_pos_max (authoritative; the context-side ubatch.pos and
   memory->seq_pos_max inside the callback were stale/0). Saves only at
   n_ubatch-aligned positions or the prompt end. Slices prompt.tokens[0..pos].
3. Position-truncated state API (the core fix):
   - llama.h: llama_state_seq_get_data_prefix_ext / _set_data_prefix_ext
     (extra llama_pos pos_limit arg).
   - llama-context.{h,cpp}: state_seq_get_data_prefix/_set_data_prefix ->
     state_seq_write_data/_read_data (now take pos_limit, default INT32_MAX).
   - llama-memory.h: state_write/read virtuals gained pos_limit=INT32_MAX.
   - llama-kv-cache.cpp state_write: added `add_cell &&= cells.pos_get(i) <
     pos_limit`. state_read ignores it (blob already truncated).
   - llama-memory-recurrent.cpp state_write: added `if (cell.pos >= pos_limit)
     continue;` in the cell loop. (R/S tensors are per-position-row, so this
     yields the true prefix recurrent state.)
   - Composites (hybrid, hybrid-iswa, msa, iswa, dsa, dsv4) forward pos_limit.

WHY v2 was needed (root cause found):
- `llama_state_seq_get_data_ext` returns a FULL-CONTEXT state blob that is
  byte-identical at the 32- and 60-token boundaries (differ only in the header
  token array, 112 bytes). Reason: the memory cell table is sized to n_ctx and
  ALL batch cells get seq_id 0 assigned during init_batch/prepare (batch-wide),
  before any ubatch computes. So a mid-prefill dump includes not-yet-computed
  cells, and the restored state reports the stale pos_max (e.g. 59 for a
  32-token chunk) -> the leftover-prefill batch at pos 32 fails the
  "Y = X+1" consistency check (llama-batch.cpp:300) -> "Invalid input batch".
- The fix is to filter cells by pos < pos_limit in state_write (done above), so
  a 32-token chunk truly holds 32 rows and restores to pos_max=31.

### BLOCKER (why build is red) - FIX NEXT
- src/llama-kv-cache-dsv4.cpp:1606-1608: llama_kv_cache_dsv4::state_read forwards
  pos_limit to llama_dsv4_comp_state::state_read, which does NOT take pos_limit
  (it's a non-override helper, llama-kv-cache-dsv4.cpp:1066). Either add the
  param to llama_dsv4_comp_state::state_read (and its callers) or stop forwarding
  it in the dsv4 composite. dsv4 is not the model we test, so the minimal fix is
  to NOT forward pos_limit in llama_kv_cache_dsv4::state_read/write (drop the
  `, pos_limit` arg on the llama_dsv4_comp_state calls there).
- After that: rebuild, then re-run the multi-chunk test (devops/llama_test.sh
  --flush <60tok> [restart] <67tok> -- -ub 32 -b 32) and verify: (a) 2 chunks
  saved at distinct sizes (NOT byte-identical - proves truncation works),
  (b) 100% match restores 60, (c) partial (67) restores 32 then prefills 35
  WITHOUT "Invalid input batch".
- If the truncated recurrent state still misbehaves on resume, the recurrent
  rs_idx/rollback machinery may need attention - investigate llama-memory-recurrent
  state_read_meta / find_slot with a truncated cell set.

## 5. Dev tooling (devops/)
- env.sh (model, port 50081, paths), llama_build.sh (cmake+ninja, Vulkan,
  logs to devops/.llama-build.log, exits 1 on failure), llama_run.sh
  (--keep-cache/--no-kv-chain/-- <extra args>), llama_kill.sh, llama_wait.sh,
  llama_prompt.sh (sends /v1/completions with cache_prompt:true), llama_test.sh
  (--flush, [restart] markers, trailing `-- <extra run args>`; NOTE: the `--`
  tail must come LAST, after all prompts).
- IMPORTANT: `llama_build.sh | tail -1` masks the exit code - check EXIT=$? or
  the "build OK" line, never assume success from the last ninja line.

## 6. Key constraints / gotchas
- Gated DeltaNet recurrent state S_n is NON-INVERTIBLE: a chunk is a
  self-consistent (attn rows 0..n-1, S_n) pair; you can only resume FROM n,
  never roll back to m<n. Never "strip the last token" from a chunk.
- -ub (n_ubatch) has a lower bound of 32 (-ub 8 is silently ignored, falls back
  to the default 2048). Use -ub 32 for multi-chunk tests.
- The server processes a whole prompt in one llama_decode call (internal
  ubatch split is invisible to the server loop); that is why the per-ubatch
  hook had to go INSIDE llama_decode, not in the server post_decode.
- n_batch/n_ubatch: server.cpp forces n_batch=n_ubatch only for embeddings;
  otherwise -ub/-b take effect (verified 256/512/1024).
