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

### v2 - WORKING (committed, tested)
Goal: per-chunk streaming persistence so an aborted long prefill (e.g. 50K of
80K) leaves the completed chunks on disk; on reprompt, restore the longest
complete chunk boundary and prefill the rest.

DESIGN (fixed bs-grid hash chain):
- chunk_size = --ubatch value (bs, default 512; use -ub 32 for testing). It is
  used PURELY as a boundary stride - we do NOT assume it matches the runtime
  prefill batch size.
- chunk k covers tokens [k*bs, (k+1)*bs). Only COMPLETE chunks are persisted;
  a trailing partial block is never written and always re-prefilled.
- Hash chain (deterministic from the token list alone):
    hash_0 = H(chunk_0_tokens, prev=0)
    hash_k = H(chunk_k_tokens, prev=hash_{k-1})
  where H is FNV-1a over the chunk's own bs token ids, seeded by the parent
  hash. A longer prompt reuses identical early chunks. The loader walks the
  chain from the root, checking each file, stopping at the first miss/corrupt.
- File: <dir>/<chunk_hash>.kvchunk = u32 magic "KVC1", u32 version,
  u32 chunk_hash, u32 n_tokens (=bs, the chunk's own tokens), llama_token[bs],
  the FULL-PREFIX state blob [0,(k+1)*bs), u64 fnv1a64 sum. Atomic .tmp+rename.
  The state blob holds the full prefix (rows + recurrent both as-of (k+1)*bs),
  so it is self-consistent. (Option A: full state per chunk; deltas are later.)

WRITE PATH:
- Per-ubatch hook: llama.h llama_context_params gained cb_ubatch(cb_ubatch_data),
  copied into cparams (llama-cparams.h, llama-context.cpp:142, common.h/cpp).
  Fired in llama_decode's ubatch loop after each ubatch commits.
- Server arms kv_chain_prefill_slot in SLOT_STATE_STARTED, clears it at the
  DONE_PROMPT->GENERATING transition (nothing saved after prefill -
  decode-generated state is worthless: tokenizer round-trip is not stable).
- kv_chain_save_prefill_ubatch: reads pos from llama_memory_seq_pos_max
  (authoritative). Dumps a chunk ONLY when pos % bs == 0 (on-grid). The
  runtime ubatch split does NOT always land on the grid (it is "chaotic" near
  the tail / for short prompts - e.g. a 46-tok prompt splits 14/42/46, a
  1260-tok prompt is a clean 32-grid in the body then a chaotic tail). We just
  skip off-grid edges - that is the "stop at the edge of the happy flow" rule.
  No dedup state needed (pos is strictly increasing across callback fires).

POSITION-TRUNCATED STATE API (the core enabler):
- llama.h: llama_state_seq_get_data_prefix_ext / _set_data_prefix_ext (extra
  llama_pos pos_limit arg).
- llama-context.{h,cpp}: state_seq_get/set_data_prefix -> state_seq_write/read_data
  (pos_limit, default INT32_MAX).
- llama-memory.h: state_write/read virtuals gained pos_limit=INT32_MAX.
- llama-kv-cache.cpp state_write: `add_cell &&= cells.pos_get(i) < pos_limit`.
- llama-memory-recurrent.cpp state_write: `if (cell.pos >= pos_limit) continue;`
  (R/S tensors are per-position-row -> true prefix recurrent state).
- Composites (hybrid, hybrid-iswa, msa, iswa, dsa, dsv4) forward pos_limit.
  dsv4 comp-state (a non-override helper) does NOT take pos_limit -> the dsv4
  composite stops forwarding it there (dsv4 is not the test model).
- tests/test-batch-alloc.cpp mock_memory gained the pos_limit param (else abstract).

WHY the truncation was needed: `llama_state_seq_get_data_ext` returns a
FULL-CONTEXT blob byte-identical at every boundary (all batch cells get seq_id 0
during init_batch/prepare before any ubatch computes), so a mid-prefill dump
restored with a stale pos_max -> "Invalid input batch". Filtering by
pos < pos_limit in state_write fixes it: a 32-tok chunk truly holds 32 rows.

TEST RESULTS (devops/llama_test.sh, Qwen3.8-27B, -ub 32 -b 32, limit 100 GiB):
- prime (1260 tok): 38 chunks saved at on-grid boundaries (32..1216).
- 100% match (restart, same prompt): restored 1216/1260 (chain stops at the
  chaotic tail edge, 44 tokens re-prefilled), coherent, no errors.
- partial/divergence (shares ~15 chunks then diverges): restored 480, chain
  stops at divergence, 57 tokens re-prefilled, coherent.
- crash-restart (kill -9 after prime): 38 chunks intact on disk, 1216 restored.
- corruption (flip a payload byte): checksum mismatch -> chain stops at that
  chunk -> prefill fallback, no crash.
All coherent output, no "Invalid input batch" / "inconsistent sequence positions".

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
- chunk_size (--ubatch) is a BOUNDARY STRIDE, not the actual prefill batch size.
  The hybrid split_equal (llama-memory-hybrid.cpp:89) divides n_ubatch across
  n_seqs and is "chaotic" near the tail / for short prompts (a 46-tok prompt
  splits 14/42/46; a 1260-tok prompt is a clean 32-grid in the body then a
  chaotic tail). We only dump when pos % bs == 0, so off-grid edges are skipped
  and the tail / short prompts simply produce fewer (or zero) chunks. This is
  an implementation detail of the model/arch - do not try to make chunk edges
  match the runtime split.
- KV_CHAIN_LIMIT_GB default is 100 (devops/env.sh). Each chunk holds the FULL
  prefix state, so chunk k is ~k times the size of chunk 1 (Option A blow-up).
  At -ub 32 a chunk is ~150-230 MiB, so a 1000-tok prompt (~30 chunks) uses
  several GiB. Raise the limit for long-prompt tests.
