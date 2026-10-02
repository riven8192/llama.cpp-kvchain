# Hash-chain KV cache - current state

Base: llama.cpp **v0.4.0** (tag `v0.4.0`, 5266f24da), branch `hash-chain-kv`
(fork: github.com/riven8192/llama.cpp-kvchain). Local hack, not upstream-grade
(see `../docs/project-plan.md` for the design). The code is the source of truth
for HOW things work; this file only holds what/where + test status + quirks.
Model under test: selected via `LLAMA_HF_REF` in `devops/env.sh`, passed to
llama-server as `-hf`. three archs supported: Qwen3.8-27B (16 full-attn + 48
Gated DeltaNet recurrent layers), DSV4F (arch `deepseek4`, section 4b) and
Qwen3.8-Flash-Next (arch `qwen4exp` QSA indexer, section 4c).

CURRENT STATE: KV_CHAIN_VERSION 6. Per chunk, files in a FLAT cache dir:
  - `<hash>.kvcache` = ATTN_ONLY blob (per-token KV rows for exactly the chunk
    window, additive across chunks)
  - `<hash>.rscache` = TAIL_ONLY blob (the fixed-size recurrent tail; on Qwen
    the Gated DeltaNet R/S; on dsv4 the compressor rings only). last write
    wins; only the LAST chunk's copy is ever read
  - `<hash>.cmcache` (dsv4 only) = COMP_ONLY blob (this chunk's comp K rows,
    additive; see 4b)
A restore NEVER covers the whole prompt: `load_prefix` searches `tokens[0, n-1)`,
so the last token is always re-prefilled (a forward pass must produce logits).
The restore STREAMS: `load_prefix` only validates the tail .rscache and returns
file paths; the replay reads each file one at a time (read -> set_data -> free),
so peak RAM = one file. The summary line logs total bytes read + wall time of
the whole restore (read + upload, streaming):
  `kv-chain[storage]: restored N tokens ... (read X.XX GiB in T.TT s = R.RR GiB/s)`

## 1. Where things live

- Chunk file format + hash chain + store: `tools/server/kv-chain-store.{h,cpp}`
  (layout, metadata blob, two-phase restore, eviction, LRU touch all commented
  there).
- Public API additions: `include/llama.h` — windowed state-seq variants
  (`llama_state_seq_get_size_window_ext` / `_get_data_window_ext` /
  `_set_data_window_ext`), flags `FULL_ONLY` / `ATTN_ONLY` (per-token KV only;
  diverge only on dsv4, where FULL_ONLY drags in the rings) / `TAIL_ONLY`
  (everything except per-token KV) / `COMP_ONLY` (dsv4 comp K caches; empty
  blob on other archs) / `APPEND` (restore without seq_rm), per-ubatch hook
  `llama_context_params.cb_ubatch`.
- State plumbing: `src/llama-context.cpp` (hook fired in `decode`'s ubatch loop),
  `src/llama-memory.h` (virtuals take `pos_lo`/`pos_limit`),
  `src/llama-kv-cache.cpp` (window filter + APPEND),
  `src/llama-memory-recurrent.cpp` (window filter),
  `src/llama-memory-hybrid{,-iswa}.cpp` (part-selection).
- Server integration: `tools/server/server-context.cpp`
  - save path: `kv_chain_save_prefill_ubatch`, armed per slot, dumps only when
    pos is on the ubs-grid; the MTP draft re-fire at the same pos is deduped
    (and its ON/OFF-GRID log suppressed) via `slot.kv_chain_last_saved_pos`
  - restore path: SLOT_STATE_STARTED block, replays matched chunks
  - `launch_slot_with_task`: drops the slot's STALE previous-task prompt buffer
    when kv-chain is set (see the comment there + the quirks below; this is the
    fix for the `raw_left` assert on slot reuse)
  - `pre_decode`: an exhaustive slot classifier builds
    `kv_chain_slots_to_incorporate` (the slotsToIncorporate) — the single
    source of truth for which slots may add tokens / draft / decode this
    iteration. When a full-chunk prefill slot AND an interleave slot
    (partial prefill / decode) both exist, a 50/50 coin decides: the chunk slot
    gets batch exclusivity (the interleave slots are gated, left completely
    untouched, retry next iteration) or the interleave slots mix freely.
    Without the coin the chunk saves would starve every concurrent generation
    behind the longest prefill. Every per-slot loop early-outs with
    `!kv_chain_slots_to_incorporate.contains(slot)`.
  - native in-memory prefix caching is bypassed when kv_chain is set
  - context checkpoints are disabled when kv_chain is set (keeps the ubatch
    grid on-stride; chunk files make them redundant anyway)
  - grid-safety startup guard: `-b`/`-ub` must have a power-of-2 ratio (the KV
    full-retry halves n_batch, so the ratio must survive arbitrary halvings) or
    the server exits(1); dsv4 additionally requires `-ub % 128 == 0`

## 2. Design in one paragraph

Chunk k = tokens `[k*ubs,(k+1)*ubs)`, `ubs == --ubatch`. Each chunk file stores
ONLY its own window — constant file size, no duplication, NO trailing checksum
(verifying one costs a full pass over every multi-hundred-MiB recr file and
dominated restore time; we trust the storage device). attn is additive across
chunks (restore: chunk 0 wipes, later chunks APPEND); the tail is one
fixed-size object (restore: the LAST chunk's rs wins — it cannot roll back
mid-chunk, hence partial-chunk reuse is descoped). Restore is two-phase: phase
1 walks the chain with `fs::exists()` only (cheap) to find the break (first
missing .kvcache / .cmcache) and `usable` = the last chunk with an rs file;
phase 2 validates the TAIL .rscache (a bad tail discards the whole restore +
deletes the corrupt file) and returns the matched chunks as file paths — the
caller replays them one at a time (attn, then dsv4 comp, then tail rs last).
A mid-replay read failure discards the WHOLE restore (the loaded rows would be
orphaned — no valid recurrent state to resume from) -> 100% re-prefill. One
version number, `KV_CHAIN_VERSION`, covers both the file layout and the root-
hash metadata blob (bump on any change to either). Hash chain: FNV-1a64,
`hash_k = H(hash_{k-1} + chunk_k_tokens)`; root = FNV-1a64 over a metadata blob
(stat-only model identity, chunk size, dtypes, rope, n_seq_max, version) so a
different model/config/parallel is a clean miss, never garbage.

## 3. Config

- `--kv-chain-dir <path>` (empty = feature off, zero behavior change),
  `--kv-chain-limit-gb N` (default 16; devops/env.sh passes 100).
- Eviction: LRU by mtime after each write; a file vanishing mid-restore is a
  benign "cache ends here". Stray `.tmp` files removed at startup.
- LRU touch on restore: `load_prefix` touches every matched .kvcache/.cmcache,
  the TAIL .rscache, PLUS the .rscache of every chunk index that is a multiple
  of `KV_CHAIN_RS_TOUCH_STRIDE` (8) below the tail — so intermediate fork-point
   rs files survive LRU (a future prompt forking off the chain halfway still
   finds its recurrent state). tested by the mtime gap-pattern check in
   `check_fork` (llama_unittests.sh).
- `KVCHAIN_VERBOSE=1` (env, OFF by default): enables the kv-chain diagnostics
   — the `/tmp/rt-save` + `/tmp/rt-restore` round-trip dumps, the qwen4exp
   idx-K tensor dumps, and the per-layer `KVCHAINDBG` trace logs. OFF means
   the normal prefill/restore path pays no diagnostic I/O (the dumps do a GPU
   sync + tensor_get each) and no per-layer logging. turn it on when bringing
   up a new architecture (the dumps let you byte-compare a no-restore prefill
   against a restore). the qsa assert / OOR error-path logs are NOT gated.

## 4. Test status

All PASS on all three supported archs: Qwen3.8-27B, Qwen3.8-Flash-Next
(qwen4exp), and DSV4F (deepseek4). Run them with `devops/
llama_unittests_models.sh` — it runs the same `devops/llama_unittests.sh`
suite per model (27B `-ub 32`, DSV4F `-ub 128`, Flash-Next `-ub 32`), teeing
each run to `.unittest-output-<model>.log` (gitignored; grep those instead of
re-running). The suite is one server session; the active `check_*` blocks are
commented in at the bottom of `llama_unittests.sh`.

Each check's header explains what it does. the oracle: restore must be
byte-identical to the no-restore prime, and cached_tokens must hit the
expected multiple-of-ubs count. `check_verbatim` is the strongest (prime 0 ->
restore, verbatim passage reply, model-agnostic expected cached count derived
from the prime's actual prompt_tokens). the `hello`/`verbatim`/`coherence`
checks exercise chat-model prompts via the chat endpoint (see the
llama_prompt.sh quirk in section 5).

**Qwen3-4B (pure full-attn, arch `qwen3`)**: the mechanics work, but the 4B is
NON-DETERMINISTIC on a LONG generation when ANY prefix is restored (the
no-kv-chain native repeat diverges just as much — a 4B long-gen numerics issue,
NOT a kv-chain bug; the 27B is fully deterministic). Short responses match
byte-for-byte. The pure-attn arch fix in the quirks is what keeps the 4B from
emitting outright garbage. Low priority.

## 4b. DSV4F support

DSV4F (arch `deepseek4`, `llama_kv_cache_dsv4`) is the second architecture.
Its state is NOT the clean Qwen split; it owns five groups (see
src/llama-kv-cache-dsv4.h):
  - kv_raw (ISWA: base+swa)  -> per-token, windowable like Qwen attn
  - kv_csa / kv_hca / kv_lid -> COMPRESSED K caches (ratios 4 / 128 / 4),
    prefix-style rows (row i covers tokens [i*ratio, (i+1)*ratio))
  - csa_state / hca_state / lid_state -> compressor RING states (fixed-size)

Part-selection flags (include/llama.h):
  - ATTN_ONLY (16): per-token KV only. on dsv4 = kv_raw (NOT rings, NOT comp).
    used for the .kvcache file.
  - TAIL_ONLY (32): the fixed-size tail, NO per-token KV. on dsv4 = the three
    compressor RINGS only. used for the .rscache tail file.
  - COMP_ONLY (64): the comp K caches only (dsv4: csa/hca/lid; every other
    cache: an EMPTY blob). prefix-style + ADDITIVE across chunks, so
    serialized per-chunk for the window [pos_lo/ratio, pos_limit/ratio) and
    replayed with APPEND. used for the .cmcache file (dsv4 only).

Why the comp caches are a separate .cmcache (not in the .rscache tail):
  - attention over a restored prefix needs the prefix's completed comp rows
    (the remainder prefill only rebuilds rows for tokens >= n_saved) — a
    rings-only tail with the comp rows nowhere -> attention sees zeros -> a
    repetitive loop (verified earlier).
  - but they are prefix-style and ADDITIVE: chunk N's comp is a strict superset
    of chunk N-1's, so storing the full prefix in EVERY .rscache is O(N^2).
    the .cmcache stores only this chunk's delta -> O(N) total.
  - the read side verifies the on-disk mode byte matches the requested
    part-selection and THROWS on mismatch (loud, not silent garbage). the COMP
    blob's rows carry row_lo in the v3 k-cache header (APPEND lands them at the
    right offset, not 0).

On-disk layout for DSV4F (three files per chunk):
  - .kvcache = ATTN_ONLY = kv_raw window [k*ubs,(k+1)*ubs), additive (chunk 0
    wipes). [exactly what Qwen uses]
  - .cmcache = COMP_ONLY = this chunk's comp rows (rows [k*ubs/ratio,
    (k+1)*ubs/ratio)), additive (chunk 0 clears the comp caches). constant-size.
  - .rscache = TAIL_ONLY = RINGS ONLY (no kv_raw, no comp). last write wins;
    only the last chunk's copy is read. constant-size (~12 MiB).
Restore order: append all .kvcache, then all .cmcache, then the last .rscache,
then set n_past and prefill the remainder (which rebuilds comp rows for tokens
>= n_saved and overwrites the now-stale tail-of-ring). A missing .cmcache
breaks the chain (like a missing .kvcache). NON-dsv4 archs: COMP_ONLY is an
empty blob, so no .cmcache is written/read (gated on arch) — 2 files,
byte-identical to the Qwen path. Versions were bumped once at the end
(KV_CHAIN_VERSION 5->6, DSV4_STATE_VERSION 1->2, DSV4_K_CACHE_STATE_VER 2->3).

## 4c. Qwen3.8-Flash-Next (qwen4exp) support

Qwen3.8-Flash-Next (arch `qwen4exp`, `llama_memory_hybrid_idx`) is the third
architecture. unlike Qwen3.8-27B (a clean full-attn + recurrent split) it adds
a QSA (sparse-attention) INDEXER: a SEPARATE plain `llama_kv_cache` (mem_idx)
that must track the main attention cache CELL FOR CELL (an assert in
qwen4exp.cpp enforces `idx.n_kv == attn.n_kv`). the indexer's rows are the
COMP_ONLY part; on every other arch COMP_ONLY is an empty blob.

On-disk layout (same three-file scheme as dsv4, different contents):
  - .kvcache = ATTN_ONLY = the main attn window [k*ubs,(k+1)*ubs), additive.
  - .cmcache = COMP_ONLY = this chunk's QSA INDEXER rows, additive.
  - .rscache = TAIL_ONLY = the recurrent tail.
Restore of the idx (`.cmcache`) goes through `llama_kvchain_set_idx_window`
(llama-context.cpp), NOT the generic set_data path: the idx is a separate cache
that must land on the SAME cell indices the attn restore just wrote, so it
adopts the attn's slot layout (sinfos_in) instead of running its own find_slot
(which drifts under APPEND). the attn layout passed in is restricted to THIS
chunk's window [pos_lo,pos_limit) — `kvchain_attn_sinfos` filters the attn
cells by position — because the per-chunk idx restore holds exactly ubs cells,
and feeding it the whole restored prefix made the mirrored-layout size check
reject the restore ("mirrored slot layout holds 352 cells, this cache restores
32") and discard the whole restore. the idx restore also needs the compute
graph invalidated after each chunk (qwen4exp QSA sizes its windows from n_kv at
graph-build time) — `llama_kvchain_invalidate_graph`.

## 5. Quirks / gotchas

- `--parallel > 1` is supported. two concerns:
  - identity: `n_seq_max` (= `--parallel`) is baked into the ROOT HASH
    (kv-chain-store.cpp `compute_root_hash`), so a chunk file written by a
    server with a different --parallel is a CLEAN MISS. cross-parallel REUSE of
    chunk files is deliberately unsupported (the .rscache is one slot's R/S).
  - grid: with n_stream > 1, `split_equal` mixes other slots' tokens into a
    prefill slot's ubatches, which would desync the ubs-grid. the classifier's
    chunk-exclusivity (section 1) is what keeps the picked slot's boundaries
    on-grid; the interleave slots are left completely untouched for that
    iteration. Do NOT restructure the attn `state_write`/`state_read` per-
    stream loop to make the blob stream-agnostic: that loop is load-bearing and
    shared by all llama state-seq callers, and a previous attempt at it
    serialized the whole recurrent ring (~16 GB files).
- **Stale prompt buffer on slot reuse (the `raw_left` assert).** a REUSED slot
  (release + relaunch) still holds the PREVIOUS task's prompt.tokens when the
  new task starts: `reset()` keeps prompt.tokens on purpose (native in-memory
  reuse + LCP slot selection rely on it), and `get_available_slot()` ->
  `prompt_load()` moves a cached prompt back in AFTER reset (prompt =
  std::move), right before launch. the classifier computes
  raw_left = task->n_tokens() - prompt.n_tokens(), which goes negative when the
  previous task's prompt+response was longer -> assert. with kv-chain the
  native prefix reuse is bypassed and the disk chain is authoritative, so
  `launch_slot_with_task` drops the buffer. it must be done THERE, not in
  reset() (prompt_load re-populates after reset) and not in the STARTED block
  (the classifier runs first). the classifier also logs the full arithmetic +
  the first 12 token ids of buffer vs task before the assert (a stale buffer is
  visible in the log, not a bare abort).
- The recurrent window filter (llama-memory-recurrent.cpp `state_write`) must
  CLOSE the open cell range on an out-of-window cell, not just `continue`:
  with --parallel > 1 the recurrent ring interleaves cells of multiple slots,
  so a bare skip let `cell_ranges` span the gap and tripped the
  `cell_count == cell_count_check` assert on the first chunk save.
- Chunk stride grid: every ubatch boundary must be a multiple of n_ubatch,
  including KV-full retry halvings (n_batch /= 2) — enforced by the startup
  guard; `-b == -ub` is trivially safe.
- Gated DeltaNet recurrent state is NON-INVERTIBLE: resume only FROM a
  boundary.
- Pure full-attn models (e.g. Qwen3-4B) get a bare `llama_kv_cache` (no
  recurrent part). `state_write` MUST honor PARTIAL_ONLY/TAIL_ONLY/COMP_ONLY by
  serializing an EMPTY state (real `n_stream` + cell_count=0 per stream — NOT
  n_stream=0, which trips the read-side "n_stream mismatch" assert). see
  llama-kv-cache.cpp.
- No-restart restore requires `seq_rm(0, -1)` BEFORE the chunk replay: the
  recurrent module's `rs_idx`/`head`/`used` counters are NOT reset by
  `set_data_window_ext` alone.
- `has_mtmd` reflects model capability (mmproj present), NOT whether the
  prompt has media. use `get_text_tokens()` (not `get_tokens()`) to get the
  token list without the `!has_mtmd` assert.
- `-ub` has a floor of 32 (`-ub 8` is silently ignored -> 2048).
- With MTP speculative decoding, the `cb_ubatch` hook fires TWICE at every
  completed boundary (target prefill + the MTP draft path re-decoding the same
  tokens). the save hook dedupes on `slot.kv_chain_last_saved_pos` BEFORE its
  ON/OFF-GRID log, so the second fire is a silent no-op.
- Model-file mtime in the metadata blob uses std::filesystem's
  last_write_time (different epoch than unix time, logs as a negative number).
  Consistent across runs so the root hash is stable; do not "fix" it without
  bumping KV_CHAIN_VERSION (would orphan old caches).
- Each chunk ~152 MiB at -ub 32 (attn ~2 MiB + recr ~150 MiB); a 1000-tok
  prompt (~30 chunks) ~4.5 GiB.
- No integrity checksum on chunk files (removed from both sides — a full
  read+hash of every multi-hundred-MiB recr file dominated restore time). a
  silent bit-flip now surfaces as garbage model output, not a clean cache miss
  (accepted trade-off). the size/magic/version/token checks in
  `read_chunk_file` still catch layout corruption. BUMP KV_CHAIN_VERSION if you
  ever re-add a checksum (old no-checksum files would be mis-parsed).
- Only the .rscache of the TAIL chunk is ever read. failure modes:
  - a .kvcache/.cmcache MISSING in phase 1 -> clean break, prefill the rest
  - a file PRESENT in phase 1 but failing to read DURING the replay (eviction
    race / corruption) -> the loaded rows would be orphaned -> WIPES the seq,
    100% re-prefill
  - a FAILED TAIL .rscache read -> no earlier rs to fall back to -> `load_
    prefix` discards the ENTIRE restore (100% prefill) and DELETES the corrupt
    file so the re-prefill re-saves a clean one instead of re-reading +
    re-deleting it every request
- devops/llama_run.sh: server stdout goes to log FILES ONLY (an inherited
  stdout pipe makes pipe-EOF-waiting callers hang); $log is a symlink to the
  newest timestamped log.
- devops/llama_test.sh supports `[cmd:PATH]` directives (runs a bash script
  with $KV_CACHE_DIR in env) for filesystem mutations between prompts.
- devops/llama_prompt.sh uses STREAM mode against `/v1/chat/completions`
  (NOT `/v1/completions`) with a `max_tokens: 2048` cap. the chat endpoint is
  REQUIRED: the tested models are chat models, and a bare prompt through the
  raw-text completions endpoint degenerates into repetition (verified on
  dsv4f: "'foo, 'foo, 'foo..." vs a clean answer via chat). deltas are read
  from `choices[0].delta.content`; `stream_options.include_usage: true` is set
  so the cached/prompt/response-token stats still reach stderr (the tests
  parse them).
