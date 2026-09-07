# Hash-chain KV cache - current state

Base: llama.cpp b10520, branch `hash-chain-kv`. Local hack, not upstream-grade
(see `../docs/project-plan.md` for the design). The code is the source of truth
for HOW things work; this file only holds what/where + test results + quirks.
Model: Qwen3.8-27B (16 full-attn layers + 48 Gated DeltaNet recurrent layers),
selected via `LLAMA_HF_REF` (devops/env.sh, default
`unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_XL`) — passed to llama-server as `-hf`, which
resolves the exact snapshot file (and downloads it if missing). The kv-chain
root hash uses the resolved path, so cache identity follows the ref.

CURRENT STATE: v5 — chunk state is split into `<hash>.kvcache` (attn) +
`<hash>.rscache` (tail) in a FLAT cache dir. The .kvcache holds an ATTN_ONLY
blob (per-token KV rows for the chunk window only); the .rscache holds a
TAIL_ONLY blob (the tail object, last write wins, only the last chunk's copy is
read). On Qwen (16 full-attn + 48 Gated DeltaNet) TAIL_ONLY == PARTIAL_ONLY
(the Gated DeltaNet R/S). On DSV4F (llama_kv_cache_dsv4) TAIL_ONLY == the three
compressed K caches + the three compressor rings — the per-token kv_raw rows
are NOT in the .rscache (the .kvcache files already carry them, and a FULL-mode
blob read would clear kv_raw and wipe the just-restored rows). Restore works
both after a restart AND in the same session (no-restart): the pre-restore
`seq_rm(0, -1)` properly resets the recurrent module's internal state. A
restore NEVER covers the whole prompt — `load_prefix` searches the chain over
`tokens[0, n-1)` only, so the last token is always re-prefilled (logits are not
cacheable; a forward pass has to produce them). The restore STREAMS:
`load_prefix` only validates the tail
.rscache and returns file paths; the replay loop reads each .kvcache one at a
time (read -> set_data -> free), so peak RAM = one file, never the whole
chain (matters on a 128 GB box running an 110 GB model).

## 1. Where things live

- Chunk file format + hash chain + store: `tools/server/kv-chain-store.{h,cpp}`
  (layout, metadata blob, two-phase restore, eviction, LRU touch all commented
  there).
- Public API additions: `include/llama.h` — windowed state-seq variants
  (`llama_state_seq_get_size_window_ext` / `_get_data_window_ext` /
  `_set_data_window_ext`), flags `FULL_ONLY` / `ATTN_ONLY` (per-token KV only;
  the two diverge only on dsv4, where FULL_ONLY drags in the rings) /
  `TAIL_ONLY` (everything except per-token KV) / `APPEND` (restore without
  seq_rm), per-ubatch hook `llama_context_params.cb_ubatch`.
- State plumbing: `src/llama-context.cpp` (hook fired in `decode`'s ubatch loop),
  `src/llama-memory.h` (virtuals take `pos_lo`/`pos_limit`),
  `src/llama-kv-cache.cpp` (window filter + APPEND),
  `src/llama-memory-recurrent.cpp` (window filter),
  `src/llama-memory-hybrid{,-iswa}.cpp` (FULL_ONLY/PARTIAL_ONLY selection).
- Server integration: `tools/server/server-context.cpp`
  - save path: `kv_chain_save_prefill_ubatch`, armed per slot,
    dumps only when pos is on the ubs-grid; the MTP draft re-fire at the same
    pos is deduped (and its ON/OFF-GRID log suppressed) via
    `slot.kv_chain_last_saved_pos`
  - restore path: SLOT_STATE_STARTED block, replays matched chunks
  - native in-memory prefix caching is bypassed when kv_chain is set
  - context checkpoints are disabled when kv_chain is set (keeps the
    ubatch grid on-stride; chunk files make them redundant anyway)
  - grid-safety startup guard: `-b`/`-ub` must have a power-of-2 ratio or the
    server exits(1)

## 2. Design in one paragraph

Chunk k = tokens `[k*ubs,(k+1)*ubs)`, `ubs == --ubatch`. Each chunk file stores
ONLY its own window (attn rows for exactly ubs positions) — constant file size,
no duplication, and NO trailing checksum (verifying one costs a full pass over
every (multi-hundred-MiB) recr file and dominated restore time; we trust the
storage device). The .kvcache blob is ATTN_ONLY (per-token KV rows only); the
.rscache blob is TAIL_ONLY (the tail object, last write wins). attn is additive
across chunks (restore: chunk 0 wipes, later chunks APPEND); the tail is one
fixed-size object per position (restore: the LAST chunk's rs wins — it cannot
roll back mid-chunk, hence partial-chunk reuse is descoped). Restore is
two-phase: phase 1 walks the chain with `fs::exists()` only (cheap, no reads)
to find the first missing .kvcache (the break) and `usable` = the last chunk
with an rs file; phase 2 validates the TAIL .rscache (the only rs ever read;
a bad tail discards the whole restore + deletes the corrupt file) and returns
the matched chunks as file paths — the caller then replays them one at a time
(read .kvcache -> set_data -> free; tail rs read last), so the whole chain is
never in RAM at once. A missing middle .rscache is simply skipped (the chain
is truncated at `usable`); a .kvcache that fails mid-replay truncates at the
deepest loaded chunk with a valid rs file. One version number,
`KV_CHAIN_VERSION`, covers both the file layout and the root-hash metadata blob
(bump on any change to either; see kv-chain-store.cpp). Hash chain: FNV-1a64,
`hash_k = H(hash_{k-1} + chunk_k_tokens)`; root = FNV-1a64 over a metadata
blob (stat-only model identity, chunk size, dtypes, rope, n_seq_max, version)
so a different model/config/parallel is a clean miss, never garbage.

## 3. Config

- `--kv-chain-dir <path>` (empty = feature off, zero behavior change),
  `--kv-chain-limit-gb N` (default 16; devops/env.sh passes 100).
- Eviction: LRU by mtime after each write; a file vanishing mid-restore is a
  benign "cache ends here". Stray `.tmp` files removed at startup.
- LRU touch on restore: `load_prefix` touches every matched .kvcache, the
  TAIL .rscache, PLUS the .rscache of every chunk index that is a multiple of
  `KV_CHAIN_RS_TOUCH_STRIDE` (8) below the tail — so intermediate fork-point
  rs files survive LRU (a future prompt forking off the chain halfway still
  finds its recurrent state). tested by the mtime gap-pattern check in
  `llama_unittest_forked_chains.sh`.

## 4. Test status

All PASS on Qwen3.8-27B (the model the tests were written for). Run them with
`devops/llama_run_unittests.sh` (builds first, runs all, prints a summary,
exits non-zero on failure); each writes `devops/out-<name>.log` — grep THAT
instead of re-running, a full pass takes many minutes.

Each script's header explains what it does; only the expected numbers are
recorded here, because those are what a regression changes:

- `restore_restart`  : cached_tokens 0/352, 6/6 phrases
- `no_kvchain`       : native in-memory reuse, cached_tokens >= 256
- `restore_session`  : cached_tokens 0/352, 6/6 phrases, >=10 files of each type
- `evict_rscache`    : still a full restore (tail rs intact)
- `evict_kvcache`    : chain breaks at chunk 2 -> cached_tokens 64
- `forked_chains`    : exactly 0/352/96/352 + rs-touch gap pattern
- `ubatch_edge`      : 0/192, 0/256, 0/192 at -ub 64 (the middle pair is the
                        255/257 control; the 192s come from searching
                        tokens[0, n-1) - the 256-token prompt's last token is
                        never restored)
- `dsv4f`            : cached_tokens 0/352, 6/6 phrases. NOT part of the runner
                       (different model, same port/cache dir - never run it
                       concurrently with the Qwen tests)
- grid-safety guard  : `-b 96 -ub 32` -> FATAL + exit(1)

Helper: `devops/llama_make_exact_prompt_len.sh <N>` converges a prompt to
exactly N tokens (binary search over the passage length, needs a running
server; prints the prompt between `[` and `]`).

**Qwen3-4B (pure full-attn, arch `qwen3`, no recurrent layers)**: the kv-chain
mechanics work (restore loads, cached_tokens correct), but the passage-
repetition test FAILS because the 4B doesn't follow the "repeat word for word"
instruction reliably — it paraphrases/confirms instead. This is a model-
capability issue, NOT a kv-chain bug. The arch fix (see quirks) is what makes
the 4B not emit garbage; the test prompt is just too demanding for it.

Tests run with `--reasoning off --reasoning-budget 0` (llama_run.sh) and
`temperature: 0` (llama_prompt.sh): reasoning models otherwise burn the whole
n_ctx on thinking tokens and get capped mid-reasoning; temperature=0 keeps
output deterministic for the fidelity checks.

## 4b. DSV4F support — DONE

The disk hash-chain cache works for DeepSeek-V4-Flash (arch `deepseek4`,
`llama_kv_cache_dsv4`). Qwen paths stay green (run `llama_run_unittests.sh` on
the 27B after every change). the `dsv4f` test covers DSV4F restore fidelity.

### How DSV4F differs from Qwen

Qwen3.8 = 16 full-attn + 48 Gated DeltaNet. Its state splits cleanly into
  "per-token KV (additive, windowable)" + "recurrent R/S (fixed-size tail)".
DSV4F is NOT that shape. `llama_kv_cache_dsv4` owns FIVE groups
(see src/llama-kv-cache-dsv4.h):
  - kv_raw  (ISWA: base+swa)  -> per-token, windowable like Qwen attn
  - kv_csa / kv_hca / kv_lid  -> COMPRESSED K caches (ratios 4 / 128 / 4),
                                 prefix-style rows (row i covers [i*ratio,(i+1)*ratio))
  - csa_state / hca_state / lid_state -> compressor RING states (fixed-size)

### The part-selection flags (see include/llama.h)

  - ATTN_ONLY (16): per-token KV part only. on dsv4 = kv_raw (NOT the rings,
    NOT the compressed caches). on the hybrid / pure-attn caches == FULL_ONLY.
    used for the per-chunk .kvcache file.
  - TAIL_ONLY (32): everything EXCEPT the per-token KV. on dsv4 = the three
    compressed K caches + the three compressor rings (NOT kv_raw). on the
    hybrid / pure-attn caches == PARTIAL_ONLY. used for the .rscache tail file.

Why TAIL_ONLY (and not PARTIAL_ONLY, and not FULL):
  - FULL (flags=0) writes kv_raw FIRST, and its state_read CLEARS kv_raw before
    loading it. the .rscache is loaded AFTER the per-chunk .kvcache replay, so a
    FULL blob would WIPE the just-restored per-token rows -> garbled output.
  - PARTIAL_ONLY writes kv_raw + rings but NOT the compressed K caches. the tail
    prefill does NOT recompute them (it only processes tokens >= n_saved), so a
    PARTIAL_ONLY tail left the compressed prefix empty -> also garbled.
  - TAIL_ONLY is the exact complement of the .kvcache files: dsv4 = compressed
    K caches + rings, no kv_raw. the read side verifies the on-disk mode byte
    matches the requested part-selection (FULL/PARTIAL/ATTN/TAIL) and THROWS on
    mismatch, so a format/flag error is loud, not silent garbage.

### On-disk layout for DSV4F

   - <hash>.kvcache per chunk = ATTN_ONLY blob = kv_raw window [k*ubs,(k+1)*ubs),
     additive, APPEND on restore (chunk 0 wipes). [exactly what Qwen uses]
   - <hash>.rscache per boundary = TAIL_ONLY blob = compressed K caches + rings.
     Fixed-size tail, last wins. the restore loads ONLY the last chunk's copy.
    - Restore: append all matched .kvcache chunks, then set_data_ext(TAIL_ONLY)
      the last .rscache, set n_past, and prefill the remainder. The remainder is
      never empty: load_prefix searches tokens[0, n-1) (the last token is always
      re-prefilled, since a forward pass is required to produce logits).

## 5. Quirks / gotchas

- `--parallel > 1` is supported: `n_seq_max` (= `--parallel`) is baked into the
  kv-chain ROOT HASH (kv-chain-store.cpp `compute_root_hash`,
  `kv_chain_metadata`), so a chunk file written by a server with a different
  --parallel is a CLEAN MISS (different chain names), not the old
  `state_read: n_stream mismatch` fallback. Do NOT restructure the attn
  `state_write`/`state_read` per-stream loop to make the blob stream-agnostic:
  that loop is load-bearing and shared by all llama state-seq callers, and a
  previous attempt at it serialized the whole recurrent ring (~16 GB files).
  Cross-parallel REUSE of chunk files is deliberately unsupported (the .rscache
  is one slot's R/S). tested by `llama_unittest_parallel.sh`.
- The recurrent window filter (llama-memory-recurrent.cpp `state_write`) must
  CLOSE the open cell range on an out-of-window cell, not just `continue`:
  with --parallel > 1 the recurrent ring interleaves cells of multiple slots,
  so a bare skip let `cell_ranges` span the gap and tripped the
  `cell_count == cell_count_check` assert on the first chunk save.
- Chunk stride grid: every ubatch boundary must be a multiple of n_ubatch,
  including KV-full retry halvings (n_batch /= 2) — enforced by the startup
  guard (section 1); `-b == -ub` is trivially safe.
- Gated DeltaNet recurrent state is NON-INVERTIBLE: resume only FROM a
  boundary.
- Pure full-attn models (e.g. Qwen3-4B, arch `qwen3`) get a bare
  `llama_kv_cache` (no recurrent part). `state_write` MUST honor the
  PARTIAL_ONLY flag by serializing an empty state (real `n_stream` +
  cell_count=0 per stream — NOT n_stream=0, which trips the read-side
  "n_stream mismatch" assert). without this, PARTIAL_ONLY dumps the full attn
  state into the .rscache file and the restore reads it back as recr,
  wiping/desyncing the cache (garbage output). see llama-kv-cache.cpp.
- No-restart restore requires `seq_rm(0, -1)` BEFORE the chunk replay: the
  recurrent module's `rs_idx`/`head`/`used` counters are NOT reset by
  `set_data_window_ext` alone. without the pre-wipe, the second prompt in a
  session produces garbled/repetitive output.
- `slot.prompt.tokens` is NOT cleared by `slot.reset()` — after the first
  prompt+response, `prompt.n_tokens()` is non-zero for the next task. the
  restore condition must NOT guard on `n_tokens() == 0`.
- `has_mtmd` reflects model capability (mmproj present), NOT whether the
  prompt has media. use `get_text_tokens()` (not `get_tokens()`) to get the
  token list without the `!has_mtmd` assert.
- `-ub` has a floor of 32 (`-ub 8` is silently ignored -> 2048).
- With MTP speculative decoding enabled, the `cb_ubatch` hook fires TWICE at
  every completed boundary (once for the target prefill, once from the MTP
  draft path re-decoding the same tokens). the save hook dedupes on
  `slot.kv_chain_last_saved_pos` BEFORE its ON/OFF-GRID log, so the second fire
  is a silent no-op (no re-dump, no duplicate log line).
- Model-file mtime in the metadata blob uses std::filesystem's
  last_write_time (different epoch than unix time, logs as a negative number).
  Consistent across runs so the root hash is stable; do not "fix" it without
  bumping KV_CHAIN_VERSION (would orphan old caches).
- Each chunk ~152 MiB at -ub 32 (attn ~2 MiB + recr ~150 MiB); a 1000-tok
  prompt (~30 chunks) ~4.5 GiB. with the split, evicting old rs files
  reclaims ~92% of that for long chains.
- No integrity checksum on chunk files: the writer used to append a trailing
  u64 FNV-1a over the whole file, and the reader verified it — but that is a
  full read+hash of every (multi-hundred-MiB) recr file in the chain, which
  dominated restore time. It was removed from BOTH sides; a silent bit-flip
  now surfaces as garbage model output, not a clean cache miss (accepted
  trade-off). The size/magic/version/token checks in `read_chunk_file` still
  catch layout corruption. BUMP KV_CHAIN_VERSION if you ever re-add it (old
  no-checksum files would otherwise be mis-parsed).
- Only the .rscache of the TAIL chunk is ever read (the recurrent tail is a
  single fixed-size object; the middle rs files are superseded). a .kvcache
  MISSING in phase 1 (fs::exists walk) is a clean break: the replay streams
  the chunks before the break (tail rs = the break-1 chunk's) and prefills the
  rest. a .kvcache PRESENT in phase 1 but failing to read DURING the replay
  (eviction race / on-disk corruption) is harsher: by then the attn rows of
  the earlier chunks are already in the KV cache, but the recurrent tail
  (loaded only after the loop) is not - those rows are orphaned (no valid
  recurrent state to resume from), so the replay loop WIPES the seq and falls
  back to a 100% prefill (a partial restore would produce garbage). a FAILED
  TAIL .rscache read is different again: there is no earlier rs to fall back
  to. `load_prefix` validates the tail up front and, on failure, DISCARDS the
  ENTIRE restore (100% prefill) and, if the file is still on disk, DELETES it
  (corrupt/stale) so the re-prefill re-saves a clean one instead of
  re-reading + re-deleting it every request.
- devops/llama_run.sh: server stdout goes to log FILES ONLY (an inherited
  stdout pipe makes pipe-EOF-waiting callers hang); $log is a symlink to the
  newest timestamped log.
- devops/llama_test.sh supports `[cmd:PATH]` directives (runs a bash script
  with $KV_CACHE_DIR in env) for filesystem mutations between prompts.
- devops/llama_prompt.sh uses stream mode with `max_tokens: 512` cap to
  prevent infinite generation loops in tests.
