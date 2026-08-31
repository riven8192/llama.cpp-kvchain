# Hash-chain KV cache - current state

Base: llama.cpp b10520, branch `hash-chain-kv`. Local hack, not upstream-grade
(see `../docs/project-plan.md` for the design). The code is the source of truth
for HOW things work; this file only holds what/where + test results + quirks.
Model: Qwen3.8-27B (16 full-attn layers + 48 Gated DeltaNet recurrent layers),
selected via `LLAMA_HF_REF` (devops/env.sh, default
`unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_XL`) — passed to llama-server as `-hf`, which
resolves the exact snapshot file (and downloads it if missing). The kv-chain
root hash uses the resolved path, so cache identity follows the ref.

CURRENT STATE: v3 — chunk state is split into `<hash>.kvcache` (attn) +
`<hash>.rscache` (recr) in a FLAT cache dir. Middle rs files can be evicted
without breaking restore (recurrent state is a tail object, last write wins;
missing rs chunks are simply skipped in the restore loop). Restore works both
after a restart AND in the same session (no-restart): the pre-restore
`seq_rm(0, -1)` properly resets the recurrent module's internal state.
See `kv-chain-store.cpp` for the full layout/algorithm.

## 1. Where things live

- Chunk file format + hash chain + store: `tools/server/kv-chain-store.{h,cpp}`
  (layout, metadata blob, checksum, eviction, LRU touch all commented there).
- Public API additions: `include/llama.h` — windowed state-seq variants
  (`llama_state_seq_get_size_window_ext` / `_get_data_window_ext` /
  `_set_data_window_ext`), flags `FULL_ONLY` (attn only) and `APPEND`
  (restore without seq_rm), per-ubatch hook `llama_context_params.cb_ubatch`.
- State plumbing: `src/llama-context.cpp` (hook fired in `decode`'s ubatch loop,
  ~line 1976), `src/llama-memory.h` (virtuals take `pos_lo`/`pos_limit`),
  `src/llama-kv-cache.cpp` (window filter + APPEND),
  `src/llama-memory-recurrent.cpp` (window filter),
  `src/llama-memory-hybrid{,-iswa}.cpp` (FULL_ONLY/PARTIAL_ONLY selection).
- Server integration: `tools/server/server-context.cpp`
  - save path: `kv_chain_save_prefill_ubatch` (~line 912), armed per slot,
    dumps only when pos is on the bs-grid
  - restore path: SLOT_STATE_STARTED block (~line 3520), replays matched chunks
  - native in-memory prefix caching is bypassed when kv_chain is set (~3317)
  - context checkpoints are disabled when kv_chain is set (~3651, keeps the
    ubatch grid on-stride; chunk files make them redundant anyway)
  - grid-safety startup guard: `-b`/`-ub` must have a power-of-2 ratio or the
    server exits(1) (~line 1445)

## 2. Design in one paragraph

Chunk k = tokens `[k*bs,(k+1)*bs)`, `bs == --ubatch`. Each chunk file stores
ONLY its own window (attn rows + recurrent rows for exactly bs positions) —
constant file size, no duplication. attn is additive across chunks (restore:
chunk 0 wipes, later chunks APPEND); the recurrent state is one tail object per
position (restore: each chunk overwrites, the last wins — it cannot roll back
mid-chunk, hence partial-chunk reuse is descoped). A missing middle .rscache is
simply skipped on restore (empty recr_blob, no set_data); the chain is
truncated at `usable` = last chunk that has BOTH files. One version number,
`KV_CHAIN_VERSION`, covers both the file layout and the root-hash metadata
blob (bump on any change to either; see kv-chain-store.cpp). Hash chain:
FNV-1a64, `hash_k = H(hash_{k-1} + chunk_k_tokens)`; root dir = FNV-1a64 over a
metadata blob (stat-only model identity, chunk size, dtypes, rope, version)
so a different model/config is a clean miss, never garbage.

## 3. Config

- `--kv-chain-dir <path>` (empty = feature off, zero behavior change),
  `--kv-chain-limit-gb N` (default 16; devops/env.sh passes 100).
- Eviction: LRU by mtime after each write; a file vanishing mid-restore is a
  benign "cache ends here". Stray `.tmp` files removed at startup.

## 4. Test status

All 5 PASS on Qwen3.8-27B (the model the tests were written for), `-ub 32 -b 32`:

- `devops/llama_unittest_1.sh`: full-chain restore with [restart] (v2-era,
  still passes with v3 files).
- `devops/llama_unittest_2.sh`: zero-behavior-change (no --kv-chain-dir).
- `devops/llama_unittest_3a.sh`: v3 full-chain restore, NO restart — prime +
  resend same session, cached_tokens=352, 6/6 phrases.
- `devops/llama_unittest_3b.sh`: delete 9 of 11 .rscache (keep newest 2),
  NO restart -> still full restore (352), 6/6 phrases.
- `devops/llama_unittest_3c.sh`: delete 3rd-oldest .kvcache, NO restart ->
  chain breaks at chunk 2, cached_tokens=64, 6/6 phrases.
- grid-safety guard: `-b 96 -ub 32` -> FATAL + exit(1).

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

## 4b. DSV4F support — plan for next session (NOT yet implemented)

Goal: make the disk hash-chain cache work for DeepSeek-V4-Flash (arch
`deepseek4`, `llama_kv_cache_dsv4`) so the user's DSV4F sessions get the same
seconds-restore as Qwen. Qwen paths MUST stay green (run `llama_run_unittests.sh`
on the 27B after every change).

### Why the naive Qwen approach does NOT transfer

Qwen3.8 = 16 full-attn + 48 Gated DeltaNet. Its state splits cleanly into
"per-token KV (additive, windowable)" + "recurrent R/S (fixed-size tail)".
DSV4F is NOT that shape. `llama_kv_cache_dsv4` owns FIVE groups
(see src/llama-kv-cache-dsv4.h):
  - kv_raw  (ISWA: base+swa)  -> per-token, windowable like Qwen attn
  - kv_csa / kv_hca / kv_lid  -> COMPRESSED K caches (ratios 4 / 128 / 4)
  - csa_state / hca_state / lid_state -> compressor RING states (fixed-size)

The compressed caches are the problem: `llama_kv_cache_dsv4::state_write`
writes them with `n_rows = (pos_max+1)/ratio` — a PREFIX from row 0, NOT the
`[pos_lo,pos_limit)` window. They are duplicative/prefix, not additive, so you
cannot APPEND them per chunk the way you append raw rows. Storing them at every
boundary is O(N^2) disk; storing only at the end loses mid-chain hits.

### The key realization: mirror the ORIGINAL in-memory checkpoint

The user observed that plain llama.cpp (no disk) skips ~90% of prefill when a
new prompt shares a 90% prefix. That is the **context-checkpoint** feature
(`-ctx-checkpoints`, default 32), and it is the model to copy:

- create_checkpoint (tools/server/server-context.cpp:2405) snapshots ONLY
  `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` at selected boundaries. It does NOT
  snapshot per-token KV (that stays resident in the single slot's memory).
- restore (server-context.cpp:3463-3498) finds the latest checkpoint with
  `pos_max <= pos_next`, calls `load_tgt(..., PARTIAL_ONLY)`, sets n_past, and
  lets the NORMAL prefill continue from there.
- The per-token KV for the common prefix is "free" (resident); the recurrent /
  compressed TAIL is restored from the small PARTIAL_ONLY snapshot; the
  compressed K ROWS for the tail are RECOMPUTED by the ordinary forward pass as
  it decodes from the restored boundary. NO special recompute code, NO O(N^2).

For DSV4F specifically, `PARTIAL_ONLY` in `llama_kv_cache_dsv4::state_write`
(line 1535) writes `kv_raw` (the SWA half) + the three `*_state` rings, and
SKIPS the three compressed K caches. So the on-disk "tail" object = rings +
raw-SWA. The compressed caches are deliberately NOT stored; they regenerate
during tail prefill. This is the "cheap recompute" — it is the normal decode.

### Therefore the disk design (generalizes the Qwen 2-file split)

  - `<hash>.kvcache` per chunk = `kv_raw` per-token window [k*bs,(k+1)*bs),
    additive, APPEND on restore (chunk 0 wipes).  [exactly what we have]
  - `<hash>.rscache` per boundary = a `PARTIAL_ONLY` snapshot via the EXISTING
    `llama_state_seq_get_data_ext` / `set_data_ext` (the SAME calls
    common_prompt_checkpoint::update_tgt/load_tgt use, common/common.cpp:2268/2308)
    — NOT the custom windowed split we use today. For DSV4F that's rings +
    raw-SWA; for Qwen it's the Gated DeltaNet R/S. Fixed-size tail, last wins.
  - Restore: APPEND all matched .kvcache chunks, then `set_data_ext(PARTIAL_ONLY)`
    the LAST .rscache, set n_past to that boundary, continue normal prefill
    (compressed rows recompute for free on DSV4F).

This is a SMALLER change than the B1/B2/B3 options considered earlier, and it
is provably correct because it is the same mechanism llama.cpp already runs
in-memory. We do NOT need new part-selection flags or windowed variants for the
rs part — PARTIAL_ONLY is a fixed tail object.

### Concrete steps for next session

1. VERIFY the one empirical assumption first (cheapest possible signal): does
   DSV4F produce byte-identical output (temp 0, seed fixed) for
   (a) full prefill vs (b) disk-restore = [APPEND kvcache chunks] +
   [set_data_ext(PARTIAL_ONLY) last rscache] + [normal prefill from boundary]?
   Build a devops unittest modeled on llama_unittest_1.sh but for DSV4F. If
   (a)==(b), the whole design is sound and step 2+ is mechanical.
2. Make `kv_chain_store::save()` write the .rscache file as a PARTIAL_ONLY
   snapshot via `llama_state_seq_get_data_ext(ctx, seq, PARTIAL_ONLY)` (drop the
   current windowed recr dump for the rs file). Keep .kvcache as the FULL_ONLY
   window. Bump KV_CHAIN_VERSION (layout of .rscache changes) — old caches
   orphan, that is fine.
3. Make the restore path in server-context.cpp use `set_data_ext(PARTIAL_ONLY)`
   for the last rscache instead of the windowed PARTIAL_ONLY set. Keep the
   per-chunk .kvcache APPEND replay as-is.
4. DSV4F gets a bare `llama_kv_cache_dsv4` (NOT hybrid), so the Qwen
   `llama_memory_hybrid` FULL_ONLY/PARTIAL_ONLY selection code does NOT run for
   it — the flag handling lives in `llama_kv_cache_dsv4::state_write` itself,
   which already branches on PARTIAL_ONLY. Confirm the FULL_ONLY path (kvcache
   file) for DSV4F writes ONLY kv_raw's per-token rows (check that FULL_ONLY on
   dsv4 does not also emit rings/compressed — if it does, the .kvcache file
   would be wrong and you'd need to scope it).
5. Keep Qwen green: the Qwen path goes through `llama_memory_hybrid`, which is
   a DIFFERENT code path than `llama_kv_cache_dsv4`. Changing the .rscache
   format (step 2) affects BOTH — so after the change, re-run the 27B
   unittests (llama_unittest_1/2/3a/3b/3c) AND add a DSV4F unittest. The Qwen
   .rscache content (Gated DeltaNet R/S) should be unchanged in meaning, only
   the serialization call site moves from windowed-ext to plain-ext.
6. Grid-safety: DSV4F ubatch boundaries must still land on the bs-grid for the
   hash chain. The existing startup guard (-b/-ub power-of-2 ratio) applies
   unchanged. DSV4F's own `reset_rs_idx_for_ubatches` / comp_plan machinery is
   internal to the forward pass and does not affect our boundary grid.

### Open questions to resolve in-session (do not guess)

- Does FULL_ONLY on `llama_kv_cache_dsv4` emit ONLY kv_raw per-token rows, or
  does it also drag in rings/compressed? (Read state_write line 1535-1567:
  FULL_ONLY takes the `!partial_only` branch which writes kv_raw + csa/hca/lid
  + rings. So FULL_ONLY on dsv4 is actually "everything" — the .kvcache file
  would be WRONG. You likely need a NEW flag or to call the sub-caches
  directly. RESOLVE THIS before writing the save path.)
- Is the compressed-row recompute during tail prefill deterministic given
  (reloaded raw KV + reloaded rings)? The in-memory checkpoint relies on the
  same assumption and works, so yes — but confirm with the step-1 test.
- DSV4F model file / ref for testing: ask the user (env.sh currently has Qwen
  refs; the DSV4F IQ3_S ref is not set). It must fit in shared memory with the
  kv-cache dir; run with the SAME single-GPU/unified-memory setup.

## 5. Quirks / gotchas

- Chunk stride grid: every ubatch boundary must be a multiple of n_ubatch,
  including KV-full retry halvings (n_batch /= 2) — enforced by the startup
  guard (section 1); `-b == -ub` is trivially safe.
- Gated DeltaNet recurrent state is NON-INVERTIBLE: resume only FROM a
  boundary, never roll back.
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
- Model-file mtime in the metadata blob uses std::filesystem's
  last_write_time (different epoch than unix time, logs as a negative number).
  Consistent across runs so the root hash is stable; do not "fix" it without
  bumping KV_CHAIN_VERSION (would orphan old caches).
- Each chunk ~152 MiB at -ub 32 (attn ~2 MiB + recr ~150 MiB); a 1000-tok
  prompt (~30 chunks) ~4.5 GiB. with the split, evicting old rs files
  reclaims ~92% of that for long chains.
- devops/llama_run.sh: server stdout goes to log FILES ONLY (an inherited
  stdout pipe makes pipe-EOF-waiting callers hang); $log is a symlink to the
  newest timestamped log.
- devops/llama_test.sh supports `[cmd:PATH]` directives (runs a bash script
  with $KV_CACHE_DIR in env) for filesystem mutations between prompts.
- devops/llama_prompt.sh uses stream mode with `max_tokens: 512` cap to
  prevent infinite generation loops in tests.
