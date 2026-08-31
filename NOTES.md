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
mid-chunk, hence partial-chunk reuse is descoped). Hash chain: FNV-1a64,
`hash_k = H(hash_{k-1} + chunk_k_tokens)`; root dir = FNV-1a64 over a
metadata blob (stat-only model identity, chunk size, dtypes, rope, format
version) so a different model/config is a clean miss, never garbage.

## 3. Config

- `--kv-chain-dir <path>` (empty = feature off, zero behavior change),
  `--kv-chain-limit-gb N` (default 16; devops/env.sh passes 100).
- Eviction: LRU by mtime after each write; a file vanishing mid-restore is a
  benign "cache ends here". Stray `.tmp` files removed at startup.

## 4. Test status (all PASS, Qwen3.8-27B, -ub 32 -b 32)

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

## 5. Quirks / gotchas

- Chunk stride grid: every ubatch boundary must be a multiple of n_ubatch,
  including KV-full retry halvings (n_batch /= 2) — enforced by the startup
  guard (section 1); `-b == -ub` is trivially safe.
- Gated DeltaNet recurrent state is NON-INVERTIBLE: resume only FROM a
  boundary, never roll back.
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
  bumping KV_CHAIN_FORMAT_VERSION (would orphan old caches).
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
