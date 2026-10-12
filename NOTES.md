# Hash-chain KV cache — current state

Base: llama.cpp **v0.6.0** (tag `v0.6.0`, 69ecae922), branch `hash-chain-kv`
(fork `github.com/riven8192/llama.cpp-kvchain`). A local hack, not upstream-grade
— see `../../docs/project-plan.md` for the design intent. The CODE is the source
of truth for HOW things work; this file holds what/where + test status + quirks.

What it does: during prefill, after each completed `--ubatch`-sized chunk the
slot's KV state for that chunk is streamed to disk. On a later request (even after
a restart) the server reconstructs as much of the prompt's KV as possible from
disk and prefills only the remainder. Prefill on this hardware costs minutes; a
restore costs seconds. That asymmetry is the whole point.

The model under test is selected via `LLAMA_HF_REF` in `devops/env.sh` (passed to
llama-server as `-hf`). Three archs are supported and tested: Qwen3.8-27B,
Qwen3.8-Flash-Next (arch `qwen4exp`), and DSV4F (arch `deepseek4`). See §3.

## 1. On-disk layout

Chunk k = tokens `[k*ubs, (k+1)*ubs)` where `ubs == --ubatch`. Only COMPLETE
chunks are persisted; a trailing partial chunk is always re-prefilled. One FLAT
directory, up to three files per chunk:

- `<hash>.kvcache` — ATTN_ONLY blob. the per-token KV rows for exactly the chunk
  window, additive across chunks (restore: chunk 0 wipes, later chunks APPEND).
- `<hash>.cmcache` — COMP_ONLY blob. written only for archs with a "comp" part
  (`has_comp()`: `deepseek4` and `qwen4exp` — see §3). this chunk's comp rows,
  additive. absent for the plain Qwen 27B (2-file arch, byte-identical to the
  Qwen path).
- `<hash>.rscache` — TAIL_ONLY blob. the fixed-size recurrent tail (the Gated
  DeltaNet R/S on Qwen; the compressor rings on dsv4; the recurrent tail on
  qwen4exp). last write wins; only the LAST chunk's copy is ever read.

Each file: `u32 magic, u32 version, u32 hash32, u32 n_tokens, token[n],
u32 blob_size, blob[blob_size]`. The token IDs are stored so a file is
self-describing and a hash collision / stale file is a clean miss, not garbage.
There is NO trailing checksum — verifying one costs a full pass over every
multi-hundred-MiB recr file and dominated restore time; we trust the device (the
magic/version/size/token checks in `read_chunk_file` still catch layout damage).

`hash_k = FNV-1a64(hash_{k-1} + chunk_k_tokens)`; `hash_0` seeds off a ROOT hash
= FNV-1a64 over a metadata blob (stat-only model identity, chunk size, cache
dtypes, rope, `n_seq_max`, a single version number). The version covers BOTH the
file layout and the root-hash blob — bump it whenever either changes so stale
files become a clean miss instead of a mis-parse. Current value:
`tools/server/kv-chain-store.cpp`.

## 2. Where things live

- `tools/server/kv-chain-store.{h,cpp}` — file format, hash chain, two-phase
  restore, LRU eviction + touch. everything is commented there.
- `include/llama.h` — public API additions: windowed state-seq variants
  (`llama_state_seq_get_size_window_ext` / `_get_data_window_ext` /
  `_set_data_window_ext`), the part-selection flags (`FULL_ONLY` / `ATTN_ONLY` /
  `TAIL_ONLY` / `COMP_ONLY` / `APPEND`), and the per-ubatch hook
  `llama_context_params.cb_ubatch`.
- `src/llama-context.cpp` — the `cb_ubatch` hook fired in `decode`'s ubatch loop;
  the kv-chain restore helpers (`llama_kvchain_set_idx_window`,
  `llama_kvchain_invalidate_graph`, `llama_kvchain_sync`).
- `src/llama-memory.h` — the `state_write`/`state_read` virtuals now take
  `pos_lo`/`pos_limit` (+ `sinfos_in` on read).
- per-cache part selection + window filter: `src/llama-kv-cache.cpp`,
  `src/llama-kv-cache-dsv4.cpp`, `src/llama-memory-recurrent.cpp`,
  `src/llama-memory-hybrid{,-iswa,-idx}.cpp`.
- `src/llama-memory-hybrid-idx.cpp` — the qwen4exp QSA indexer (mem_idx) is a
  SEPARATE plain `llama_kv_cache` that must track the main attn cache
  cell-for-cell; its COMP_ONLY serialization lives here.
- `tools/server/server-context.cpp` — all server integration:
  - save: `kv_chain_save_prefill_ubatch` (per-slot, fires only on the ubs grid;
    the MTP draft re-fire at the same pos is deduped via
    `slot.kv_chain_last_saved_pos`).
  - restore: the `SLOT_STATE_STARTED` block, replays matched chunks (attn, then
    comp, then tail rs last), then `n_past = n_saved` and prefills the rest.
  - `launch_slot_with_task`: drops the slot's stale previous-task prompt buffer
    (fixes the `raw_left` assert on slot reuse — see quirks).
  - `pre_decode`: an exhaustive slot classifier builds
    `kv_chain_slots_to_incorporate` — the single source of truth for which slots
    may add tokens / draft / decode this iteration. When a full-chunk prefill
    slot AND an interleave slot (partial-prefill / decode) both exist, a 50/50
    coin decides: the chunk slot gets batch exclusivity (the interleave slots are
    left untouched, retry next iteration) or the interleave slots mix freely.
    Without the coin the chunk saves would starve every concurrent generation
    behind the longest prefill. Every per-slot loop early-outs with
    `!kv_chain_slots_to_incorporate.contains(slot)`.
  - native in-memory prefix caching and context checkpoints are both disabled
    when kv-chain is set (the disk chain is authoritative; checkpoints would
    break the ubatch grid).
  - grid-safety startup guard: `-b`/`-ub` must be a power-of-2 ratio (the KV-full
    retry halves `n_batch`, so the ratio must survive arbitrary halvings) or the
    server exits(1); dsv4 additionally requires `-ub % 128 == 0`.

## 3. The three archs and their state split

- **Qwen3.8-27B** — 16 full-attention layers (real per-token KV) + 48 Gated
  DeltaNet linear-attention layers (a FIXED-SIZE recurrent state per layer, no
  per-token entries). → 2 files: `.kvcache` (the 16 attn layers) + `.rscache`
  (the recurrent tail). No comp part.
- **DSV4F** (`deepseek4`, `llama_kv_cache_dsv4`) — five cache groups
  (`src/llama-kv-cache-dsv4.h`): `kv_raw` (per-token) → `.kvcache`; three
  COMPRESSED K caches `kv_csa/kv_hca/kv_lid` (prefix-style, row i covers tokens
  `[i*ratio,(i+1)*ratio)`, ADDITIVE) → `.cmcache`; three compressor RING states
  (fixed-size) → `.rscache`. Restore order: append all `.kvcache`, then all
  `.cmcache`, then the last `.rscache`; the remainder prefill rebuilds comp rows
  for tokens >= n_saved. A missing `.cmcache` breaks the chain like a missing
  `.kvcache`. The comp rows are a separate additive file (not in the tail)
  because the full prefix would be O(N^2) in every tail otherwise, and because
  attention over the restored prefix needs the prefix's completed comp rows.
- **Qwen3.8-Flash-Next** (`qwen4exp`, `llama_memory_hybrid_idx`) — adds a QSA
  (sparse-attention) INDEXER: a SEPARATE plain `llama_kv_cache` (`mem_idx`) that
  must track the main attn cache cell-for-cell (a hard assert in `qwen4exp.cpp`
  enforces `idx.n_kv == attn.n_kv`). Layout: main attn → `.kvcache`; the idx
  rows (its COMP_ONLY part) → `.cmcache`; the recurrent tail → `.rscache`.

  The idx (`.cmcache`) restore goes through `llama_kvchain_set_idx_window`
  (`src/llama-context.cpp`), NOT the generic set_data path — and it is a
  **NON-mirror** restore: the idx is a plain `llama_kv_cache`, so it is restored
  into its OWN cells via `find_slot` (exactly like the attn `.kvcache` restore),
  not by adopting the attn cache's slot layout. (An earlier version adopted the
  attn layout as a mirror, but the `.cmcache` blob is serialized under the
  SAVING slot's stream while the adopted layout sits in the RESTORING slot's
  stream — under `--parallel > 1` those differ, so the mirror check rejected the
  restore with "mirrored cache holds cells this one does not" and fell back to a
  full prefill. `find_slot` is stream-robust; the cell-for-cell invariant still
  holds because both caches are restored from the same positions.) The idx
  restore also invalidates the compute graph after each chunk
  (`llama_kvchain_invalidate_graph`) because qwen4exp QSA sizes its windows from
  `n_kv` at graph-build time.

## 4. Config + eviction

- `--kv-chain-dir <path>` — empty/unset = feature OFF (zero behavior change).
- `--kv-chain-limit-gb N` — default 16 (`devops/env.sh` passes 100); 0 = no limit.
- Eviction: LRU by mtime, run after each write; a file vanishing mid-restore is a
  benign "cache ends here". Stray `.tmp` files removed at startup.
- LRU touch on restore: `load_prefix` touches every matched `.kvcache`/`.cmcache`,
  the TAIL `.rscache`, plus the `.rscache` of every chunk index a multiple of the
  stride (8) below the tail — so intermediate fork-point rs files survive LRU
  (a future prompt forking off the chain halfway still finds its recurrent
  state). `check_fork` asserts the resulting mtime gap pattern.
- `KVCHAIN_VERBOSE=1` (env, OFF by default) — enables the diagnostics: the
  `/tmp/rt-save` + `/tmp/rt-restore` round-trip dumps, the qwen4exp idx-K tensor
  dumps, and the per-layer `KVCHAINDBG` trace logs. OFF means the normal
  prefill/restore path pays no diagnostic I/O (the dumps do a GPU sync +
  `tensor_get` each) and no per-layer logging. Turn it on when bringing up a new
  arch (the dumps let you byte-compare a no-restore prefill against a restore).

## 5. Tests

Run the whole matrix with `devops/llama_unittests_models.sh`. It runs
`devops/llama_unittests.sh` once per (model × parallel) — 27B `-ub 32`, DSV4F
`-ub 128`, Flash-Next `-ub 32`, each at `--parallel 1` and `2` — and captures
each combo's exit code from the pipeline (via `PIPESTATUS`, since a bare
`| tee` reports tee's exit and masks a failing suite), teeing each to
`.unittest-output-<model>-p<par>.log` (gitignored; grep those instead of
re-running), then prints a per-combo PASS/FAIL table and exits non-zero on any
failure.

Current status: **all 6 combos PASS** (no segfaults, the qwen4exp cell-for-cell
assert never fires).

The suite is one server session; the active `check_*` blocks are at the bottom of
`llama_unittests.sh`. The oracle: a restore must be byte-identical to the
no-restore prime AND `cached_tokens` must hit the expected multiple-of-`ubs`
count. Checks:
- `check_restore` — prime then same-prompt restore, 2-token pinned reply.
- `check_fork` — A/A/B/B hash-chain fork (B shares A's first 3 chunks) + the
  rs-touch mtime gap pattern.
- `check_ubatch_edge_cases` — prompt lengths 255/256/257 around a chunk boundary
  (256 = exact multiple → the last token stays re-prefilled).
- `check_rscache_file_deletion` — delete middle rs files; the tail rs is the only
  one ever read, so restore still succeeds.
- `check_kvcache_file_deletion` — delete a middle kvcache file; the chain breaks
  there (clean partial restore).
- `check_verbatim` — the strongest: prime then restore, verbatim passage reply.
- `check_coherence` — long filler + passage; the reply must contain all the
  distinctive phrases (proves attention over the whole restored prefix).
- `check_parallel_prefill` — two LONG identical prompts fired at the SAME instant
  (backgrounded), so the prefills actually overlap on separate slots. Asserts no
  idx-restore mirror mismatch. This is the regression test for the qwen4exp
  `--parallel > 1` restore bug (see §3).

`check_fork` and `check_kvcache_file_deletion` are skipped from the full run for
DSV4F (detected by `LLAMA_HF_REF`): they hardcode `-ub 32` expectations (3 chunks
= 96 tokens; evict window [32,96] = 1–3 chunks) that don't hold at DSV4F's
`-ub 128`. They still run if passed explicitly (`./llama_unittests.sh fork`).

**Qwen3-4B** (pure full-attn, arch `qwen3`, still selectable in `devops/env.sh`):
the mechanics work, but the 4B is NON-DETERMINISTIC on a LONG generation when ANY
prefix is restored (the no-kv-chain native repeat diverges just as much — a 4B
long-gen numerics issue, NOT a kv-chain bug; the 27B is fully deterministic).
Short responses match byte-for-byte. Low priority.

## 6. Quirks / gotchas

- `--parallel > 1`: `n_seq_max` (= `--parallel`) is baked into the ROOT HASH, so a
  chunk file from a server with a different `--parallel` is a CLEAN MISS.
  Cross-parallel REUSE of chunk files is deliberately unsupported (the `.rscache`
  is one slot's recurrent state).
- The attn `state_write`/`state_read` per-stream loop in `llama-kv-cache.cpp` is
  LOAD-BEARING and shared by all llama state-seq callers. Do NOT restructure it to
  make the blob stream-agnostic — a previous attempt serialized the whole
  recurrent ring (~16 GB files).
- **Stale prompt buffer on slot reuse (the `raw_left` assert).** a REUSED slot
  still holds the PREVIOUS task's `prompt.tokens` when the new task starts:
  `reset()` keeps them on purpose (native in-memory reuse + LCP slot selection
  rely on it), and `prompt_load()` moves a cached prompt back in AFTER reset,
  right before launch. the classifier computes `raw_left = task - prompt`, which
  goes negative when the previous prompt+response was longer → assert. with
  kv-chain the native reuse is bypassed and the disk chain is authoritative, so
  `launch_slot_with_task` drops the buffer. it must be done THERE (not in
  `reset()` — `prompt_load` re-populates after; not in the STARTED block — the
  classifier runs first).
- The recurrent window filter (`llama-memory-recurrent.cpp state_write`) must
  CLOSE the open cell range on an out-of-window cell, not just `continue`: with
  `--parallel > 1` the recurrent ring interleaves multiple slots' cells, so a
  bare skip let `cell_ranges` span the gap and tripped the
  `cell_count == cell_count_check` assert on the first chunk save.
- Pure full-attn models (e.g. Qwen3-4B) get a bare `llama_kv_cache` (no
  recurrent part). `state_write` MUST honor PARTIAL_ONLY/TAIL_ONLY/COMP_ONLY by
  serializing an EMPTY state (the real `n_stream` + cell_count=0 per stream — NOT
  `n_stream=0`, which trips the read-side "n_stream mismatch" assert).
- No-restart restore requires `seq_rm(0, -1)` BEFORE the chunk replay: the
  recurrent module's `rs_idx`/`head`/`used` counters are NOT reset by
  `set_data_window_ext` alone.
- A mid-replay read failure (a file present in phase 1 but gone/corrupt by replay
  time) WIPES the seq and falls back to a 100% re-prefill — a partial restore
  would leave orphaned attn rows with no recurrent state to resume from. A FAILED
  TAIL `.rscache` read likewise discards the whole restore and DELETES the corrupt
  file so the re-prefill re-saves a clean one.
- Gated DeltaNet recurrent state is NON-INVERTIBLE: resume only FROM a boundary.
- `has_mtmd` reflects model capability (mmproj present), NOT whether the prompt
  has media. use `get_text_tokens()` (not `get_tokens()`) to get the token list
  without the `!has_mtmd` assert.
- `-ub` has a floor of 32 (`-ub 8` is silently ignored → 2048).
- With MTP speculative decoding the `cb_ubatch` hook fires TWICE at every
  completed boundary (target prefill + the MTP draft re-decode). the save hook
  dedupes on `slot.kv_chain_last_saved_pos` BEFORE its ON/OFF-GRID log, so the
  second fire is a silent no-op.
- Model-file mtime in the metadata blob uses `std::filesystem`
  `last_write_time` (a different epoch than unix time, so it logs as a negative
  number). Consistent across runs so the root hash is stable; do NOT "fix" it
  without bumping the version (would orphan old caches).
- Each chunk is ~152 MiB at `-ub 32` (attn ~2 MiB + recr ~150 MiB); a 1000-token
  prompt (~30 chunks) is ~4.5 GiB.

## 7. devops scripts

- `llama_build.sh` — full Vulkan Release build. WIPES the build dir and rebuilds
  everything (no incremental — it's the accepted way to avoid flaky builds).
- `llama_run.sh` — starts the server in the background (kills any previous one
  first). server stdout goes to log FILES only (an inherited stdout pipe makes
  pipe-EOF-waiting callers hang); `$log` is a symlink to the newest timestamped
  log. args after `--` are passed to llama-server.
- `llama_prompt.sh "prompt"` — sends a prompt, streams the reply, prints
  cached/prompt/response token counts. Uses STREAM mode against
  `/v1/chat/completions` (NOT `/v1/completions`) — the tested models are chat
  models, and a bare prompt through the raw-text endpoint degenerates into
  repetition. `temperature 0` + a fixed `seed` for determinism.
- `llama_test.sh "p1" "p2" "[restart]" "p3" -- <server args>` — clears the cache,
  (re)starts the server, runs prompts. supports `[cmd:PATH]` directives to run a
  bash script (with `$KV_CACHE_DIR` in env) for filesystem mutations between
  prompts.
- `llama_unittests.sh [check_name]` / `llama_unittests_models.sh` — see §5.
- `git_show_state.sh --base <ref>` / `git_diff_chunks.sh --base <ref> -x <path>`
  — session-start helpers (see `../../docs/project-plan.md` "Next step").
- `git_push.sh` — pushes with the token read from `github-key.txt` (inline, never
  stored in `.git/config`).
