# Hash-chain KV cache - current state

Base: llama.cpp b10520, branch `hash-chain-kv`. Local hack, not upstream-grade
(see `../docs/project-plan.md` for the design). The code is the source of truth
for HOW things work; this file only holds what/where + test results + quirks.
NEXT TASK: section 0 (split kv/rs into 2 files) - fully specified, not yet
implemented.
Model: Qwen3.8-27B (16 full-attn layers + 48 Gated DeltaNet recurrent layers),
selected via `LLAMA_HF_REF` (devops/env.sh, default
`unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_XL`) — passed to llama-server as `-hf`, which
resolves the exact snapshot file (and downloads it if missing). The kv-chain
root hash uses the resolved path, so cache identity follows the ref.

CURRENT STATE (as of this commit): the chunk state is STILL the single
`<hash>.kvchunk` file (v2: attn + recr in one file). Section 0 below is the
NEXT TASK (split into `.kvcache` + `.rscache`) — fully specified, NOT yet
implemented. Everything else in this file describes the working v2 state.

## 0. NEXT TASK: split the chunk state into 2 files (*.kvcache + *.rscache)

### 0.1 Why

rs (recurrent) is ~140 MiB of the ~152 MiB/chunk — the expensive part. Today
EVERY chunk file carries its own rs snapshot (duplicated by design, so any
chunk is self-contained). That means a 38-chunk prompt costs ~5.8 GiB of which
~5.4 GiB is rs, of which only ONE rs (the tail) is ever read on restore.
Splitting into 2 files lets LRU eviction drop old rs files first, reclaiming
~92% of disk for long chains, while a usable chain still restores exactly.

Correctness argument (why zeroed rs is fine): attn row i is a function of the
prefix [0..i]; the recurrent tail at k*bs is a function of prefix [0..k*bs].
The engine reads ONLY the tail cell of the recurrent state (n_rs_seq == 0). So
for a restored prefix [0, T*bs):
- attn rows [0, T*bs) come from the kvcache files (all required),
- recurrent state = the snapshot from the LAST rs file in the chain; the
  per-chunk rs values for EARLIER chunks are never read.
Therefore: a chunk's rs file may be missing and we just stream a zeroed rs
blob for it (deserializes as valid zeros, immediately overwritten by later
chunks; the last chunk's real rs wins). The only constraint:
    usable_chain = min(last_kvcache_chunk, last_rscache_chunk)
A chain with kv files 1..8 and rs files {4,6} is usable up to chunk 6:
replay kv 1..6, rs = zeros for 1..5 + file-data for 6.

### 0.2 On-disk layout (ONE dir per root hash, contains BOTH extensions)

    <cache_dir>/<root_hash_hex>/<chunk_hash_hex>.kvcache   (attn only)
    <cache_dir>/<root_hash_hex>/<chunk_hash_hex>.rscache   (recr only)

- Same chunk_hash as today (FNV-1a chain over the chunk's tokens) - the hash
  does not change, only the file naming/splitting.
- `.kvcache` file: u32 magic KVC1, u32 version=3, u32 hash32, u32 n_tokens,
  llama_token[n_tokens], u32 attn_size, attn_blob[attn_size], u64 fnv1a
  checksum. (i.e. the v2 layout MINUS the recr part.)
- `.rscache` file: identical header (magic, version=3, hash32, n_tokens,
  tokens) + u32 recr_size + recr_blob[recr_size] + u64 checksum.
  (i.e. the v2 layout MINUS the attn part.)
- Both files keep their own fnv1a checksum (self-contained corruption
  detection). Both carry the token IDs (validation against the prompt).
- Bump KV_CHAIN_VERSION 2 -> 3 AND KV_CHAIN_FORMAT_VERSION 1 -> 2 (old
  single-file caches live under the same root dir but never match the new
  names; also the root dir itself changes because format_version is in the
  metadata blob -> old files are cleanly orphaned, never read).

### 0.3 Save path (kv_chain_store::save, tools/server/kv-chain-store.cpp)

- NO behavioral change to WHEN we save (still per on-grid ubatch boundary,
  during prefill only). We now always write BOTH files, no skipping, even
  though most rs files will never be read (we cannot know in advance which
  chunk becomes the tail).
- write_chunk() becomes: write `<hash>.kvcache.tmp` + rename, then
  `<hash>.rscache.tmp` + rename. If the .kvcache already exists but the
  .rscache is missing (crash between the two writes), still write the missing
  one (idempotent). entry_bytes for the eviction pre-check = kv_file +
  rs_file sizes.
- The two dumps (dump_window FULL_ONLY / PARTIAL_ONLY) already exist; just
  route each into its own file.

### 0.4 Restore path (kv_chain_store::load_prefix)

Current algorithm walks the chain and stops at the first missing file. NEW
algorithm, single walk:

    k = 0; prev_hash = 0; usable = 0
    while k < n_chunks:
        block = tokens[k*bs .. (k+1)*bs)
        h = hash_chunk(block, prev_hash)
        kv_file = dir/(hex(h) + ".kvcache")
        if not exists(kv_file): break            # kv is REQUIRED for every chunk
        if not read_kv(kv_file): break           # corrupt -> stop
        if kv.tokens != block: break             # header validation
        rs_file = dir/(hex(h) + ".rscache")
        if exists(rs_file) and read_rs(rs_file):
            recr_blob = rs.recr_blob
            usable = k + 1                       # last chunk WITH an rs file
        else:
            recr_blob = ZEROS (see 0.4.1)        # rs missing/evicted -> zeros
        chunks.push_back({attn_blob, recr_blob, tokens})
        prev_hash = h; k++
    # after the loop, TRUNCATE chunks to `usable`:
    #   a chunk k >= usable has NO rs file (by definition of usable), and its
    #   attn rows are useless without the recurrent tail, so drop them.
    chunks.resize(usable)
    *n_tokens = usable * bs

- Touch (utimensat mtime=now) EXACTLY the files that were actually read and
  replayed: chunks[0..usable-1]'s .kvcache files, plus the ONE .rscache file
  of chunk (usable-1) (the tail). Do NOT touch any other file. (Example: 7 kv
  files on disk, usable=5 -> touch 5 kvcache + 1 rscache.)
- INFO log at the end (the exact format the user asked for):
      kv-chain: 10 prompt chunks, first 7 kv-files found on disk, last rs-file found for chunk 5
  i.e. `kv-chain: %zu prompt chunks, first %zu kv-files found on disk, last rs-file found for chunk %zu`
  (if usable == 0: `... no usable chain (no kv+rs pair on disk)` or similar;
  if the kv chain ends before the prompt: the "first N kv-files found" already
  conveys it.)

0.4.1 Zeroed rs blob: construct a minimal valid PARTIAL_ONLY seq-state blob
  whose recurrent cells deserialize to zeros for the window [k*bs,(k+1)*bs).
  HOW (verify against src/llama-memory-recurrent.cpp state_read before coding):
  the recurrent state_read reads u32 cell_count then per-cell meta + data; a
  blob with cell_count=0 (or cells with zeroed data of the right size) should
  wipe-and-leave-zero. SAFEST approach if a hand-built zero blob proves
  fiddly: build the zero blob ONCE at startup by dumping a PARTIAL_ONLY window
  from a freshly-cleared context (or from the live context at pos 0 before any
  prefill) and reusing that byte pattern for every missing-rs chunk. The
  server-context restore loop does NOT change: it still calls
  set_data_window_ext(PARTIAL_ONLY) per chunk (overwriting), so a zero blob
  behaves exactly like "no state yet".
  NOTE: if a zero/empty recr blob turns out to be REJECTED by state_read
  (e.g. it expects >= 1 cell), fall back to: stop the chain at `usable`
  WITHOUT truncating attn (i.e. restore attn for [0, usable*bs) and rs only
  from the tail) - but prefer making the zero blob work, it is cleaner.

### 0.5 Eviction (kv_chain_store::evict_oldest)

- ONE flat scan of the root_hash dir over BOTH `.kvcache` and `.rscache`
  files (regular files only), sort by mtime ascending, delete from the front
  until `total_bytes_cur + need_bytes <= limit_bytes`. No tree-integrity
  checks (per project plan): deleting an old rs file just lowers `usable` for
  chains that would have used it; deleting a kv file shortens the kv chain.
  Both are benign "cache ends here" cases on the next restore.
- total_bytes_cur: at startup, sum sizes of BOTH extensions in the root dir;
  maintain it on save (+ both files) and on eviction (- deleted size).
- Pre-write eviction call passes need_bytes = kv_entry + rs_entry (both
  files about to be written).
- Stray `.tmp` cleanup at startup: match BOTH `.kvcache.tmp` and
  `.rscache.tmp`.

### 0.6 What does NOT change

- server-context.cpp restore loop (per-chunk set_data_window_ext calls),
  save hook (kv_chain_save_prefill_ubatch), hash chain definition, root-hash
  metadata (except the format_version bump), grid-safety guard, config flags,
  devops scripts.
- The chunk_hash values are unchanged by this refactor (same tokens, same
  FNV-1a) - only the filenames/extensions and the internal layout change.

### 0.7 Test plan (add to devops/, run with -ub 32 -b 32, Qwen3.8-27B)

1. Unit (kv-chain-store, could be a small C++ test or scripted):
   - write chunk -> two files appear, same hash stem, checksums valid.
   - read back -> attn from .kvcache, recr from .rscache, round-trip equal.
   - delete a middle .rscache -> load_prefix: usable stops at the previous
     rs file, attn for chunks after that is dropped, *n_tokens = usable*bs.
   - delete a middle .kvcache -> load_prefix stops there entirely.
   - corrupt one file's payload byte -> that file reads as corrupt -> chain
     stops (kv: stop; rs: treat as missing -> usable may drop, not crash).
2. Integration (extend llama_unittest_1.sh or a new llama_unittest_3.sh):
   - prime a ~10-chunk prompt; verify 10 .kvcache + 10 .rscache on disk.
   - restart, resend: expect the exact log line
     `kv-chain: 10 prompt chunks, first 10 kv-files found on disk, last rs-file found for chunk 9`
     (or whatever the true tail is), cached_tokens = usable*bs, 6/6 phrases.
   - DELETE .rscache files for chunks 0..7 (keep 8,9) -> resend: expect
     `... first 10 kv-files found on disk, last rs-file found for chunk 9`
     still full restore (zeros for 0..8, real rs for 9), 6/6 phrases.
   - DELETE .rscache for chunks 8,9 too (keep 0..7) -> usable drops to 8 ->
     cached_tokens = 8*bs, coherent output.
   - DELETE .kvcache for chunk 5 -> chain stops at 5, coherent.
   - Eviction: run with --kv-chain-limit-gb small enough to force evictions
     (or unit-test evict_oldest directly): verify oldest-by-mtime (both
     extensions mixed) go first, and total stays under the limit.
3. Acceptance: temperature-0 output byte-identical to the uncached run in
   every scenario; with the feature disabled, zero behavior change (unchanged
   code paths, but re-run llama_unittest_2.sh to be sure).

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

- prime (1260 tok): 38 chunks, ALL ~152 MiB (constant, no ramp).
- 100% match (restart, same prompt): restored 1216/1260 (chain stops at the
  chaotic tail edge), coherent.
- partial/divergence: shared prefix restored, chain stops at divergence, rest
  prefilled, coherent.
- crash-restart (kill -9 after prime): chunks intact, restore works.
- corruption (flip a payload byte): checksum mismatch -> chain stops -> prefill
  fallback, no crash.
- `devops/llama_unittest_1.sh`: 250-word passage repeated verbatim; prime
  (cached_tokens:0) and restart (cached_tokens:320) BOTH reproduce 6/6
  distinctive phrases -> the full chain's attn+recurrent KV restores correctly.
- `devops/llama_unittest_2.sh`: zero-behavior-change counterpart (no
  --kv-chain-dir, same prompt twice in one session): native
  get_common_prefix reuse (cached_tokens:370 of 374) reproduces 6/6 phrases.
- grid-safety guard: `-b 96 -ub 32` -> FATAL + exit(1); `-b 64 -ub 32` ->
  starts, logs "grid-safe ... = 2 (power of 2)".
- `devops/llama_ubatch_probe.sh` (29474-tok prompt, -b 2048 -ub 2048 -c 65536):
  with the checkpoint gate, prefill = 14x2048 ON-GRID + one 802 OFF-GRID tail
  (29474 = 14*2048 + 802). Before the gate the tail was ragged (802/2044/4).
  Raggedness is TAIL-ONLY and deterministic.

## 5. Quirks / gotchas

- Chunk stride grid: every ubatch boundary must be a multiple of n_ubatch,
  including KV-full retry halvings (n_batch /= 2) — enforced by the startup
  guard (section 1); `-b == -ub` is trivially safe.
- Gated DeltaNet recurrent state is NON-INVERTIBLE: resume only FROM a
  boundary, never roll back.
- `-ub` has a floor of 32 (`-ub 8` is silently ignored -> 2048).
- Model-file mtime in the metadata blob uses std::filesystem's
  last_write_time (different epoch than unix time, logs as a negative number).
  Consistent across runs so the root hash is stable; do not "fix" it without
  bumping KV_CHAIN_FORMAT_VERSION (would orphan old caches).
- Each chunk ~152 MiB at -ub 32; a 1000-tok prompt (~30 chunks) ~4.5 GiB.
- devops/llama_run.sh: server stdout goes to log FILES ONLY (an inherited
  stdout pipe makes pipe-EOF-waiting callers hang); $log is a symlink to the
  newest timestamped log.
