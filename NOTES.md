# Hash-chain KV cache - current state

Base: llama.cpp b10520, branch `hash-chain-kv`. Local hack, not upstream-grade
(see `../docs/project-plan.md` for the design). The code is the source of truth
for HOW things work; this file only holds what/where + test results + quirks.
Model: Qwen3.8-27B (16 full-attn layers + 48 Gated DeltaNet recurrent layers),
selected via `LLAMA_HF_REF` (devops/env.sh, default
`unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_XL`) — passed to llama-server as `-hf`, which
resolves the exact snapshot file (and downloads it if missing). The kv-chain
root hash uses the resolved path, so cache identity follows the ref.

CURRENT STATE: v4 — chunk state is split into `<hash>.kvcache` (attn) +
`<hash>.rscache` (tail) in a FLAT cache dir. The .kvcache holds an ATTN_ONLY
blob (per-token KV rows for the chunk window only); the .rscache holds a
TAIL_ONLY blob (the tail object, last write wins, only the last chunk's copy is
read). On Qwen (16 full-attn + 48 Gated DeltaNet) TAIL_ONLY == PARTIAL_ONLY
(the Gated DeltaNet R/S). On DSV4F (llama_kv_cache_dsv4) TAIL_ONLY == the three
compressed K caches + the three compressor rings — the per-token kv_raw rows
are NOT in the .rscache (the .kvcache files already carry them, and a FULL-mode
blob read would clear kv_raw and wipe the just-restored rows). Restore works
both after a restart AND in the same session (no-restart): the pre-restore
`seq_rm(0, -1)` properly resets the recurrent module's internal state.

## 1. Where things live

- Chunk file format + hash chain + store: `tools/server/kv-chain-store.{h,cpp}`
  (layout, metadata blob, two-phase restore, eviction, LRU touch all commented
  there).
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
ONLY its own window (attn rows for exactly bs positions) — constant file size,
no duplication, and NO trailing checksum (verifying one costs a full pass over
every (multi-hundred-MiB) recr file and dominated restore time; we trust the
storage device). The .kvcache blob is ATTN_ONLY (per-token KV rows only); the
.rscache blob is TAIL_ONLY (the tail object, last write wins). attn is additive
across chunks (restore: chunk 0 wipes, later chunks APPEND); the tail is one
fixed-size object per position (restore: the LAST chunk's rs wins — it cannot
roll back mid-chunk, hence partial-chunk reuse is descoped). Restore is
two-phase: phase 1 walks the chain with `fs::exists()` only (cheap, no reads)
to find the first missing .kvcache (the break) and `usable` = the last chunk
with an rs file; phase 2 reads the .kvcache of every chunk in [0,usable) and
the .rscache of the TAIL chunk ALONE (the earlier rs files are superseded, so
they are never read). A missing middle .rscache is simply skipped (empty
recr_blob, no set_data); the chain is truncated at `usable`. One version number,
`KV_CHAIN_VERSION`, covers both the file layout and the root-hash metadata blob
(bump on any change to either; see kv-chain-store.cpp). Hash chain: FNV-1a64,
`hash_k = H(hash_{k-1} + chunk_k_tokens)`; root dir = FNV-1a64 over a metadata
blob (stat-only model identity, chunk size, dtypes, rope, version) so a
different model/config is a clean miss, never garbage.

## 3. Config

- `--kv-chain-dir <path>` (empty = feature off, zero behavior change),
  `--kv-chain-limit-gb N` (default 16; devops/env.sh passes 100).
- Eviction: LRU by mtime after each write; a file vanishing mid-restore is a
  benign "cache ends here". Stray `.tmp` files removed at startup.

## 4. Test status

All PASS on Qwen3.8-27B (the model the tests were written for), `-ub 32 -b 32`:

- `devops/llama_unittest_1.sh`: full-chain restore with [restart] (v2-era,
  still passes with v3 files).
- `devops/llama_unittest_2.sh`: zero-behavior-change (no --kv-chain-dir).
- `devops/llama_unittest_3a.sh`: v3 full-chain restore, NO restart — prime +
  resend same session, cached_tokens=352, 6/6 phrases.
- `devops/llama_unittest_3b.sh`: delete 9 of 11 .rscache (keep newest 2),
  NO restart -> still full restore (352), 6/6 phrases.
- `devops/llama_unittest_3c.sh`: delete 3rd-oldest .kvcache, NO restart ->
  chain breaks at chunk 2, cached_tokens=64, 6/6 phrases.
- `devops/llama_unittest_4.sh`: forked chains (PROMPT_A, PROMPT_A, PROMPT_B,
  PROMPT_B, where B = A with a mid-insertion) -> 0/352/96/352: the 2nd B finds
  A's trunk (3 chunks) + B's own branch (8 chunks) saved by the 1st B.
- `devops/llama_unittest_5_dsv4f.sh`: DSV4F (DeepSeek-V4-Flash, arch `deepseek4`)
  full-chain restore with [restart] — prime + resend, 6/6 phrases,
  cached_tokens=352. proves the ATTN_ONLY/TAIL_ONLY split + compressed-K-cache
  restore works end-to-end on the second architecture.
- `devops/llama_unittest_6.sh`: prompt whose length is an EXACT multiple of the
  ubatch (-ub 64, 256 tokens). prime + resend. currently FAILS (the §4b bug
  below) — the resend restores all 256 tokens but returns an empty response.
  In the general case, the response is not empty, but non-sensical, meaning:
  it can reply to dialog, with an C comment-block: `/* ... */` or json, or xml,
  or regular English text that is obviously not a proper reply to the prompt.
  With the deepseek-v4-flash model, the output is non-sensical, with qwen 3.8
  it hits a GGML_ASSERT which breaks the response, leaving it empty. Focus on
  Qwen for now, as a hard error is easier to troubleshoot.
  helper: `devops/llama_make_exact_prompt_len.sh <N>` converges a prompt to
  exactly N tokens (binary search over the passage length, needs the server
  running; prints the prompt between `[` and `]`). llama_test.sh gained a
  `[flush]` directive (wipe the cache dir mid-run, no restart).
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

## 4b. Known bug: prompts whose length is an exact multiple of the ubatch size

**Status: OPEN — no fix committed yet.** unittest_6 reproduces it
deterministically on Qwen3.8.

**Symptom.** A prompt that tokenizes to exactly `N*bs` tokens restores fully on
the 2nd send (`restored 256 tokens ... 0 tokens left to prefill`, `kv-chain:
full prompt restored, starting decode`) but the response is empty/garbage:
Qwen3.8 samples an EOS/special first token (empty text, immediate stop; other
runs emit `"\n\n"` + a degenerate `...` chain). DSV4F: coherent-looking
nonsense. Adding 1 token to the prompt (N+1 -> N restored, 1 re-prefilled)
makes it correct. Restart does not help. This is the only case that breaks.

**N.B.: THE FOLLOWING IS WRITTEN BY QWEN 3.8 AND MAY NOT BE ACCURATE:**

**Root cause (state management, not the cache).** All `N` chunks are saved and
restored correctly. The bug is the zero-prefill transition: a full restore
(`n_saved == n_prompt`) jumps straight to `SLOT_STATE_GENERATING` with an
*empty* batch (server-context.cpp, the `kv_chain_full_restore` branch in
pre_decode). No forward pass ever runs over the prompt's final position, so the
sampler never gets real prompt logits and the first decode samples from a
wrongly-seeded chain. The normal path (>=1 token left to prefill) decodes the
last prompt token with `output=true`, which seeds the sampler. Side effect:
`stats.t_prompt_last` is never set, so in a build with asserts active the first
sampled token trips `GGML_ASSERT(t_prompt_last > 0)` in
`server_slot_stats::update_gen_last()` (server-common.h:376, called from
post_decode). In the current Release build the assert is silent (ggml_abort
fflushes stdout only; the log is a file, so the abort message is lost and the
server dies without a trace in the log) — which is why unittest_6 shows an
empty response, not a crash.

**Why the obvious fix does NOT work (attempted, reverted).** Re-decoding the
last prompt token in place (add it to the batch at pos N-1 with its restored
KV) fails with "Invalid input batch": `llama_kv_cache` requires
`pos > seq_pos_max` for every token of the seq (the restored KV already occupies
pos N-1, so X == Y violates X < Y). Re-decoding an existing position is not
possible — the token's position must be REMOVED from the cache first.

**N.B.: THE FOLLOWING IS WRITTEN BY QWEN 3.8 AND MAY NOT BE ACCURATE:**

**Fix direction (next session).** Roll the restore back by ONE chunk and let
the normal prefill machinery re-prefill it:
1. `slot.mem.seq_rm(slot.id, (n-1)*bs, -1)` — drops the last chunk's attn rows
   (attn is per-token and freely removable; it does NOT touch the recurrent
   module, whose state is moved only by set_data_ext).
2. `llama_state_seq_set_data_ext(TAIL_ONLY)` of `chunks[n-2].recr_blob` —
   re-rolls the recurrent tail to the (n-1) boundary. THIS WORKS because each
   .rscache holds a FULL-PREFIX recurrent snapshot, not a per-window delta:
   `llama_memory_recurrent::state_write` collects ALL cells of the seq (the
   pos_lo/pos_limit filter matches every cell, since the module only holds
   cells of the prefix); its state_read does `seq_rm(seq, -1, -1)` first, so it
   is an idempotent overwrite. (Verified in src/llama-memory-recurrent.cpp.)
3. Reset `n_past = (n-1)*bs`, `slot.prompt.tokens.keep_first(n_past)`, adjust
   `stats.n_prompt_cached` / `metrics.add_prompt_cached(-bs)`; clear
   `kv_chain_restored` so the `seq_rm(p0, -1)` + `kv_chain_restored = false`
   below the branch is a harmless no-op. The fill loop then re-prefills the
   last chunk normally (KV recompute + recurrent update + logits with
   output=true) — exactly the partial-restore path, triggered from a full hit.
   Single-chunk prompt (n==1): nothing to roll back to -> discard the whole
   restore (seq_rm(0,-1), n_past=0) and prefill from scratch.
   Cost: one extra chunk re-prefill per exact-multiple hit (~150 MiB recurrent
   recompute on Qwen at -ub 64) — acceptable, and it makes the restored prefix
   byte-identical to a full prefill (verify with unittest_6: response must be
   `OK`, cached_tokens = N-bs).
   The full-restore branch currently at server-context.cpp (~line 3864, the
   `slot.state = SLOT_STATE_GENERATING` skip) is what must be replaced.

**Test harness.** `devops/llama_unittest_6.sh` (committed): builds a 257-token
prompt (primes + restores fine, the control) and a 256-token prompt (the bug),
via `llama_make_exact_prompt_len.sh`; sequence is LARGE, LARGE, [restart],
[flush], EXACT, EXACT at -ub 64 -b 64. Expected after the fix: responses
OK/OK/OK/OK and cached_tokens 0/256/0/192.

## 4c. DSV4F support — DONE

The disk hash-chain cache works for DeepSeek-V4-Flash (arch `deepseek4`,
`llama_kv_cache_dsv4`). Qwen paths stay green (run `llama_run_unittests.sh` on
the 27B after every change). unittest_5 covers DSV4F restore fidelity.

### How DSV4F differs from Qwen

Qwen3.8 = 16 full-attn + 48 Gated DeltaNet. Its state splits cleanly into
"per-token KV (additive, windowable)" + "recurrent R/S (fixed-size tail)".
DSV4F is NOT that shape. `llama_kv_cache_dsv4` owns FIVE groups
(see src/llama-kv-cache-dsv4.h):
  - kv_raw  (ISWA: base+swa)  -> per-token, windowable like Qwen attn
  - kv_csa / kv_hca / kv_lid  -> COMPRESSED K caches (ratios 4 / 128 / 4),
                                 prefix-style rows (row i covers [i*ratio,(i+1)*ratio))
  - csa_state / hca_state / lid_state -> compressor RING states (fixed-size)

### The two new flags (see include/llama.h)

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

   - <hash>.kvcache per chunk = ATTN_ONLY blob = kv_raw window [k*bs,(k+1)*bs),
     additive, APPEND on restore (chunk 0 wipes). [exactly what Qwen uses]
   - <hash>.rscache per boundary = TAIL_ONLY blob = compressed K caches + rings.
     Fixed-size tail, last wins. the restore loads ONLY the last chunk's copy.
   - Restore: append all matched .kvcache chunks, then set_data_ext(TAIL_ONLY)
     the last .rscache, set n_past, and prefill the remainder. NOTE: on a full
     chain hit the "remainder" is empty - that transition is the broken path of
     the §4b bug (do not "fix" it by skipping the forward pass).

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
- No integrity checksum on chunk files: the writer used to append a trailing
  u64 FNV-1a over the whole file, and the reader verified it — but that is a
  full read+hash of every (multi-hundred-MiB) recr file in the chain, which
  dominated restore time. It was removed from BOTH sides; a silent bit-flip
  now surfaces as garbage model output, not a clean cache miss (accepted
  trade-off). The size/magic/version/token checks in `read_chunk_file` still
  catch layout corruption. BUMP KV_CHAIN_VERSION if you ever re-add it (old
  no-checksum files would otherwise be mis-parsed).
- Two-phase restore reads the .rscache of the TAIL chunk only. a FAILED ATTN
  (.kvcache) read at chunk k is a benign "cache ends here": walk back to the
  last fully-loaded chunk that has a valid rs file (its recr IS the tail we
  already read) and prefill the rest. a FAILED TAIL .rscache read is different:
  the recurrent tail is a single fixed-size object and we only ever read the
  LAST one, so there is no earlier rs to fall back to. `load_prefix` then
  DISCARDS the ENTIRE restore (n_loaded=0 -> 100% prefill) and, if the file is
  still on disk, DELETES it (corrupt/stale) so the re-prefill re-saves a clean
  one instead of re-reading + re-deleting it every request.
- devops/llama_run.sh: server stdout goes to log FILES ONLY (an inherited
  stdout pipe makes pipe-EOF-waiting callers hang); $log is a symlink to the
  newest timestamped log.
- devops/llama_test.sh supports `[cmd:PATH]` directives (runs a bash script
  with $KV_CACHE_DIR in env) for filesystem mutations between prompts.
- devops/llama_prompt.sh uses stream mode with `max_tokens: 512` cap to
  prevent infinite generation loops in tests.
