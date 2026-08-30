# Hash-chain KV cache - current state

Base: llama.cpp b10520, branch `hash-chain-kv`.
Goal: disk-backed, content-addressed (hash-chained) KV cache for llama-server
(see `../docs/project-plan.md` for the design; this file is the as-built state).
Model: Qwen3.8-27B, hybrid memory (16 full-attn layers + 48 Gated DeltaNet
recurrent layers). Local hack - public API changed freely, not upstream-grade.

## 1. The state save/load API we build on (src/llama-context.cpp)

- Seq-level C API: `llama_state_seq_get_size_ext` / `_get_data_ext` / `_set_data_ext`
  (flags), plus the windowed variants we added: `llama_state_seq_get_size_window_ext`,
  `llama_state_seq_get_data_window_ext`, `llama_state_seq_set_data_window_ext`
  (extra `pos_lo`, `pos_limit` args).
- A seq state blob = u32 io_magic `0xaf143cd8` + u32 src_seq_id + per-module data.
  `state_seq_write_data`/`state_seq_read_data` dispatch to `memory->state_write/read`;
  each memory module filters cells by seq_id itself.
- Flags (llama.h): `NONE`, `PARTIAL_ONLY` (recurrent/SWA only, skips full attn KV),
  `FULL_ONLY` (inverse: attn only, skips recurrent - we added it), `APPEND` (on restore,
  skip the seq_rm so cells accumulate - we added it), `ON_DEVICE`.
- state virtuals (llama-memory.h:126-127) now take `pos_lo`, `pos_limit` (window
  `[pos_lo, pos_limit)`), threaded through kv-cache, recurrent, hybrid, hybrid-iswa,
  msa, iswa, dsa, dsv4 + the test-batch-alloc mock.
  - attn (llama-kv-cache.cpp): `add_cell &&= pos >= pos_lo && pos < pos_limit`.
  - recurrent (llama-memory-recurrent.cpp): `if (cell.pos < pos_lo || >= pos_limit) continue;`
- Composites select parts: hybrid/hybrid-iswa write/read attn iff `FULL_ONLY || !PARTIAL_ONLY`,
  recurrent iff `PARTIAL_ONLY || !FULL_ONLY`. dsv4 comp-state (a non-override helper) does
  NOT take the window - the dsv4 composite stops forwarding it there (dsv4 is not the test model).

## 2. Why the window + append were needed (the two hard parts)

- **Attn is additive, recurrent is a single tail object.** The attn K/V rows are
  genuinely per-position (the model attends over all of them - this is why prefill
  cost decays with prefix length, so the attn cache is the real, irreducible, growing
  cache). The recurrent state is ONE object per position (the accumulated DeltaNet
  matrix); at inference the engine reads only the TAIL cell's state (`n_rs_seq == 0`
  here, so no rollback planes). So:
  - attn must ACCUMULATE across chunks -> the restore `state_read_meta` does `seq_rm`
    first (wipes), which would trash earlier chunks. Fixed with `APPEND` (skip seq_rm):
    chunk 0 restores with APPEND cleared (wipes any stale slot cells), later chunks
    with APPEND set (append). Each call scatters its rows at the stored positions.
  - recurrent can OVERWRITE per chunk -> each `state_read` wipes + rewrites; only the
    last chunk's tail state survives, which is exactly what the engine reads. No append needed.
- **Each chunk stores ONLY its own window** `[k*bs,(k+1)*bs)` (attn rows + recurrent rows
  for exactly those bs positions). Constant file size, no duplication (each position stored
  once across the chain). This replaced the earlier full-prefix-per-chunk layout (which
  ramped in size with the prompt).

## 3. Chunk file format (tools/server/kv-chain-store.{h,cpp})

- `<cache_dir>/<root_hash_hex>/<chunk_hash_hex>.kvchunk`:
  one directory per model/config identity (the root_hash dir), one file per chunk.
  u32 magic `KVC1`, u32 version (=2), u32 chunk_hash (LOW 32 BITS of the 64-bit hash),
  u32 n_tokens (=bs), llama_token[bs] (this chunk's own tokens, for validation),
  u32 attn_size, attn_blob[attn_size], u32 recr_size, recr_blob[recr_size],
  u64 fnv1a64 checksum of everything before it.
- Both blobs are self-contained seq-state blobs (each carries io_magic, src_seq, module
  headers) so they feed straight to the `*_set_data_window_ext` API.
- attn_blob dumped with `FULL_ONLY`, recr_blob with `PARTIAL_ONLY`, both for `[pos_lo,pos_hi)`.
- Atomic `.tmp`+`rename`. `read_chunk` verifies magic+version+checksum and bounds n_tokens
  against the file size; a bad version/checksum/size = miss (chain stops).

## 4. Hash chain (deterministic from the token list + model/config identity)

- `chunk_size` = `--ubatch` (bs), used PURELY as a boundary stride (NOT the runtime prefill
  batch size). chunk k covers tokens `[k*bs,(k+1)*bs)`. Only COMPLETE chunks are persisted;
  the trailing partial block is never written, always re-prefilled.
- `hash_0 = H(chunk_0_tokens, prev=0)`; `hash_k = H(chunk_k_tokens, prev=hash_{k-1})`,
  H = FNV-1a (FULL 64-bit) over the chunk's own bs token ids seeded by the parent.
  A longer prompt reuses identical early chunks. `load_prefix` walks from the root,
  stopping at the first missing/corrupt file, returning the matched
  `kv_chain_chunk{attn_blob,recr_blob,tokens}` list (in order) + `*n_tokens = n_chunks*bs`.
  It also VERIFIES each file's header token IDs against the prompt (mismatch = chain
  stops, never a garbage restore).
- **root_hash** = FNV-1a64 over a canonical metadata blob (length-prefixed fields, struct order):
  `KV_CHAIN_FORMAT_VERSION` (int32), `chunk_size` (int32), model file size (int64, stat only),
  model file mtime (int64 seconds, stat only), arch string (`llama_model_arch_name`),
  ftype string (`llama_ftype_name(llama_model_ftype)`), `type_k` (ggml_type), `type_v` (ggml_type),
  `rope_scaling_type` (int32), `rope_freq_base` (float bits), `rope_freq_scale` (float bits),
  model path (string). Gathering is cheap: one stat(), no file reads.
  The root_hash dir isolates different models/configs so a stale cache for a different
  model is never read (clean miss instead of garbage).

## 5. Write path (tools/server/server-context.cpp)

- Per-ubatch hook: `llama_context_params.cb_ubatch(cb_ubatch_data)` (llama.h), copied into
  cparams (llama-cparams.h, llama-context.cpp), fired in `llama_decode`'s ubatch loop after
  each ubatch commits. the 2nd arg is `ubatch.pos[n_tokens-1]` (the boundary just
  completed); the hook ALSO reads the authoritative pos from the memory module.
- **How the server feeds the prompt (verified, not assumed):** the server does NOT send the
  whole prompt as one `llama_decode`. `update_slots` fills a batch of <= `n_batch` tokens
  (the `-b` value) per iteration, and the slice loop (server-context.cpp:2924) feeds each
  batch to `llama_decode` in `min(n_batch, remaining)` views. with `-b == -ub == B` each
  view is one `llama_decode` call -> one ubatch of B tokens. a prompt of N tokens therefore
  becomes `floor(N/B)` full B-ubatches + one trailing `N mod B` partial ubatch. the
  `print_timing` progress lines are printed in `post_decode` (AFTER each `llama_decode`),
  so the diffs between consecutive `n_tokens =` values = the per-decode batch sizes.
- **Context checkpoints are DISABLED when kv_chain is set** (server-context.cpp:3621,
  `if (kv_chain) do_checkpoint = false;`). WHY: upstream checkpoints break the prompt batch
  early at `checkpoint_offsets = {4+n_ubatch, 4}` to create a save point, which fragments
  the tail into ragged, OFF-GRID ubatches (observed: a 29474-tok prompt at -b 2048 gave a
  `802/2044/4` tail instead of one `802`). our chunk files ARE already superior checkpoints
  (attn+recurrent state at every B-boundary, content-addressed, survive restart), so the
  in-memory ones are redundant AND they desync the hash chain. with the gate, the tail is a
  single `N mod B` partial chunk (verified: 29474 -> 14x2048 ON-GRID + one 802 OFF-GRID).
  with kv-chain off, checkpoints run exactly as upstream (zero behavior change).
- `kv_chain_save_prefill_ubatch` (server-context.cpp:912): arms during prefill (armed in
  SLOT_STATE_STARTED, cleared at the transition to SLOT_STATE_GENERATING - decode-generated
  state is worthless to cache, and the clearing is also what keeps the per-ubatch log from
  spamming during generation). Reads pos from `llama_memory_seq_pos_max` (authoritative).
  Dumps a chunk ONLY when `pos % bs == 0` (on-grid); off-grid edges (the trailing partial
  chunk) are skipped. No dedup state needed (pos is strictly increasing). Calls
  `kv_chain->save(ctx, seq, pos_lo=chunk_lo, pos_hi=pos, chunk_hash, chunk_tokens)`.
- **Debug logging (INFO, kept for future debugging):**
  - `kv-chain[decode]: batch.size off n_tokens n_batch` - one per highlevel `llama_decode`
    call that processes a PROMPT slice (`n_tokens > 1` guard skips the per-token gen calls
    that would otherwise spam one line per output token).
  - `kv-chain[ubatch]: pos cb_n_pos_last ub_n ub_pos=[lo..hi] pos%bs ON/OFF-GRID` - one per
    internal ubatch, PREFILL ONLY (the `kv_chain_prefill_slot` guard means it never fires
    during generation). `pos` = memory-module seq_pos_max+1; `cb_n_pos_last` = ubatch.pos
    last (always `pos-1`, verified); `ub_n`/`ub_pos` = the ubatch's own token count + range
    (plumbed through `kv_chain_ubatch_state` from llama_decode's ubatch loop).

## 6. Restore path (server-context.cpp, SLOT_STATE_STARTED, n_past==0, slot empty, cache_prompt)

- **Native in-memory prefix caching is disabled when kv_chain is enabled**
  (server-context.cpp:3255). When `kv_chain` is set and `cache_prompt` is true, the
  whole native block - `slot.prompt.tokens.get_common_prefix(input_tokens)` (the LCP
  reuse), the alora invocation-start clipping, and the `n_cache_reuse` KV-shifting
  path - is bypassed and `n_past` is forced to 0. Reason: the disk chain is the
  single source of truth for prefix reuse; the native LCP reuse could set n_past to
  the LCP with the previous in-memory prompt (e.g. a divergent re-run reusing a
  stale in-memory prefix the disk chain does not have), desyncing n_past from the
  restored chain. A bypass notice is logged when skipped, and an UNEXPECTED
  warning fires if the native block is ever entered while kv-chain is enabled
  (last-resort fallback check - grep the run logs for "kv-chain: UNEXPECTED").
  With kv-chain disabled the native block runs exactly as upstream (zero behavior
  change; see llama_unittest_2.sh).
- `kv_chain->load_prefix(tokens, &n_saved)` -> list of matched chunks.
  - `load_prefix` walks `<cache_dir>/<root_hash_hex>/`, stopping at the first missing/corrupt file.
  - On hit, it `utimensat`-touches every matched file (mtime=now) so LRU eviction keeps the
    hottest chains alive (relatime mounts do not update mtime on read).
- For each matched chunk k in order:
  - attn: `set_data_window_ext(ctx, attn_blob, seq, FULL_ONLY | (k>0 ? APPEND : 0), k*bs, (k+1)*bs)`
  - recr: `set_data_window_ext(ctx, recr_blob, seq, PARTIAL_ONLY, k*bs, (k+1)*bs)`
- On success: push the restored prompt tokens, `n_past = n_saved`, set `kv_chain_restored`
  (skip the seq_rm truncation - recurrent can't roll back) and `kv_chain_full_restore`
  (whole prompt restored -> go straight to GENERATING, avoids the empty-batch assert).
  Also skip the [TAG_PROMPT_LOGITS] `n_past--` when restored.

## 7. Config

- `--kv-chain-dir <path>` (empty = feature off, zero behavior change), `--kv-chain-limit-gb N`
  (default 16 in common.h; devops/env.sh passes 100 explicitly). Eviction is LRU by mtime
  after each write; a file vanishing mid-restore is a benign "cache ends here". On startup,
  stray `.tmp` files in the root_hash dir are removed.
- KNOWN QUIRK: the model-file mtime in the metadata blob is std::filesystem's
  last_write_time (a different clock epoch than unix time, so it logs as a negative
  number). It is consistent across runs, so the root hash is stable; do not "fix" it
  to time_since_epoch without bumping KV_CHAIN_FORMAT_VERSION (old caches would then
  be orphaned).

## 8. Test status (all PASS, Qwen3.8-27B, -ub 32 -b 32)

- prime (1260 tok): 38 chunks, ALL exactly ~152 MiB (constant, no ramp).
- 100% match (restart, same prompt): restored 1216/1260 (chain stops at the chaotic tail
  edge), coherent.
- partial/divergence: restored the shared prefix, chain stops at divergence, rest prefilled,
  coherent.
- crash-restart (kill -9 after prime): chunks intact on disk, restore works.
- corruption (flip a payload byte): checksum mismatch -> chain stops -> prefill fallback,
  no crash.
- `devops/llama_unittest_1.sh`: 250-word passage repeated verbatim. prime (cached_tokens:0)
  and restart (cached_tokens:320, 10 chunks) BOTH reproduce 6/6 distinctive phrases ->
  proves the full chain's attn+recurrent KV restores correctly.
- `devops/llama_unittest_2.sh`: the zero-behavior-change counterpart. runs the server
  WITHOUT --kv-chain-dir (native impl), sends the SAME passage twice in one session (no
  restart - the in-memory KV cache does not survive a restart). prime (cached_tokens:0)
  and repeat (cached_tokens:370 of 374, native get_common_prefix reuse) BOTH reproduce
  6/6 phrases -> proves the native in-memory prefix cache is untouched when the feature
  is disabled.
- **ubatch-grid verification (`devops/llama_ubatch_probe.sh`, -ub 2048 -b 2048 -c 65536,
  29474-tok prompt):** with the checkpoint gate ON, prefill produced 14x2048 ON-GRID
  ubatches + ONE trailing 802 OFF-GRID partial chunk (29474 = 14*2048 + 802). BEFORE the
  gate the same prompt gave a ragged `802/2044/4` tail (checkpoints breaking the batch
  early). confirms the hash chain stays in sync for the whole prompt and only the single
  trailing partial chunk is re-prefilled. the middle of the prompt is always clean (the
  raggedness is tail-only, deterministic server behavior, not random).

## 9. Dev tooling (devops/)

- env.sh (model, port 50081, paths), llama_build.sh (cmake+ninja, Vulkan; check EXIT=$?,
  `| tail -1` masks the exit code), llama_run.sh (--keep-cache/--no-kv-chain/-- <args>;
  also WAITS for "listening on" before returning - llama_wait.sh was merged into it),
  llama_kill.sh, llama_prompt.sh (sends /v1/completions cache_prompt:true, prints
  `cached_tokens: N prompt_tokens: M` then the full text), llama_test.sh ([restart]
  markers, trailing `-- <extra run args>`; the `--` tail must come LAST),
  llama_unittest_1.sh (the disk hash-chain restore fidelity test above),
  llama_unittest_2.sh (the native in-memory prefix-cache check, no disk cache, no restart),
  llama_ubatch_probe.sh (paste a long prompt into it; sends it to the dev server and greps
  the `kv-chain[decode]`/`kv-chain[ubatch]` boundary lines out of the log - used to verify
  the ubatch grid / the checkpoint-gate fix above).
- llama_run.sh log handling: the server writes ONLY to $log.<ts>.log (never to stdout -
  an inherited stdout pipe makes pipe-EOF-waiting callers hang forever). $log is a
  SYMLINK to the newest tslog, so the readiness wait-loop always reads the current run.

## 10. Key constraints / gotchas

- Gated DeltaNet recurrent state is NON-INVERTIBLE: resume only FROM a boundary, never roll
  back to m<n. Never "strip the last token" from a chunk.
- `-ub` has a lower bound of 32 (`-ub 8` is silently ignored -> 2048). Use `-ub 32` for tests.
- `chunk_size` (--ubatch) is a BOUNDARY STRIDE. With `-b == -ub == B` the server feeds the
  prompt in B-sized `llama_decode` slices, each becoming ONE on-grid B-ubatch; only the
  trailing `N mod B` partial ubatch is off-grid (and skipped). The raggedness is TAIL-ONLY
  and deterministic - it is NOT random mid-prompt chaos. The one thing that USED to break
  the grid mid-tail was upstream context-checkpoints (batch breaks at {4+n_ubatch, 4});
  that is now disabled when kv_chain is set (see section 5). We only dump when
  `pos % bs == 0`.
- **n_batch vs n_ubatch (why two knobs):** n_batch (logical) = how many tokens the SERVER
  stages per llama_decode (outer slice loop, server-context.cpp:2924); n_ubatch (physical)
  = how many tokens ONE ggml graph eval / ubatch processes (the GPU micro-batch, controls
  peak prefill VRAM). invariant: n_ubatch = min(n_batch, n_ubatch) (llama-context.cpp:250),
  so n_ubatch can never exceed n_batch. the chunk stride is n_ubatch, so the hash chain
  stays in sync only if EVERY ubatch boundary is a multiple of n_ubatch.
- **The grid-safety rule: n_batch / n_ubatch must be a power of 2.** two reasons:
  (1) a single llama_decode of n_batch tokens splits into n_batch/n_ubatch ubatches - all
      on-grid only if n_batch is a multiple of n_ubatch;
  (2) on KV-full the server RETRIES with n_batch /= 2 and re-slices the SAME tokens
      (server-context.cpp:3907, off does not advance) - a mid-prompt retry re-decodes those
      tokens in a smaller slice, so the halved size must ALSO be a multiple of n_ubatch,
      i.e. n_batch/n_ubatch must survive arbitrary halvings => power of 2.
  `-b == -ub` (ratio 1 = 2^0) is the trivially-safe choice; `-b 2048 -ub 512` (ratio 4) is
  also fine. `-b 2048 -ub 300` is NOT (and the trailing N mod n_ubatch partial chunk is
   re-prefilled either way). ENFORCED in code: when --kv-chain-dir is set, the server
   exits(1) at startup unless n_batch % n_ubatch == 0 AND n_batch/n_ubatch is a power of
   2 (server-context.cpp, kv_chain_store construction). logs "kv-chain: grid-safe" on
   success, a FATAL message + exit on failure. with kv-chain off, no check (zero
   behavior change).
- Each chunk is ~152 MiB at -ub 32 (constant). A 1000-tok prompt (~30 chunks) is ~4.5 GiB.
  `KV_CHAIN_LIMIT_GB` default 100.
