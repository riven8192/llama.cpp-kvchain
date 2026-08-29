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
  u32 magic `KVC1`, u32 version (=2), u32 chunk_hash, u32 n_tokens (=bs),
  llama_token[bs] (this chunk's own tokens, for validation), u32 attn_size, attn_blob[attn_size],
  u32 recr_size, recr_blob[recr_size], u64 fnv1a64 checksum of everything before it.
- Both blobs are self-contained seq-state blobs (each carries io_magic, src_seq, module
  headers) so they feed straight to the `*_set_data_window_ext` API.
- attn_blob dumped with `FULL_ONLY`, recr_blob with `PARTIAL_ONLY`, both for `[pos_lo,pos_hi)`.
- Atomic `.tmp`+`rename`. `read_chunk` verifies magic+version+checksum; a bad version or
  checksum = miss (chain stops).

## 4. Hash chain (deterministic from the token list + model/config identity)

- `chunk_size` = `--ubatch` (bs), used PURELY as a boundary stride (NOT the runtime prefill
  batch size). chunk k covers tokens `[k*bs,(k+1)*bs)`. Only COMPLETE chunks are persisted;
  the trailing partial block is never written, always re-prefilled.
- `hash_0 = H(chunk_0_tokens, prev=0)`; `hash_k = H(chunk_k_tokens, prev=hash_{k-1})`,
  H = FNV-1a over the chunk's own bs token ids seeded by the parent. A longer prompt reuses
  identical early chunks. `load_prefix` walks from the root, stopping at the first
  missing/corrupt file, returning the matched `kv_chain_chunk{attn_blob,recr_blob}` list
  (in order) + `*n_tokens = n_chunks*bs`.
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
  each ubatch commits (the server sees the whole prompt as ONE llama_decode call, so the
  hook had to go inside it).
- `kv_chain_save_prefill_ubatch` (server-context.cpp:906): arms during prefill (armed in
  SLOT_STATE_STARTED, cleared at DONE_PROMPT->GENERATING - decode-generated state is
  worthless to cache). Reads pos from `llama_memory_seq_pos_max` (authoritative; the
  context-side ubatch.pos / seq_pos_max inside the callback were stale). Dumps a chunk ONLY
  when `pos % bs == 0` (on-grid): the runtime ubatch split is "chaotic" near the tail / for
  short prompts, so off-grid edges are skipped ("stop at the edge of the happy flow"). No
  dedup state needed (pos is strictly increasing). Calls `kv_chain->save(ctx, seq,
  pos_lo=chunk_lo, pos_hi=pos, chunk_hash, chunk_tokens)`.

## 6. Restore path (server-context.cpp, SLOT_STATE_STARTED, n_past==0, slot empty, cache_prompt)

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
  (default 100, devops/env.sh). Eviction is LRU by mtime after each write; a file vanishing
  mid-restore is a benign "cache ends here". On startup, stray `.tmp` files in the root_hash
  dir are removed.

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
  and restart (cached_tokens:288, 9 chunks) BOTH reproduce 6/6 distinctive phrases ->
  proves the full chain's attn+recurrent KV restores correctly.

## 9. Dev tooling (devops/)

- env.sh (model, port 50081, paths), llama_build.sh (cmake+ninja, Vulkan; check EXIT=$?,
  `| tail -1` masks the exit code), llama_run.sh (--keep-cache/--no-kv-chain/-- <args>),
  llama_kill.sh, llama_wait.sh, llama_prompt.sh (sends /v1/completions cache_prompt:true,
  prints `cached_tokens: N prompt_tokens: M` then the full text), llama_test.sh (--flush,
  [restart] markers, trailing `-- <extra run args>`; the `--` tail must come LAST),
  llama_unittest_1.sh (the full-chain fidelity test above).

## 10. Key constraints / gotchas

- Gated DeltaNet recurrent state is NON-INVERTIBLE: resume only FROM a boundary, never roll
  back to m<n. Never "strip the last token" from a chunk.
- `-ub` has a lower bound of 32 (`-ub 8` is silently ignored -> 2048). Use `-ub 32` for tests.
- `chunk_size` (--ubatch) is a BOUNDARY STRIDE, not the runtime prefill batch size. The hybrid
  `split_equal` (llama-memory-hybrid.cpp:89) divides n_ubatch across n_seqs and is chaotic near
  the tail / for short prompts. We only dump when `pos % bs == 0`; do not try to make chunk
  edges match the runtime split.
- Each chunk is ~152 MiB at -ub 32 (constant). A 1000-tok prompt (~30 chunks) is ~4.5 GiB.
  `KV_CHAIN_LIMIT_GB` default 100.
