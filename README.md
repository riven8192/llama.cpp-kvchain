# Streaming kv-cache to disk for predictable cache-hits

The current caching strategy is llama.cpp is fully in VRAM and/or RAM. This means that when
resources are limited, valuable kv-caches are evicted. Small prompts can and will evict
large caches. Especially on local hardware this leads to the rebulding of kv-cache that may
take a significant amount of time, where streaming kv caches to disk would lead to guaranteed
cache hits.


## How it works

Every prompt is split into chunks, with the size of the llama.cpp `--ubatch` parameter. Each
chunk will be prefilled, and the resulting kv-cache of that region will be written to disk.
When multiple prompts share a common prefix, they will share these chunk-sized kv-cache files.

When the next prompt is presented, it will do the same split, find out which usable kv-cache
files are already on disk, load them into VRAM and prefill the remaining tokens.


## How to use

### Building from source

```bash
cmake -S . -B ./build-directory -G Ninja \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON \
  -DLLAMA_CURL=ON \
  -DLLAMA_OPENSSL=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_BORINGSSL=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF

cmake --build ./build-directory
```

### Running llama-server

Parameter `--kv-chain-dir` enables the kv-cache streaming.
Parameter `--kv-chain-limit-gb` determines how much disk space will at most be allocated.

```bash
./build-directory/llama-server \
  -hf <<your model>>
  --kv-chain-dir ./kv-cache-dir
  --kv-chain-limit-gb 100
  --cache-ram 0 \
  --no-cache-idle-slots \
  --fit off \
  --no-mmproj \
  --flash-attn on \
  --jinja
```



## Tested models

- unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_XL
- unsloth/DeepSeek-V4-Flash-0731-GGUF:UD-IQ3_S
- unsloth/MiniMax-M2.7-GGUF:UD-Q3_K_XL
- unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q8_K_XL


## Tested hardware

- AMD Strix Halo - 128 GB (shared memory)
- Linux 6.18.12-200.fc43.x86_64
- On the Vulkan backend, with --parallel `1` & `2`
