# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Manual Download

If you prefer to download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |

## MPS Implementation

### 1. Python and Environment Compatibility

*   **Problem:** The original code used modern Python 3.10+ features that are not backward-compatible.
*   **Fixes:**
    *   **Type Hinting:** Replaced all instances of the `|` operator in type hints (e.g., `dict | None`) with the older, compatible `typing.Union` and `typing.Optional` syntax across all files.
    *   **Conditional Imports:** The `triton` and `flash-attn` libraries are CUDA-only. The initial unconditional `import` statements caused a `ModuleNotFoundError` on MPS. This was fixed by wrapping these imports in a conditional block that only runs if a CUDA device is detected.

### 2. Tensor Parallelism and Distributed Code

*   **Problem:** The model's layers were designed for tensor parallelism and unconditionally called `torch.distributed` functions (like `dist.get_rank()`), which are not initialized in a single-device environment like MPS. This caused a `ValueError`.
*   **Fix:** All calls to `torch.distributed` functions were wrapped in an `if dist.is_initialized():` check. In the `else` block, the code now assumes a default tensor parallel size of 1 and a rank of 0, allowing the same code to run seamlessly in both single-device and distributed modes.

### 3. Memory Management and KV Cache Allocation

*   **Problem:** The initial attempt to calculate available memory on MPS was unreliable and led to an `AssertionError` because it calculated negative available memory. The `torch.mps` API does not provide the same memory management utilities as the CUDA API.
*   **Fix:** The fragile automatic memory calculation for MPS was completely removed. Instead, the logic was updated to:
    1.  Detect the `mps` backend in `nanovllm/config.py`.
    2.  If the user has not specified a value for `num_kvcache_blocks`, set a safe, conservative default (e.g., 1024).
    3.  This makes memory allocation on MPS predictable and stable, while still allowing advanced users to override the default. The informative `AssertionError` message was kept to guide users if they set an invalid number.

### 4. Core Attention Logic (The MPS Fallback)

This was the most complex part of the implementation and required several iterations.

*   **Problem 1: `IndexError` during Warmup:** The MPS fallback logic initially tried to access the shape of the KV cache during the model's warmup run, but the cache had not been allocated yet, causing a crash.
*   **Problem 2: `TypeError` during Prefill:** The logic did not correctly handle the initial prefill step where `block_tables` is `None`, leading to a `TypeError`.
*   **Problem 3: `RuntimeError` during Decode (Head Mismatch):** The final and most subtle bug. Even with correct tensor shapes, the `F.scaled_dot_product_attention` function itself was failing on the MPS backend specifically during the decode step (when query length is 1 and key length is >1). This indicated a bug within the PyTorch MPS implementation.
*   **Final Solution:**
    1.  The fallback attention logic in `nanovllm/layers/attention.py` was completely rewritten to be more robust.
    2.  It now correctly handles three distinct cases: the warmup run, the initial prefill (without a cache history), and the paged attention steps (decode or prefill with a cache history).
    3.  To bypass the suspected bug in PyTorch's fused kernel, the call to `F.scaled_dot_product_attention` was **replaced with a manual implementation** using fundamental, stable PyTorch operations (`torch.matmul`, `F.softmax`, and manual masking). This is slightly less performant but guarantees correctness on the MPS backend.

### 5. Developer Experience and Debugging

*   **Problem:** As debugging became more complex, there was no easy way to inspect the internal state of the program.
*   **Solution:** A centralized logger was implemented in a new file, `nanovllm/utils/logging.py`.
    *   This allows for consistent logging throughout the library.
    *   The verbosity can be controlled by setting the `NANOVLLM_LOG_LEVEL` environment variable (e.g., `export NANOVLLM_LOG_LEVEL=DEBUG`).
    *   Detailed debug messages were added to key areas, especially in the attention layer, to print tensor shapes and logic paths, which was instrumental in solving the final bug.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)