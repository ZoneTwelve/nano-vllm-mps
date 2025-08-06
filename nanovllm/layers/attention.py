# FILE: nanovllm/layers/attention.py
import torch
from torch import nn
import torch.nn.functional as F
from nanovllm.utils.context import get_context
from typing import Optional

# Conditionally import CUDA-specific libraries
_IS_CUDA_AVAILABLE = torch.cuda.is_available()
if _IS_CUDA_AVAILABLE:
    try:
        import triton
        import triton.language as tl
        from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    except ImportError:
        _IS_CUDA_AVAILABLE = False

# Define Triton kernel only if CUDA and Triton are available
if _IS_CUDA_AVAILABLE:
    @triton.jit
    def store_kvcache_kernel(
        key_ptr, key_stride, value_ptr, value_stride,
        k_cache_ptr, v_cache_ptr, slot_mapping_ptr, D: tl.constexpr,
    ):
        idx = tl.program_id(0)
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)
        slot = tl.load(slot_mapping_ptr + idx)
        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)

    def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
        N, num_heads, head_dim = key.shape
        D = num_heads * head_dim
        store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)

# PyTorch fallback for storing KV cache, used on MPS/CPU
def store_kvcache_pytorch(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    D = key.shape[1] * key.shape[2]
    key = key.view(-1, D)
    value = value.view(-1, D)
    k_cache.view(-1, D)[slot_mapping] = key
    v_cache.view(-1, D)[slot_mapping] = value


class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        use_cuda_path = _IS_CUDA_AVAILABLE and q.device.type == 'cuda'

        # This is the main check. If the cache is empty, it's the warmup run.
        is_warmup = self.k_cache.numel() == 0

        # --- CUDA Path (Optimized) ---
        if use_cuda_path and not is_warmup:
            k_cache, v_cache = self.k_cache, self.v_cache
            if context.slot_mapping is not None and context.slot_mapping.numel() > 0:
                store_kvcache(k.view(-1, self.num_kv_heads, self.head_dim), v.view(-1, self.num_kv_heads, self.head_dim), k_cache, v_cache, context.slot_mapping)
            
            q = q.view(-1, self.num_heads, self.head_dim)
            if context.is_prefill:
                o = flash_attn_varlen_func(q, k_cache, v_cache,
                                           cu_seqlens_q=context.cu_seqlens_q,
                                           cu_seqlens_k=context.cu_seqlens_k,
                                           max_seqlen_q=context.max_seqlen_q,
                                           max_seqlen_k=context.max_seqlen_k,
                                           softmax_scale=self.scale, causal=True, block_table=context.block_tables)
            else: # decode
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                            softmax_scale=self.scale, causal=True)
        # --- MPS / CPU / CUDA-Warmup Path (Fallback) ---
        else:
            q = q.view(-1, self.num_heads, self.head_dim)
            
            # Scenario 1: Warmup run (cache is empty). Perform standard attention.
            if is_warmup:
                k = k.view(-1, self.num_kv_heads, self.head_dim)
                v = v.view(-1, self.num_kv_heads, self.head_dim)
                # This logic handles batched attention for the warmup sequences
                outputs = []
                for i in range(len(context.cu_seqlens_q) - 1):
                    q_seq = q[context.cu_seqlens_q[i]:context.cu_seqlens_q[i+1]]
                    k_seq = k[context.cu_seqlens_k[i]:context.cu_seqlens_k[i+1]]
                    v_seq = v[context.cu_seqlens_k[i]:context.cu_seqlens_k[i+1]]

                    if self.num_heads != self.num_kv_heads:
                        k_seq = k_seq.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
                        v_seq = v_seq.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
                    
                    o_seq = F.scaled_dot_product_attention(q_seq.unsqueeze(0), k_seq.unsqueeze(0), v_seq.unsqueeze(0), scale=self.scale, is_causal=True)
                    outputs.append(o_seq.squeeze(0))
                o = torch.cat(outputs, dim=0) if outputs else torch.empty_like(q)

            # Scenario 2: Normal run (cache exists). Perform paged attention.
            else:
                k_cache, v_cache = self.k_cache, self.v_cache
                if context.slot_mapping is not None and context.slot_mapping.numel() > 0:
                    store_kvcache_pytorch(k.view(-1, self.num_kv_heads, self.head_dim), v.view(-1, self.num_kv_heads, self.head_dim), k_cache, v_cache, context.slot_mapping)

                outputs = []
                num_seqs = len(context.cu_seqlens_q) - 1 if context.is_prefill else q.shape[0]
                block_size = context.block_size
                assert block_size > 0, "Block size must be set in context for paged attention."

                k_cache_flat = k_cache.view(-1, self.num_kv_heads, self.head_dim)
                v_cache_flat = v_cache.view(-1, self.num_kv_heads, self.head_dim)

                for i in range(num_seqs):
                    if context.is_prefill:
                        q_seq = q[context.cu_seqlens_q[i]:context.cu_seqlens_q[i+1]]
                        k_len = (context.cu_seqlens_k[i+1] - context.cu_seqlens_k[i]).item()
                        block_table = context.block_tables[i]
                    else: # decode
                        q_seq = q[i:i+1]
                        k_len = context.context_lens[i].item()
                        block_table = context.block_tables[i]

                    if k_len == 0:
                        outputs.append(torch.zeros_like(q_seq))
                        continue

                    k_seq = torch.empty(k_len, self.num_kv_heads, self.head_dim, dtype=q.dtype, device=q.device)
                    v_seq = torch.empty(k_len, self.num_kv_heads, self.head_dim, dtype=q.dtype, device=q.device)

                    for token_idx in range(k_len):
                        block_idx = block_table[token_idx // block_size].item()
                        block_offset = token_idx % block_size
                        slot_idx = block_idx * block_size + block_offset
                        k_seq[token_idx] = k_cache_flat[slot_idx]
                        v_seq[token_idx] = v_cache_flat[slot_idx]

                    if self.num_heads != self.num_kv_heads:
                        k_seq = k_seq.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
                        v_seq = v_seq.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

                    attn_mask: Optional[torch.Tensor] = None
                    q_len, is_causal = q_seq.shape[0], True
                    if context.is_prefill and q_len != k_len:
                        is_causal = False
                        prefix_len = k_len - q_len
                        attn_mask = torch.ones(q_len, k_len, dtype=torch.bool, device=q.device)
                        attn_mask[:, :prefix_len] = False
                        attn_mask[:, prefix_len:] = torch.triu(attn_mask[:, prefix_len:], diagonal=1)
                    
                    o_seq = F.scaled_dot_product_attention(q_seq.unsqueeze(0), k_seq.unsqueeze(0), v_seq.unsqueeze(0), attn_mask=attn_mask, scale=self.scale, is_causal=is_causal)
                    outputs.append(o_seq.squeeze(0))
                
                o = torch.cat(outputs, dim=0) if outputs else torch.empty_like(q)

        return o.view(-1, self.num_heads * self.head_dim)