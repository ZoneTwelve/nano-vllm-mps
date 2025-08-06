# FILE: nanovllm/layers/attention.py
import torch
from torch import nn
import torch.nn.functional as F
from nanovllm.utils.context import get_context
from nanovllm.utils.logging import logger
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
        is_warmup = self.k_cache.numel() == 0

        # --- CUDA Path (Optimized) ---
        if use_cuda_path and not is_warmup:
            k_cache, v_cache = self.k_cache, self.v_cache
            if context.slot_mapping is not None and context.slot_mapping.numel() > 0:
                store_kvcache(k.view(-1, self.num_kv_heads, self.head_dim), v.view(-1, self.num_kv_heads, self.head_dim), k_cache, v_cache, context.slot_mapping)
            
            q = q.view(-1, self.num_heads, self.head_dim)
            if context.is_prefill:
                k_to_use = k_cache if context.block_tables is not None else k.view(-1, self.num_kv_heads, self.head_dim)
                v_to_use = v_cache if context.block_tables is not None else v.view(-1, self.num_kv_heads, self.head_dim)
                o = flash_attn_varlen_func(q, k_to_use, v_to_use,
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
            logger.debug(f"Entering MPS/CPU fallback. is_warmup={is_warmup}, is_prefill={context.is_prefill}")
            q = q.view(-1, self.num_heads, self.head_dim)
            
            # This path handles all non-flash-attn cases by manually implementing attention.
            # This is to bypass suspected bugs in the MPS implementation of F.scaled_dot_product_attention.
            outputs = []
            num_seqs = len(context.cu_seqlens_q) - 1 if context.is_prefill else q.shape[0]

            for i in range(num_seqs):
                # 1. Get the Q, K, V for the current sequence
                if is_warmup or (context.is_prefill and context.block_tables is None):
                    q_start, q_end = context.cu_seqlens_q[i], context.cu_seqlens_q[i+1]
                    k_start, k_end = context.cu_seqlens_k[i], context.cu_seqlens_k[i+1]
                    q_seq = q[q_start:q_end]
                    k_seq = k.view(-1, self.num_kv_heads, self.head_dim)[k_start:k_end]
                    v_seq = v.view(-1, self.num_kv_heads, self.head_dim)[k_start:k_end]
                else: # Paged attention
                    if context.is_prefill:
                        q_seq = q[context.cu_seqlens_q[i]:context.cu_seqlens_q[i+1]]
                        k_len = (context.cu_seqlens_k[i+1] - context.cu_seqlens_k[i]).item()
                    else: # decode
                        q_seq = q[i:i+1]
                        k_len = context.context_lens[i].item()
                    
                    block_table = context.block_tables[i]
                    k_cache_flat = self.k_cache.view(-1, self.num_kv_heads, self.head_dim)
                    v_cache_flat = self.v_cache.view(-1, self.num_kv_heads, self.head_dim)
                    
                    k_seq = torch.empty(k_len, self.num_kv_heads, self.head_dim, dtype=q.dtype, device=q.device)
                    v_seq = torch.empty(k_len, self.num_kv_heads, self.head_dim, dtype=q.dtype, device=q.device)
                    
                    for token_idx in range(k_len):
                        block_idx = block_table[token_idx // context.block_size].item()
                        block_offset = token_idx % context.block_size
                        slot_idx = block_idx * context.block_size + block_offset
                        k_seq[token_idx] = k_cache_flat[slot_idx]
                        v_seq[token_idx] = v_cache_flat[slot_idx]

                # 2. Handle GQA by repeating K and V heads
                if self.num_heads != self.num_kv_heads:
                    k_seq = k_seq.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
                    v_seq = v_seq.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

                # 3. Manually compute scaled dot-product attention
                q_len, k_len = q_seq.shape[0], k_seq.shape[0]
                q_seq = q_seq.transpose(0, 1) # [num_heads, q_len, head_dim]
                k_seq = k_seq.transpose(0, 1) # [num_heads, k_len, head_dim]
                v_seq = v_seq.transpose(0, 1) # [num_heads, k_len, head_dim]

                logger.debug(f"Seq {i} shapes for matmul: q_seq={q_seq.shape}, k_seq={k_seq.shape}")
                attn_weights = torch.matmul(q_seq, k_seq.transpose(-1, -2)) * self.scale
                
                # Apply causal mask
                if q_len == k_len:
                    mask = torch.triu(torch.ones(q_len, k_len, dtype=torch.bool, device=q.device), diagonal=1)
                    attn_weights.masked_fill_(mask, -torch.inf)
                elif q_len < k_len: # Prefill with prefix
                    prefix_len = k_len - q_len
                    mask = torch.ones(q_len, k_len, dtype=torch.bool, device=q.device)
                    mask[:, :prefix_len] = False
                    mask[:, prefix_len:] = torch.triu(mask[:, prefix_len:], diagonal=1)
                    attn_weights.masked_fill_(mask, -torch.inf)

                attn_weights = F.softmax(attn_weights, dim=-1)
                o_seq = torch.matmul(attn_weights, v_seq)
                o_seq = o_seq.transpose(0, 1).contiguous()
                
                outputs.append(o_seq)
            
            o = torch.cat(outputs, dim=0) if outputs else torch.empty_like(q)

        return o.view(-1, self.num_heads * self.head_dim)