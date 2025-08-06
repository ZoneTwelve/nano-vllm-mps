# FILE: nanovllm/config.py
import os
from dataclasses import dataclass
from transformers import AutoConfig
import torch
from typing import Optional
from nanovllm.utils.logging import logger


@dataclass
class Config:
    model: str
    device: str = "auto"
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: Optional[AutoConfig] = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        logger.debug(f"Initial config: {self}")
        # assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8

        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info("Auto-detected CUDA backend.")
            elif torch.backends.mps.is_available():
                self.device = "mps"
                logger.info("Auto-detected MPS backend.")
            else:
                self.device = "cpu"
                logger.info("Auto-detected CPU backend.")
        
        if self.device == "mps":
            logger.info("MPS backend in use. Forcing tensor_parallel_size=1 and enforce_eager=True.")
            self.tensor_parallel_size = 1
            self.enforce_eager = True
            if self.num_kvcache_blocks == -1:
                self.num_kvcache_blocks = 128
                logger.info(f"Setting default num_kvcache_blocks={self.num_kvcache_blocks} for MPS.")

        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
        logger.debug(f"Final config: {self}")
