# FILE: nanovllm/engine/model_runner.py
import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
from typing import Union, List

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Union[Event, List[Event]]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.device = config.device
        self.is_cuda = self.device == "cuda"

        if self.world_size > 1:
            dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        
        if self.is_cuda:
            torch.cuda.set_device(rank)
        
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device(self.device)
        
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        
        if self.is_cuda and not self.enforce_eager:
            self.capture_cudagraph()
        
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        
        if self.is_cuda and not self.enforce_eager:
            del self.graphs, self.graph_pool
        
        if self.is_cuda:
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()
            
        if self.world_size > 1:
            dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and not self.rank
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        if self.is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        elif self.device == "mps":
            torch.mps.empty_cache()

        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(self.config.max_num_seqs, 4)
        warmup_len = min(max_model_len, 1024)
        seqs = [Sequence([0] * warmup_len) for _ in range(num_seqs)]
        
        self.run(seqs, True)
        
        if self.is_cuda:
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        
        # For CUDA, we calculate the number of blocks dynamically.
        if self.is_cuda and config.num_kvcache_blocks == -1:
            free, total = torch.cuda.mem_get_info()
            used = total - free
            peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
            current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
            available_mem = total * config.gpu_memory_utilization - used - peak + current
            
            num_kv_heads = hf_config.num_key_value_heads // self.world_size
            block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize
            
            config.num_kvcache_blocks = int(available_mem) // block_bytes
        
        # For MPS, a default value is set in the Config.
        # This assertion will now catch if the default is too large for the system,
        # or if the user provides an invalid number.
        assert config.num_kvcache_blocks > 0, (
            f"Insufficient GPU memory to allocate KV cache. "
            f"The number of KV cache blocks is configured to {config.num_kvcache_blocks}, which is not possible. "
            "If you are on an MPS device, the default may be too high for your system. "
            "Try explicitly setting a smaller `num_kvcache_blocks` when creating the LLM object."
        )

        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        self.kv_cache = torch.zeros(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, hf_config.head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs) if seqs else 0
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=self.is_cuda).to(self.device, non_blocking=self.is_cuda)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)
        
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=self.is_cuda).to(self.device, non_blocking=self.is_cuda)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=self.is_cuda).to(self.device, non_blocking=self.is_cuda)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=self.is_cuda).to(self.device, non_blocking=self.is_cuda)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=self.is_cuda).to(self.device, non_blocking=self.is_cuda)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=self.is_cuda).to(self.device, non_blocking=self.is_cuda)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables, block_size=self.block_size)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq))
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=self.is_cuda).to(self.device, non_blocking=self.is_cuda)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=self.is_cuda).to(self.device, non_blocking=self.is_cuda)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=self.is_cuda).to(self.device, non_blocking=self.is_cuda)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=self.is_cuda).to(self.device, non_blocking=self.is_cuda)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables, block_size=self.block_size)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=self.is_cuda).to(self.device, non_blocking=self.is_cuda)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or not self.is_cuda or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs], block_size=self.block_size)
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )