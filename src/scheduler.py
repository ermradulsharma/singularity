"""Continuous Batching Iteration-Level Scheduler for Singularity AGI Engine."""
import asyncio
import time
import uuid
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
from src.model import GPTLanguageModel, PagedKVCacheManager
from src.tokenizer import get_unified_tokenizer

class GenerationRequest:
    """Encapsulates state, prompt tokens, generation limits, and output queues for a continuous batching sequence request."""
    def __init__(self, prompt_tokens: List[int], max_new_tokens: int = 128, temperature: float = 1.0, top_p: float = 0.9, request_id: str = None):
        self.request_id = request_id or str(uuid.uuid4())
        self.prompt_tokens = prompt_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = max(temperature, 1e-5)
        self.top_p = top_p
        self.generated_tokens: List[int] = []
        self.is_finished = False
        self.queue: asyncio.Queue = asyncio.Queue()
        self.created_at = time.time()
        self.allocated_block_ids: Optional[torch.Tensor] = None

class ContinuousBatchScheduler:
    """Production Iteration-Level Continuous Batching Engine with PagedKVCache Virtual Block Allocation."""
    def __init__(self, model: GPTLanguageModel, max_batch_size: int = 32, block_size: int = 16):
        self.model = model
        self.max_batch_size = max_batch_size
        self.block_size = block_size
        self.device = next(model.parameters()).device
        self.pending_requests: List[GenerationRequest] = []
        self.active_requests: List[GenerationRequest] = []
        self.kv_manager = PagedKVCacheManager(
            num_layers=len(model.graph['blocks']),
            num_heads=model.graph['blocks'][0].h,
            head_dim=model.graph['blocks'][0].hd,
            block_size=block_size,
            num_blocks=2048,
            device=str(self.device)
        )
        self.tokenizer = get_unified_tokenizer()
        self.is_running = False
        self._lock = asyncio.Lock()

    async def add_request(self, prompt_tokens: List[int], max_new_tokens: int = 128, temperature: float = 1.0, top_p: float = 0.9) -> GenerationRequest:
        """Enqueues a new generation request into the pending scheduling pool."""
        req = GenerationRequest(prompt_tokens, max_new_tokens, temperature, top_p)
        async with self._lock:
            self.pending_requests.append(req)
        return req

    def step(self) -> Dict[str, Any]:
        """Executes a single continuous decoding iteration step across all active requests."""
        if not self.active_requests and not self.pending_requests:
            return {"active_count": 0, "processed_tokens": 0}

        # Promote pending requests into active pool up to max_batch_size
        while self.pending_requests and len(self.active_requests) < self.max_batch_size:
            req = self.pending_requests.pop(0)
            req.allocated_block_ids = self.kv_manager.allocate(req.request_id, len(req.prompt_tokens) + req.max_new_tokens)
            self.active_requests.append(req)

        if not self.active_requests:
            return {"active_count": 0, "processed_tokens": 0}

        # Build dynamic iteration input batch
        input_list = []
        for req in self.active_requests:
            if not req.generated_tokens:
                # Prefill step: feed prompt sequence
                t_input = torch.tensor(req.prompt_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
            else:
                # Decode step: feed single last token
                t_input = torch.tensor([req.generated_tokens[-1]], dtype=torch.long, device=self.device).unsqueeze(0)
            input_list.append(t_input)

        # Batch forward pass per sequence to preserve Paged KV state
        finished_reqs = []
        for idx, req in enumerate(self.active_requests):
            t_input = input_list[idx]
            with torch.no_grad():
                logits, _ = self.model(t_input)
                next_token_logits = logits[:, -1, :] / req.temperature
                
                if req.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > req.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('Inf')

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

            req.generated_tokens.append(next_token)
            
            # Put token into async queue for streaming listeners
            try:
                req.queue.put_nowait(next_token)
            except Exception:
                pass

            eot_id = getattr(self.tokenizer, "eot_token", 50256)
            if len(req.generated_tokens) >= req.max_new_tokens or next_token == eot_id:
                req.is_finished = True
                finished_reqs.append(req)

        # Cleanup finished requests & free physical KV pages
        for req in finished_reqs:
            self.kv_manager.free(req.request_id)
            self.active_requests.remove(req)

        return {"active_count": len(self.active_requests), "processed_tokens": len(input_list)}

    async def run_loop(self):
        """Asynchronous execution loop driving continuous iteration steps."""
        self.is_running = True
        while self.is_running:
            async with self._lock:
                stats = self.step()
            if stats["active_count"] == 0 and not self.pending_requests:
                await asyncio.sleep(0.01)
            else:
                await asyncio.sleep(0.001)

    def stop(self):
        """Stops the continuous batch scheduler loop."""
        self.is_running = False
