import time
from typing import Dict, List, Tuple, Union

import torch

KVLayerCache = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


class KVOffloadManager:
    def __init__(self, pin_memory: bool = True):
        self.pin_memory = pin_memory
        self.cpu_cache: Dict[int, List[KVLayerCache]] = {}
        self.stats = {
            "num_offload_events": 0,
            "num_restore_events": 0,
            "offload_time_sec": 0.0,
            "restore_time_sec": 0.0,
            "offloaded_bytes": 0,
            "restored_bytes": 0,
        }
        self._events = []

    def _get_cache_device(self, gpu_cache: List[KVLayerCache]) -> torch.device:
        first_layer = gpu_cache[0]
        if isinstance(first_layer, tuple):
            return first_layer[0].device
        return first_layer.device

    def _copy_blocks_to_cpu(
        self,
        tensor: torch.Tensor,
        block_idx: torch.Tensor,
    ) -> torch.Tensor:
        selected = tensor.index_select(0, block_idx)

        if not self.pin_memory:
            return selected.cpu()

        cpu_tensor = torch.empty(
            selected.shape,
            dtype=selected.dtype,
            device="cpu",
            pin_memory=True,
        )
        cpu_tensor.copy_(selected, non_blocking=True)
        return cpu_tensor

    def _tensor_nbytes(self, tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()

    def offload(self, seq_id: int, gpu_cache: List[KVLayerCache], block_ids: List[int]) -> None:
        start = time.monotonic()
        device = self._get_cache_device(gpu_cache)
        block_idx = torch.tensor(block_ids, dtype=torch.long, device=device)

        per_layer_cpu = []
        total_bytes = 0

        for layer_cache in gpu_cache:
            if isinstance(layer_cache, tuple):
                k_cache, v_cache = layer_cache

                cpu_k = self._copy_blocks_to_cpu(k_cache, block_idx)
                cpu_v = self._copy_blocks_to_cpu(v_cache, block_idx)

                total_bytes += self._tensor_nbytes(cpu_k)
                total_bytes += self._tensor_nbytes(cpu_v)

                per_layer_cpu.append((cpu_k, cpu_v))
            else:
                cpu_layer = self._copy_blocks_to_cpu(layer_cache, block_idx)
                total_bytes += self._tensor_nbytes(cpu_layer)
                per_layer_cpu.append(cpu_layer)

        torch.cuda.synchronize(device)

        self.cpu_cache[seq_id] = per_layer_cpu

        elapsed = time.monotonic() - start
        self.stats["num_offload_events"] += 1
        self.stats["offload_time_sec"] += elapsed
        self.stats["offloaded_bytes"] += total_bytes

        self._events.append(
            {
                "event_type": "offload",
                "seq_id": seq_id,
                "num_blocks": len(block_ids),
                "num_bytes": total_bytes,
                "elapsed_sec": elapsed,
                "cached_offloaded_seqs": len(self.cpu_cache),
                "timestamp": time.monotonic(),
            }
        )

    def restore(self, seq_id: int, gpu_cache: List[KVLayerCache], new_block_ids: List[int]) -> None:
        start = time.monotonic()
        assert seq_id in self.cpu_cache, f"No CPU KV cache for seq {seq_id}"

        device = self._get_cache_device(gpu_cache)
        block_idx = torch.tensor(new_block_ids, dtype=torch.long, device=device)
        per_layer_cpu = self.cpu_cache.pop(seq_id)

        total_bytes = 0

        for layer_cache, cpu_layer in zip(gpu_cache, per_layer_cpu):
            if isinstance(layer_cache, tuple):
                k_cache, v_cache = layer_cache
                cpu_k, cpu_v = cpu_layer

                total_bytes += self._tensor_nbytes(cpu_k)
                total_bytes += self._tensor_nbytes(cpu_v)

                k_cache.index_copy_(0, block_idx, cpu_k.to(device, non_blocking=True))
                v_cache.index_copy_(0, block_idx, cpu_v.to(device, non_blocking=True))
            else:
                total_bytes += self._tensor_nbytes(cpu_layer)
                layer_cache.index_copy_(
                    0,
                    block_idx,
                    cpu_layer.to(device, non_blocking=True),
                )

        torch.cuda.synchronize(device)

        elapsed = time.monotonic() - start
        self.stats["num_restore_events"] += 1
        self.stats["restore_time_sec"] += elapsed
        self.stats["restored_bytes"] += total_bytes

        self._events.append(
            {
                "event_type": "restore",
                "seq_id": seq_id,
                "num_blocks": len(new_block_ids),
                "num_bytes": total_bytes,
                "elapsed_sec": elapsed,
                "cached_offloaded_seqs": len(self.cpu_cache),
                "timestamp": time.monotonic(),
            }
        )

    def drain_events(self):
        events = self._events
        self._events = []
        return events

    def get_stats(self):
        stats = dict(self.stats)
        stats["cached_offloaded_seqs"] = len(self.cpu_cache)
        return stats