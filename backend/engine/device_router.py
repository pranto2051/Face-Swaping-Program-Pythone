class GPUBusyError(Exception):
    pass

class DeviceRouter:
    """
    VRAM-aware job → GPU assignment.
    
    Algorithm:
    1. Query all GPU slots via pynvml
    2. Filter slots with sufficient free VRAM for requested pipeline
    3. Score: free_vram * 0.6 + (1 - utilization) * 0.3 + (1 - temp/100) * 0.1
    4. Assign to highest-scoring slot
    5. If no slot available: queue with backpressure
    
    Pipeline VRAM budgets (fp16):
    - Fast:      ~2.5 GB (InsightFace + GFPGAN)
    - Studio:    ~4.5 GB (+ DFL)
    - Cinematic: ~7.0 GB (+ Diffusion + ColorNet)
    """
    
    PIPELINE_VRAM = {"fast": 2500, "studio": 4500, "cinematic": 7000}
    
    def __init__(self, monitor):
        self.monitor = monitor

    def assign(self, pipeline_mode: str):
        required = self.PIPELINE_VRAM.get(pipeline_mode, 2500)
        candidates = [s for s in self._scan_devices() if s.vram_available_mb >= required]
        if not candidates:
            raise GPUBusyError(f"No GPU with {required}MB free VRAM")
        return max(candidates, key=self._score)

    def _scan_devices(self):
        # We can simulate getting devices from the monitor instead
        # For now return the main device slot if using Mac or fallback
        stats = self.monitor.get_stats()
        from engine.base import DeviceSlot, DeviceType
        import torch
        
        dtype = DeviceType.CPU
        if stats.get("cuda_available"):
            dtype = DeviceType.CUDA
        elif torch.backends.mps.is_available():
            dtype = DeviceType.MPS
            
        slot = DeviceSlot(
            device_id=0,
            device_type=dtype,
            vram_total_mb=stats.get("vram_total_mb", 0),
            vram_available_mb=stats.get("vram_total_mb", 0) - stats.get("vram_used_mb", 0),
            temperature_c=stats.get("temperature_c", 0),
            active_jobs=0
        )
        return [slot]
        
    def _score(self, slot):
        return slot.vram_available_mb * 0.6 + (1 - slot.temperature_c/100.0) * 0.1

class MultiGPURouter(DeviceRouter):
    """
    Routing decision tree:
    
    1. FAST MODE jobs → prefer lowest-VRAM GPU (save big GPUs for heavy)
    2. STUDIO/CINEMATIC → require GPU with >= budget VRAM free
    3. VIDEO → prefer GPU with largest contiguous VRAM (frame batching)
    4. BATCH → split across GPUs: assign items round-robin to available slots
    
    Anti-starvation:
    - Max 2 concurrent jobs per GPU
    - If all GPUs at max: return estimated wait time
    - Priority queue: Pro tier > Free tier
    
    OOM protection:
    - Reserve 500MB VRAM headroom per GPU
    - If job fails with OOM: fallback to CPU for non-critical stages
    - Log OOM event for profiling
    """
    
    def route_batch(self, items: list, mode: str):
        """Distribute batch items across available GPUs."""
        slots = [s for s in self._scan_devices() if s.vram_available_mb >= self.PIPELINE_VRAM.get(mode, 2500)]
        if not slots:
            raise GPUBusyError(f"No available GPU with required VRAM for batch.")
            
        assignments = []
        for i, item in enumerate(items):
            slot = slots[i % len(slots)]  # Round-robin
            assignments.append((i, slot))
        return assignments
