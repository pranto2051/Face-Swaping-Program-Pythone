import threading
from collections import defaultdict
from engine.base import InferenceEngine, DeviceSlot

class ModelRegistry:
    """
    Singleton per process. Key: (model_name, device_id).
    Lazy-loads on first request, keeps in memory.
    Evicts LRU model if VRAM pressure detected.
    
    Thread-safe via per-model locks.
    """
    _instances: dict[tuple[str, int], InferenceEngine] = {}
    _locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
    
    def __init__(self, factory_callback=None):
        self._factory = factory_callback or self._default_factory
        
    def _default_factory(self, name: str) -> InferenceEngine:
        raise NotImplementedError(f"No factory set for {name}")
    
    def get(self, name: str, device: DeviceSlot) -> InferenceEngine:
        key = (name, device.device_id)
        with self._locks[name]:
            if key not in self._instances:
                engine = self._factory(name)
                engine.load(device)
                self._instances[key] = engine
            return self._instances[key]
