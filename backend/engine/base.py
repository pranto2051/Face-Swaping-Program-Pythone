from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class DeviceType(Enum):
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"

@dataclass
class DeviceSlot:
    device_id: int
    device_type: DeviceType
    vram_total_mb: int
    vram_available_mb: int
    temperature_c: int
    active_jobs: int

class InferenceEngine(ABC):
    """
    All model wrappers inherit this.
    Enforces: load once, pin to device, memory-safe inference.
    """
    @abstractmethod
    def load(self, device: DeviceSlot) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def infer(self, *args, **kwargs): ...

    @abstractmethod
    def vram_estimate_mb(self) -> int:
        """Return estimated VRAM for this model at fp16."""
        ...
