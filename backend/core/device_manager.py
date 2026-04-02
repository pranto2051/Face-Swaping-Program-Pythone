"""
Apple Silicon (MPS) Device Management for M1/M2 Pro MacBooks
Provides optimal device selection, memory management, and performance tuning
for Metal Performance Shaders (MPS) backend.
"""

import torch
import psutil
import numpy as np
from typing import Tuple, Optional
import os


class MPSDeviceManager:
    """Manages PyTorch device selection and optimization for Apple Silicon."""
    
    def __init__(self):
        self.device = self._get_optimal_device()
        self.device_name = self._get_device_name()
        self.memory_info = self._get_memory_info()
        self.supports_float16 = self._check_float16_support()
        
    def _get_optimal_device(self) -> torch.device:
        """
        Detect and return optimal device for Apple Silicon.
        Priority: MPS > CPU
        """
        try:
            if torch.backends.mps.is_available():
                # Verify MPS can actually be used
                if torch.backends.mps.is_built():
                    return torch.device("mps")
        except Exception as e:
            print(f"MPS check failed: {e}")
        
        return torch.device("cpu")
    
    def _get_device_name(self) -> str:
        """Get human-readable device name."""
        if self.device.type == "mps":
            return "Apple Metal (MPS)"
        return "CPU"
    
    def _get_memory_info(self) -> dict:
        """Get unified memory information for M1/M2 Pro."""
        vm = psutil.virtual_memory()
        return {
            "total": vm.total / (1024**3),  # GB
            "available": vm.available / (1024**3),
            "percent": vm.percent,
            "used": vm.used / (1024**3),
        }
    
    def _check_float16_support(self) -> bool:
        """Check if float16 is supported on this device."""
        if self.device.type == "cpu":
            return False
        
        try:
            # Test float16 on MPS
            test_tensor = torch.randn(2, 2, device=self.device)
            test_tensor.half()
            return True
        except:
            return False
    
    def get_device(self) -> torch.device:
        """Get the optimal device for inference."""
        return self.device
    
    def get_dtype(self, use_half: bool = False) -> torch.dtype:
        """
        Get optimal dtype based on device capability.
        float16 is faster on MPS but has precision trade-offs.
        """
        if use_half and self.supports_float16 and self.device.type == "mps":
            return torch.float16
        return torch.float32
    
    def get_memory_budget(self) -> dict:
        """
        Calculate safe memory budgets for M1 Pro (16GB unified memory).
        Returns reasonable limits to prevent OOM.
        """
        available_gb = self.memory_info["available"]
        
        # Conservative allocation: use only 60% of available memory
        safe_allocation = available_gb * 0.6
        
        return {
            "available_gb": available_gb,
            "safe_model_gb": min(safe_allocation, 6.0),  # Cap at 6GB per model
            "batch_size": self._calculate_batch_size(available_gb),
            "max_image_size": self._calculate_max_image_size(available_gb),
        }
    
    def _calculate_batch_size(self, available_gb: float) -> int:
        """
        Calculate safe batch size based on available memory.
        For face enhancement, batch_size=1 is optimal.
        """
        if self.device.type == "mps":
            return 1  # Process faces sequentially on MPS
        return 1
    
    def _calculate_max_image_size(self, available_gb: float) -> int:
        """Maximum safe image resolution for processing."""
        if self.device.type == "mps":
            # M1 Pro with 16GB: can handle up to 2048x2048 comfortably
            return 2048
        return 1024
    
    def clear_cache(self):
        """Clear device cache to free memory."""
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()
    
    def report_device_info(self) -> str:
        """Generate detailed device information report."""
        report = f"""
╔══════════════════════════════════════╗
║    DEVICE CONFIGURATION REPORT       ║
╚══════════════════════════════════════╝

Device Type: {self.device_name}
PyTorch Version: {torch.__version__}
MPS Built: {torch.backends.mps.is_built()}
Float16 Support: {self.supports_float16}

Memory (Unified):
  Total: {self.memory_info['total']:.1f} GB
  Available: {self.memory_info['available']:.1f} GB
  Used: {self.memory_info['used']:.1f} GB ({self.memory_info['percent']:.1f}%)

Recommendations:
  - Process faces one at a time (batch_size=1)
  - Max image size: {self._calculate_max_image_size(self.memory_info['available'])}x{self._calculate_max_image_size(self.memory_info['available'])}
  - Use float32 for numerical stability
  - Clear cache between large operations
  - Avoid 4x upscaling; use 2x upscale instead
        """
        return report


class PerformanceOptimizer:
    """Optimize performance on M1/M2 Pro for face enhancement."""
    
    @staticmethod
    def optimize_tensor_on_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Move tensor to device with optimal memory alignment."""
        return tensor.to(device, non_blocking=True)
    
    @staticmethod
    def optimize_model_inference(model, use_half: bool = False) -> callable:
        """
        Wrap model inference with optimization.
        Returns optimized inference function.
        """
        if use_half:
            model = model.half()
        
        model.eval()
        
        def optimized_forward(input_tensor):
            with torch.no_grad():
                return model(input_tensor)
        
        return optimized_forward
    
    @staticmethod
    def create_safe_context():
        """Create context for safe tensor operations."""
        return torch.no_grad()
    
    @staticmethod
    def estimate_memory_usage(height: int, width: int, channels: int = 3, 
                            dtype: torch.dtype = torch.float32) -> float:
        """Estimate memory usage for tensor in GB."""
        dtype_size = 2 if dtype == torch.float16 else 4
        bytes_needed = height * width * channels * dtype_size
        return bytes_needed / (1024**3)


# Global device manager instance
_device_manager = None

def get_device_manager() -> MPSDeviceManager:
    """Get or create global device manager."""
    global _device_manager
    if _device_manager is None:
        _device_manager = MPSDeviceManager()
        print(_device_manager.report_device_info())
    return _device_manager


def get_device() -> torch.device:
    """Quick access to optimal device."""
    return get_device_manager().get_device()


def get_dtype(use_half: bool = False) -> torch.dtype:
    """Quick access to optimal dtype."""
    return get_device_manager().get_dtype(use_half)
