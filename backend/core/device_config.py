"""
Advanced GPU/Device Configuration and Management
Handles MPS (Apple Silicon), CUDA, and CPU execution
"""
import torch
import platform
import psutil
import logging
from typing import Literal, Optional, Dict
import os

logger = logging.getLogger(__name__)


class DeviceConfig:
    """Manages device configuration for optimal processing"""
    
    def __init__(self):
        self.device_type = self._detect_device()
        self.device = torch.device(self.device_type)
        self.memory_limit = self._get_memory_limit()
        self.processor_info = self._get_processor_info()
        
    def _detect_device(self) -> str:
        """Detect best available device: mps > cuda > cpu"""
        # Check Apple Silicon MPS first
        if torch.backends.mps.is_available():
            try:
                torch.ones(1, device='mps')
                logger.info("✓ Apple MPS (Metal Performance Shaders) available")
                return 'mps'
            except Exception as e:
                logger.warning(f"MPS available but not working: {e}")
        
        # Check CUDA
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            return 'cuda'
        
        # Fallback to CPU
        logger.info("→ Using CPU device")
        return 'cpu'
    
    def _get_memory_limit(self) -> int:
        """Get safe memory limit in bytes"""
        system_memory = psutil.virtual_memory().total
        
        if self.device_type == 'mps':
            # M1/M2/M3 unified memory - use 60% safely
            limit = int(system_memory * 0.6)
        elif self.device_type == 'cuda':
            # NVIDIA - use 90% of VRAM
            limit = int(torch.cuda.get_device_properties(0).total_memory * 0.9)
        else:
            # CPU - use 50% of system RAM
            limit = int(system_memory * 0.5)
        
        return limit
    
    def _get_processor_info(self) -> Dict:
        """Get processor information"""
        system = platform.system()
        processor = platform.processor()
        
        info = {
            'system': system,
            'processor': processor,
            'device_type': self.device_type,
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3),
        }
        
        if self.device_type == 'cuda':
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return info
    
    def get_device(self) -> torch.device:
        """Get the configured device"""
        return self.device
    
    def get_dtype(self) -> torch.dtype:
        """Get optimal dtype based on device"""
        if self.device_type in ['mps', 'cpu']:
            return torch.float32
        else:  # CUDA
            return torch.float16  # Use half precision on NVIDIA GPUs
    
    def clear_cache(self) -> None:
        """Clear device cache"""
        if self.device_type == 'cuda':
            torch.cuda.empty_cache()
        elif self.device_type == 'mps':
            torch.mps.empty_cache()
    
    def get_config_summary(self) -> str:
        """Get human-readable configuration summary"""
        summary = f"""
╔══════════════════════════════════════════════════════╗
║              DEVICE CONFIGURATION                    ║
╠══════════════════════════════════════════════════════╣
║ Device Type:      {self.processor_info['device_type'].upper():30} ║
║ System:           {self.processor_info['system']:30} ║
║ Processor:        {self.processor_info['processor']:30} ║
║ Total Memory:     {self.processor_info['total_memory_gb']:.1f} GB{' ':24}║
║ Memory Limit:     {self.memory_limit / (1024**3):.1f} GB{' ':24}║
╚══════════════════════════════════════════════════════╝
        """
        if self.device_type == 'cuda':
            summary += f"\nGPU: {self.processor_info['gpu_name']} ({self.processor_info['gpu_memory_gb']:.1f} GB)"
        return summary


# Singleton instance
_device_config: Optional[DeviceConfig] = None


def get_device_config() -> DeviceConfig:
    """Get or create device configuration singleton"""
    global _device_config
    if _device_config is None:
        _device_config = DeviceConfig()
        print(_device_config.get_config_summary())
    return _device_config
