import pynvml
import torch
import psutil
import platform

class GPUMonitor:
    def __init__(self):
        self.has_cuda = False
        try:
            pynvml.nvmlInit()
            self.has_cuda = True
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self.has_cuda = False

    def get_stats(self) -> dict:
        stats = {
            "platform": platform.system(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        }

        if self.has_cuda:
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            stats.update({
                "gpu_name": pynvml.nvmlDeviceGetName(self.handle),
                "vram_used_mb": mem.used // 1048576,
                "vram_total_mb": mem.total // 1048576,
                "vram_percent": round(mem.used / mem.total * 100, 1),
                "gpu_util_percent": util.gpu,
                "temperature_c": temp,
            })
        elif stats["mps_available"]:
            # Apple Silicon Unified Memory
            vm = psutil.virtual_memory()
            stats.update({
                "gpu_name": "Apple Silicon (MPS)",
                "vram_used_mb": vm.used // 1048576,
                "vram_total_mb": vm.total // 1048576,
                "vram_percent": vm.percent,
                "gpu_util_percent": 0, # Hard to get directly without external tools
                "temperature_c": 0,
            })
        else:
            vm = psutil.virtual_memory()
            stats.update({
                "gpu_name": "CPU",
                "vram_used_mb": vm.used // 1048576,
                "vram_total_mb": vm.total // 1048576,
                "vram_percent": vm.percent,
                "gpu_util_percent": psutil.cpu_percent(),
                "temperature_c": 0,
            })
            
        return stats
