"""
Quality Mode Configuration for Face Enhancement
Provides preset configurations optimized for different use cases on M1 Pro
"""

from typing import Dict, Any
from enum import Enum


class QualityModeEnum(Enum):
    """Available quality modes."""
    FAST = "fast"
    HIGH_QUALITY = "high_quality"
    ULTRA = "ultra"


class QualityModeConfig:
    """Central configuration for all quality modes."""
    
    MODES: Dict[str, Dict[str, Any]] = {
        "fast": {
            "name": "Fast Processing",
            "description": "CPU-only processing without enhancement. Fastest performance.",
            "use_enhancer": False,
            "use_color_correction": True,
            "use_seamless_blend": False,
            "upscale_factor": 1,
            "device_strategy": "cpu",
            "processing_time_estimate": "Fast (<10s for 1080p)",
            "quality": "Standard - baseline quality",
            "recommended_for": "Quick previews, batch processing",
            "memory_usage": "~2GB peak",
        },
        
        "high_quality": {
            "name": "High Quality",
            "description": "MPS-accelerated enhancement with CodeFormer + 2x upscale",
            "use_enhancer": True,
            "enhancer_type": "codeformer",
            "codeformer_fidelity": 0.7,  # Balanced between quality and fidelity
            "use_color_correction": True,
            "use_seamless_blend": True,
            "use_sharpening": False,
            "upscale_factor": 2,
            "device_strategy": "mps",
            "processing_time_estimate": "Moderate (30-60s for 1080p)",
            "quality": "High - sharp, natural face",
            "recommended_for": "Most use cases, professional output",
            "memory_usage": "~4-5GB peak",
            "optimal": True,
        },
        
        "ultra": {
            "name": "Ultra High Quality",
            "description": "Maximum quality with CodeFormer 0.8 + 2x upscale + sharpening",
            "use_enhancer": True,
            "enhancer_type": "codeformer",
            "codeformer_fidelity": 0.8,  # Higher fidelity (more faithful to original)
            "use_color_correction": True,
            "use_seamless_blend": True,
            "use_sharpening": True,
            "upscale_factor": 2,
            "device_strategy": "mps",
            "processing_time_estimate": "Slow (60-90s for 1080p)",
            "quality": "Ultra - sharp details, perfect lighting",
            "recommended_for": "Final output, close-ups, high-stakes content",
            "memory_usage": "~5-6GB peak",
        },
    }
    
    # M1 Pro Optimization Settings
    M1_PRO_SETTINGS = {
        "max_memory_gb": 16,
        "safe_allocation_percent": 60,
        "batch_size": 1,  # Always process faces one at a time
        "max_face_size": 2048,  # Max resolution per face
        "use_float16": False,  # Use float32 for stability
        "clear_cache_interval": 1,  # Clear after each face
        "num_threads": 8,  # Use 8 efficiency cores on M1 Pro
        "gpu_utilization_target": 70,  # Target 70% for thermal efficiency
    }
    
    @classmethod
    def get_mode_config(cls, mode: str) -> Dict[str, Any]:
        """Get configuration for a specific mode."""
        return cls.MODES.get(mode.lower(), cls.MODES["high_quality"])
    
    @classmethod
    def get_all_modes(cls) -> Dict[str, Dict[str, Any]]:
        """Get all available modes."""
        return cls.MODES
    
    @classmethod
    def get_optimal_mode_for_m1_pro(cls) -> str:
        """Get the optimal mode for M1 Pro with 16GB unified memory."""
        return "high_quality"
    
    @classmethod
    def validate_mode(cls, mode: str) -> bool:
        """Check if mode exists."""
        return mode.lower() in cls.MODES
    
    @classmethod
    def format_mode_info(cls, mode: str) -> str:
        """Format mode information as a user-friendly string."""
        config = cls.get_mode_config(mode)
        
        info = f"""
╔════════════════════════════════════════════════════╗
║ {config['name'].upper():^48} ║
╚════════════════════════════════════════════════════╝

Description: {config['description']}

⏱️  Processing Time: {config['processing_time_estimate']}
📊 Quality Level: {config['quality']}
💾 Memory Usage: {config['memory_usage']}
🎯 Recommended For: {config['recommended_for']}

Settings:
  • Enhancement: {'Enabled' if config['use_enhancer'] else 'Disabled'}
  • Color Correction: {'Enabled' if config['use_color_correction'] else 'Disabled'}
  • Seamless Blending: {'Enabled' if config['use_seamless_blend'] else 'Disabled'}
  • Upscaling: {f"{config['upscale_factor']}x" if config['upscale_factor'] > 1 else 'None'}
  • Sharpening: {'Enabled' if config.get('use_sharpening', False) else 'Disabled'}
        """
        
        if config.get('enhancer_type') == 'codeformer':
            info += f"  • CodeFormer Fidelity: {config.get('codeformer_fidelity', 0.7)}\n"
        
        return info


class ProcessingConfig:
    """Configuration for processing pipeline."""
    
    # Image quality settings
    OUTPUT_JPEG_QUALITY = 95  # High quality JPEG
    OUTPUT_PNG_COMPRESSION = 9  # Maximum PNG compression
    
    # Interpolation for resizing
    INTERPOLATION_METHOD = "LANCZOS4"  # Use LANCZOS4 to prevent pixel loss
    
    # Blending settings
    FEATHER_SIZE = 25  # Size of feathering mask
    SEAMLESS_CLONE_METHOD = "MIXED_CLONE"  # Poisson blending method
    
    # Model cache directories
    MODEL_CACHE_DIRS = {
        "codeformer": "~/.cache/codeformer",
        "realesrgan": "~/.cache/realesrgan",
        "gfpgan": "~/.cache/gfpgan",
        "insightface": "~/.insightface/models",
    }
    
    # Performance tuning for M1 Pro
    PERFORMANCE_HINTS = {
        "disable_async_processing": True,  # Process sequentially
        "use_pinned_memory": False,  # Not beneficial on MPS
        "enable_cudnn_benchmark": False,  # Not applicable on Apple Silicon
        "num_workers": 0,  # No multiprocessing workers
    }


def print_enhancement_guide():
    """Print comprehensive enhancement guide for M1 Pro."""
    guide = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                  MPS FACE ENHANCEMENT GUIDE - M1 PRO                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎯 RECOMMENDED SETTINGS FOR M1 PRO WITH 16GB UNIFIED MEMORY:

Quality Mode: HIGH_QUALITY (Default - Optimal Balance)
  ✓ CodeFormer with fidelity=0.7
  ✓ 2x upscaling (NOT 4x - too slow)
  ✓ Color & lighting correction
  ✓ Seamless blending
  ✓ Processing time: 30-60 seconds per 1080p image

Why not ULTRA?
  → ULTRA adds sharpening and fidelity 0.8
  → 60-90 seconds per image (2x slower)
  → Only needed for close-up shots or final deliverables

Why not FAST?
  → No enhancement = blurry, low-quality faces
  → Only use for previews

⚡ PERFORMANCE TIPS:

1. Always use 2x upscale, NOT 4x
   • 4x takes 3-4x longer on M1 Pro
   • Minimal quality improvement

2. Process face-only, not full image
   • Reduces memory usage by 90%
   • Focuses enhancement on important region
   • Faster processing

3. Use PNG for maximum quality
   • JPEG at quality=95 is acceptable
   • PNG provides lossless output

4. Batch processing:
   • Process one image at a time
   • Clear GPU cache between images
   • Prevents memory buildup

5. Monitor resource usage:
   • Check Activity Monitor for memory
   • Stop if memory usage > 14GB
   • Thermal throttling above 85°C

⚠️  DO NOT:
  ✗ Use float16 (causes instability)
  ✗ Use batch_size > 1
  ✗ Resize entire image to 512x512 (pixel loss!)
  ✗ Use 4x upscaling
  ✗ Process full HD+ video in real-time

🔧 MODEL REQUIREMENTS:

Required Models (download from GitHub):
  1. CodeFormer: https://github.com/sczhou/CodeFormer/releases
     → Place in: ~/.cache/codeformer/codeformer.pth
  
  2. Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN/releases
     → Download: RealESRGAN_x2_compact.pth
     → Place in: ~/.cache/realesrgan/

  3. InsightFace: Automatic download
     → Models cached in: ~/.insightface/models

💡 QUALITY COMPARISON:

┌─────────────────┬──────────┬─────────┬──────────┬─────────────────┐
│ Mode            │ Time     │ Sharpness │ Skin    │ Lighting        │
├─────────────────┼──────────┼─────────┼──────────┼─────────────────┤
│ Fast            │ <10s     │ Blurry  │ Poor    │ Unchanged       │
│ HIGH_QUALITY    │ 40-60s   │ Sharp   │ Natural │ Corrected ✓     │
│ Ultra           │ 60-90s   │ V.Sharp │ Perfect │ Enhanced        │
└─────────────────┴──────────┴─────────┴──────────┴─────────────────┘

✅ EXPECTED RESULTS WITH HIGH_QUALITY:
  • Sharp facial features (no pixelation)
  • Natural skin texture
  • Proper lighting match
  • No artificial smoothing
  • Original face resolution preserved
  • Seamless blend with background
    """
    
    print(guide)
    
    # Print all mode details
    print("\n" + "="*80)
    print("DETAILED MODE CONFIGURATIONS")
    print("="*80)
    
    for mode_name in ["fast", "high_quality", "ultra"]:
        print(QualityModeConfig.format_mode_info(mode_name))
