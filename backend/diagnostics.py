#!/usr/bin/env python3
"""
MPS Device & Quality Configuration Tester for Face Enhancement
Tests and reports device capabilities, optimal settings, and performance
"""

import torch
import cv2
import numpy as np
import time
import psutil
from pathlib import Path
from core.device_manager import get_device_manager, PerformanceOptimizer
from core.quality_config import QualityModeConfig, print_enhancement_guide


def test_mps_availability():
    """Test if MPS is available and properly configured."""
    print("\n" + "="*80)
    print("MPS AVAILABILITY TEST")
    print("="*80)
    
    available = torch.backends.mps.is_available()
    built = torch.backends.mps.is_built()
    
    print(f"MPS Available: {available}")
    print(f"MPS Built: {built}")
    
    if available and built:
        print("✅ MPS is ready to use")
        return True
    else:
        print("❌ MPS not available - install torch with MPS support:")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cpu")
        return False


def test_mps_performance():
    """Benchmark MPS vs CPU performance."""
    print("\n" + "="*80)
    print("MPS vs CPU PERFORMANCE TEST")
    print("="*80)
    
    sizes = [100, 500, 1000, 2000]
    
    print(f"{'Matrix Size':<20} {'CPU (ms)':<15} {'MPS (ms)':<15} {'Speedup':<10}")
    print("-" * 60)
    
    for size in sizes:
        # CPU test
        x_cpu = torch.randn(size, size)
        start = time.time()
        for _ in range(10):
            torch.matmul(x_cpu, x_cpu)
        cpu_time = (time.time() - start) * 100
        
        # MPS test
        if torch.backends.mps.is_available():
            x_mps = torch.randn(size, size, device='mps')
            start = time.time()
            for _ in range(10):
                torch.matmul(x_mps, x_mps)
            torch.mps.synchronize()
            mps_time = (time.time() - start) * 100
            speedup = cpu_time / mps_time
            print(f"{size}x{size:<15} {cpu_time:<15.2f} {mps_time:<15.2f} {speedup:<10.2f}x")
        else:
            print(f"{size}x{size:<15} {cpu_time:<15.2f} {'N/A':<15} {'N/A':<10}")


def test_device_manager():
    """Test device manager initialization."""
    print("\n" + "="*80)
    print("DEVICE MANAGER TEST")
    print("="*80)
    
    mgr = get_device_manager()
    print(mgr.report_device_info())
    
    return mgr


def test_model_loading():
    """Test loading enhancement models."""
    print("\n" + "="*80)
    print("MODEL AVAILABILITY TEST")
    print("="*80)
    
    models = {
        "CodeFormer": Path.home() / ".cache" / "codeformer" / "codeformer.pth",
        "Real-ESRGAN 2x": Path.home() / ".cache" / "realesrgan" / "RealESRGAN_x2_compact.pth",
    }
    
    for model_name, model_path in models.items():
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024**2)
            print(f"✅ {model_name}: {size_mb:.1f} MB")
        else:
            print(f"❌ {model_name}: NOT FOUND")
            print(f"   Place at: {model_path}")
    
    # Check InsightFace
    insightface_dir = Path.home() / ".insightface" / "models"
    if insightface_dir.exists():
        count = len(list(insightface_dir.glob("**/*")))
        print(f"✅ InsightFace Models: {count} files")
    else:
        print(f"❌ InsightFace Models: Will auto-download on first use")


def test_quality_modes():
    """Display and test all quality modes."""
    print("\n" + "="*80)
    print("QUALITY MODES CONFIGURATION")
    print("="*80)
    
    modes = QualityModeConfig.get_all_modes()
    
    for mode_name in modes.keys():
        print(QualityModeConfig.format_mode_info(mode_name))


def test_memory_management():
    """Test memory management."""
    print("\n" + "="*80)
    print("MEMORY MANAGEMENT TEST")
    print("="*80)
    
    mgr = get_device_manager()
    
    # Simulate tensor creation
    test_sizes = [
        ("Small (256x256x3)", 256, 256, 3),
        ("Medium (512x512x3)", 512, 512, 3),
        ("Large (1024x1024x3)", 1024, 1024, 3),
        ("XLarge (2048x2048x3)", 2048, 2048, 3),
    ]
    
    print(f"\n{'Image Size':<25} {'Memory Usage':<15} {'Safe':<10}")
    print("-" * 50)
    
    for label, h, w, c in test_sizes:
        estimated = PerformanceOptimizer.estimate_memory_usage(h, w, c)
        budget = mgr.get_memory_budget()
        safe = "✅" if estimated < budget['safe_model_gb'] else "⚠️"
        print(f"{label:<25} {estimated:.3f} GB{'':<10} {safe:<10}")


def test_image_processing():
    """Test image processing performance."""
    print("\n" + "="*80)
    print("IMAGE PROCESSING PERFORMANCE TEST")
    print("="*80)
    
    # Create test image
    test_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    print(f"Test Image Size: {test_image.shape}")
    
    # Test face region extraction
    print("\n1. Face Region Processing:")
    face_h, face_w = 256, 256
    face_region = test_image[400:400+face_h, 700:700+face_w].copy()
    print(f"   Face region size: {face_region.shape} ✓")
    
    # Test color space conversions
    print("\n2. Color Space Conversions:")
    start = time.time()
    bgr_to_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
    rgb_to_lab = cv2.cvtColor(bgr_to_rgb, cv2.COLOR_RGB2LAB)
    lab_to_rgb = cv2.cvtColor(rgb_to_lab, cv2.COLOR_LAB2RGB)
    elapsed = (time.time() - start) * 1000
    print(f"   BGR→RGB→LAB→RGB: {elapsed:.2f}ms ✓")
    
    # Test resizing with different interpolations
    print("\n3. Resizing Performance (up to 512x512):")
    upscale_target = (512, 512)
    
    methods = [
        ("INTER_LINEAR", cv2.INTER_LINEAR),
        ("INTER_LANCZOS4", cv2.INTER_LANCZOS4),
        ("INTER_CUBIC", cv2.INTER_CUBIC),
    ]
    
    for name, method in methods:
        start = time.time()
        resized = cv2.resize(face_region, upscale_target, interpolation=method)
        elapsed = (time.time() - start) * 1000
        print(f"   {name:<20}: {elapsed:.2f}ms")
    
    print("\n   Recommendation: Use INTER_LANCZOS4 (best quality)")


def run_all_tests():
    """Run all diagnostic tests."""
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║           MPS FACE ENHANCEMENT - SYSTEM DIAGNOSTICS                         ║")
    print("║              MacBook Pro M1/M2 Pro - PyTorch MPS Backend                    ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    
    # Run tests
    mps_ok = test_mps_availability()
    
    if mps_ok:
        test_mps_performance()
    
    mgr = test_device_manager()
    test_model_loading()
    test_quality_modes()
    test_memory_management()
    test_image_processing()
    
    # Print guide
    print("\n" * 1)
    print_enhancement_guide()
    
    # Final recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR YOUR M1 PRO")
    print("="*80)
    
    mem_info = mgr.memory_info
    
    recommendations = f"""
✓ OPTIMAL SETTINGS:
  • Quality Mode: HIGH_QUALITY (recommended)
  • Upscaling: 2x (NOT 4x - much slower)
  • CodeFormer Fidelity: 0.7 (balance quality/fidelity)
  • Device: MPS (Metal Performance Shaders)
  • Data Type: float32 (for accuracy)
  • Batch Size: 1 (process one face at a time)

✓ MEMORY CONFIGURATION:
  • Total Unified Memory: {mem_info['total']:.1f} GB
  • Currently Available: {mem_info['available']:.1f} GB
  • Safe Allocation: {mgr.get_memory_budget()['safe_model_gb']:.1f} GB
  • Keep Free: ≥ 2GB (system buffer)

✓ PROCESSING EXPECTATIONS:
  • FAST Mode: <10 seconds per image
  • HIGH_QUALITY: 40-60 seconds per 1080p image
  • ULTRA: 60-90 seconds per 1080p image

⚠️  IMPORTANT DO NOTs:
  • DON'T use float16 (instability on MPS)
  • DON'T scale full image to 512x512 (pixel loss!)
  • DON'T use 4x upscaling (too slow for marginal gain)
  • DON'T process more than one face simultaneously
  • DON'T leave system RAM < 2GB

✅ BEST PRACTICES:
  • Always use LANCZOS4 for resizing
  • Process face regions only (not full image)
  • Clear GPU cache between images
  • Monitor Activity Monitor for thermal throttling
  • Use PNG output for maximum quality
  • Download models to fast SSD (not external drives)

🚀 TO GET STARTED:
  1. Download CodeFormer & Real-ESRGAN models (see MPS_SETUP_GUIDE.md)
  2. Test the setup: python3 backend/app.py
  3. Check frontend: npm run dev
  4. Start processing with HIGH_QUALITY mode
    """
    
    print(recommendations)
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)
    print("\nFor detailed setup guide, see: MPS_SETUP_GUIDE.md")


if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⚠️  Diagnostics interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during diagnostics: {e}")
        import traceback
        traceback.print_exc()
