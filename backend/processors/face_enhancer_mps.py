"""
MPS-Optimized High-Resolution Face Enhancement Pipeline
Designed specifically for Apple Silicon (M1/M2 Pro) MacBooks
Combines CodeFormer, GFPGAN, and Real-ESRGAN with intelligent quality modes
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Union
import os
from pathlib import Path
from core.device_manager import get_device_manager, get_device, get_dtype, PerformanceOptimizer


class QualityMode:
    """Configuration for different quality modes optimized for M1 Pro."""
    
    FAST = {
        "name": "fast",
        "use_enhancer": False,
        "upscale_factor": 1,
        "use_color_correction": True,
        "processing_region": "face_only",
        "device_strategy": "cpu",
        "description": "CPU-only, no enhancement, fastest processing"
    }
    
    HIGH_QUALITY = {
        "name": "high_quality",
        "use_enhancer": True,
        "enhancer_type": "codeformer",
        "codeformer_fidelity": 0.7,  # 0.0-1.0: balance between quality and fidelity
        "upscale_factor": 2,
        "use_color_correction": True,
        "use_seamless_blend": True,
        "processing_region": "face_only",
        "device_strategy": "mps",
        "description": "MPS-accelerated, CodeFormer enhancement, 2x upscale"
    }
    
    ULTRA = {
        "name": "ultra",
        "use_enhancer": True,
        "enhancer_type": "codeformer",
        "codeformer_fidelity": 0.8,
        "upscale_factor": 2,  # 2x is optimal for M1 Pro; avoid 4x
        "use_color_correction": True,
        "use_seamless_blend": True,
        "use_sharpening": True,
        "processing_region": "face_only",
        "device_strategy": "mps",
        "description": "MPS-accelerated, CodeFormer 0.8 fidelity, 2x upscale + sharpening"
    }


class ColorCorrectionEngine:
    """
    Corrects color and lighting discrepancies between swapped face and original image.
    Uses LAB color space for natural results.
    """
    
    @staticmethod
    def match_color_lab(source_face: np.ndarray, target_face: np.ndarray) -> np.ndarray:
        """
        Match color of source face to target face using LAB color space.
        Preserves luminance from target, uses chrominance from source.
        
        Args:
            source_face: RGB face to color-correct (swapped face)
            target_face: Reference face for color matching (original region)
        
        Returns:
            Color-corrected face in RGB
        """
        # Convert to LAB
        source_lab = cv2.cvtColor(source_face, cv2.COLOR_RGB2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target_face, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Calculate mean and std for each channel
        source_mean = source_lab.mean(axis=(0, 1))
        source_std = source_lab.std(axis=(0, 1))
        
        target_mean = target_lab.mean(axis=(0, 1))
        target_std = target_lab.std(axis=(0, 1))
        
        # Match statistics
        result_lab = source_lab.copy()
        for i in range(3):
            result_lab[:, :, i] = ((source_lab[:, :, i] - source_mean[i]) * 
                                   (target_std[i] / source_std[i] + 1e-6) + 
                                   target_mean[i])
        
        # Clip values and convert back
        result_lab = np.clip(result_lab, [0, -128, -128], [255, 127, 127])
        result_rgb = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        return result_rgb
    
    @staticmethod
    def blend_illumination(swapped_face: np.ndarray, original_region: np.ndarray,
                          alpha: float = 0.3) -> np.ndarray:
        """
        Blend illumination from original region into swapped face.
        Helps match lighting naturally.
        """
        # Convert to grayscale for illumination
        original_gray = cv2.cvtColor(original_region, cv2.COLOR_RGB2GRAY).astype(np.float32)
        swapped_gray = cv2.cvtColor(swapped_face, cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        # Calculate illumination difference
        illumination_diff = original_gray - swapped_gray
        
        # Apply smoothing to avoid artifacts
        illumination_diff = cv2.GaussianBlur(illumination_diff, (5, 5), 0)
        
        # Apply to each channel
        result = swapped_face.astype(np.float32)
        for i in range(3):
            result[:, :, i] = np.clip(
                result[:, :, i] + illumination_diff * alpha,
                0, 255
            )
        
        return result.astype(np.uint8)


class SeamlessBlendingEngine:
    """Seamlessly blend enhanced face back into original image."""
    
    @staticmethod
    def seamless_clone(background: np.ndarray, foreground: np.ndarray, 
                       mask: np.ndarray, center: Tuple[int, int]) -> np.ndarray:
        """
        Seamlessly clone foreground into background using Poisson blending.
        
        Args:
            background: Original image
            foreground: Enhanced face region
            mask: Binary mask of face region
            center: Center coordinates for blending
        """
        try:
            result = cv2.seamlessClone(
                foreground,
                background,
                mask,
                center,
                cv2.MIXED_CLONE
            )
            return result
        except Exception as e:
            print(f"Seamless clone failed, using alpha blend: {e}")
            # Fallback to alpha blending
            return SeamlessBlendingEngine.alpha_blend(background, foreground, mask)
    
    @staticmethod
    def alpha_blend(background: np.ndarray, foreground: np.ndarray,
                   mask: np.ndarray, feather_size: int = 25) -> np.ndarray:
        """
        Blend using feathered alpha mask for smooth edges.
        
        Args:
            background: Original image
            foreground: Enhanced face
            mask: Binary mask
            feather_size: Size of feathering kernel
        """
        # Feather the mask for smooth blending
        feathered_mask = cv2.GaussianBlur(
            mask.astype(np.float32) / 255.0,
            (feather_size, feather_size),
            0
        )
        
        # Blend
        result = background.copy().astype(np.float32)
        feathered_mask = np.stack([feathered_mask] * 3, axis=2)  # 3 channels
        
        result = (result * (1 - feathered_mask) + 
                 foreground.astype(np.float32) * feathered_mask)
        
        return np.clip(result, 0, 255).astype(np.uint8)


class CodeFormerWrapper:
    """
    Wrapper for CodeFormer model optimized for MPS.
    CodeFormer provides superior face restoration with fidelity control.
    """
    
    def __init__(self, device: torch.device, use_half: bool = False):
        self.device = device
        self.use_half = use_half and device.type == "mps"
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load CodeFormer model with error handling."""
        try:
            # Try to import CodeFormer
            from codeformer import CodeFormer
            
            # Check for cached model
            cache_dir = Path.home() / ".cache" / "codeformer"
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = cache_dir / "codeformer.pth"
            
            if not model_path.exists():
                print("⚠️  CodeFormer model not found. Download from:")
                print("https://github.com/sczhou/CodeFormer/releases")
                print(f"Place in: {cache_dir}")
                return
            
            # Load model
            self.model = CodeFormer(device=self.device)
            self.model.eval()
            
            if self.use_half:
                self.model = self.model.half()
            
            print(f"✓ CodeFormer loaded (device: {self.device}, half: {self.use_half})")
            
        except ImportError:
            print("⚠️  CodeFormer not installed. Install with:")
            print("pip install codeformer-torch")
        except Exception as e:
            print(f"⚠️  Failed to load CodeFormer: {e}")
    
    def enhance(self, face_image: np.ndarray, fidelity: float = 0.7) -> Optional[np.ndarray]:
        """
        Enhance face using CodeFormer.
        
        Args:
            face_image: Input face (cv2 BGR format)
            fidelity: 0.0-1.0, higher = more faithful to input, lower = more enhanced
        
        Returns:
            Enhanced face or None if enhancement fails
        """
        if self.model is None:
            return None
        
        try:
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            face_tensor = torch.from_numpy(face_rgb).float() / 255.0
            face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0)
            face_tensor = face_tensor.to(self.device)
            
            if self.use_half:
                face_tensor = face_tensor.half()
            
            with torch.no_grad():
                # CodeFormer with fidelity control
                enhanced = self.model(face_tensor, fidelity=fidelity)
            
            # Convert back to numpy
            enhanced = enhanced.squeeze(0).permute(1, 2, 0)
            enhanced = (enhanced.cpu().numpy() * 255).astype(np.uint8)
            
            # Convert RGB back to BGR
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
            
            return enhanced_bgr
            
        except Exception as e:
            print(f"CodeFormer enhancement failed: {e}")
            return None
    
    def close(self):
        """Clean up model."""
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache() if torch.cuda.is_available() else torch.mps.empty_cache()


class RealESRGANWrapper:
    """
    Wrapper for Real-ESRGAN optimized for MPS.
    Best for 2x upscaling on M1 Pro.
    """
    
    def __init__(self, device: torch.device, scale: int = 2, use_half: bool = False):
        self.device = device
        self.scale = min(scale, 2)  # Cap at 2x for M1 Pro
        self.use_half = use_half and device.type == "mps"
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Real-ESRGAN model."""
        try:
            from realesrgan import RealESRGANer
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact
            
            cache_dir = Path.home() / ".cache" / "realesrgan"
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Use compact model for M1 Pro
            model_path = cache_dir / f"RealESRGAN_x{self.scale}_compact.pth"
            
            if not model_path.exists():
                print(f"⚠️  Real-ESRGAN x{self.scale} model not found.")
                print("Download from: https://github.com/xinntao/Real-ESRGAN/releases")
                return
            
            self.model = RealESRGANer(
                scale=self.scale,
                model_path=str(model_path),
                upsampler=SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, 
                                         num_conv=16, upscale=self.scale),
                tile=0,  # No tiling needed for face regions
                device=self.device
            )
            
            print(f"✓ Real-ESRGAN x{self.scale} loaded (device: {self.device})")
            
        except Exception as e:
            print(f"⚠️  Failed to load Real-ESRGAN: {e}")
    
    def upscale(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Upscale image using Real-ESRGAN.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            Upscaled image or None if upscaling fails
        """
        if self.model is None:
            return None
        
        try:
            with torch.no_grad():
                output, _ = self.model.enhance(image, outscale=self.scale)
            return output
        except Exception as e:
            print(f"Real-ESRGAN upscaling failed: {e}")
            return None


class SharpeningEngine:
    """Advanced sharpening for ultra mode."""
    
    @staticmethod
    def unsharp_mask(image: np.ndarray, kernel_size: int = 5, 
                    strength: float = 1.5) -> np.ndarray:
        """
        Apply unsharp mask for controlled sharpening.
        
        Args:
            image: Input image
            kernel_size: Gaussian blur kernel size
            strength: Sharpening strength (1.0 = natural)
        """
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        sharpened = cv2.addWeighted(
            image, 1.0 + strength,
            blurred, -strength,
            0
        )
        return np.clip(sharpened, 0, 255).astype(np.uint8)


class MPS_FaceEnhancer:
    """
    Main face enhancement engine optimized for Apple Silicon M1 Pro.
    
    Processing Pipeline:
    1. Face detection (external)
    2. CodeFormer restoration (if enabled)
    3. Real-ESRGAN 2x upscaling (if enabled)
    4. Color/lighting correction (LAB color space)
    5. Seamless blending back to original image
    6. Sharpening (if ultra mode)
    """
    
    def __init__(self, quality_mode: str = "high_quality"):
        """
        Initialize enhancement engine.
        
        Args:
            quality_mode: "fast", "high_quality", or "ultra"
        """
        self.device_manager = get_device_manager()
        self.device = self.device_manager.get_device()
        self.dtype = self.device_manager.get_dtype(use_half=False)
        
        # Select quality configuration
        if quality_mode.lower() == "fast":
            self.config = QualityMode.FAST
        elif quality_mode.lower() == "ultra":
            self.config = QualityMode.ULTRA
        else:
            self.config = QualityMode.HIGH_QUALITY
        
        print(f"\n🎨 MPS_FaceEnhancer initialized")
        print(f"   Mode: {self.config['name'].upper()}")
        print(f"   Device: {self.device_manager.device_name}")
        print(f"   Description: {self.config['description']}")
        
        # Initialize enhancement models
        self.codeformer = None
        self.realesrgan = None
        
        if self.config.get("use_enhancer"):
            if self.config.get("enhancer_type") == "codeformer":
                self.codeformer = CodeFormerWrapper(self.device, use_half=False)
            
            if self.config.get("upscale_factor", 1) > 1:
                self.realesrgan = RealESRGANWrapper(
                    self.device,
                    scale=self.config.get("upscale_factor", 2),
                    use_half=False
                )
        
        self.color_engine = ColorCorrectionEngine()
        self.blend_engine = SeamlessBlendingEngine()
        self.sharpener = SharpeningEngine()
    
    def enhance_face_region(self, swapped_face: np.ndarray,
                           original_region: np.ndarray,
                           face_mask: np.ndarray,
                           original_image: np.ndarray,
                           face_bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Complete enhancement pipeline for a single face region.
        
        Args:
            swapped_face: The face-swapped result (BGR)
            original_region: Original face region for reference (BGR)
            face_mask: Binary mask of face (same size as swapped_face)
            original_image: Full original image for seamless blending
            face_bbox: (x, y, w, h) bounding box of face in original image
        
        Returns:
            Enhanced face region ready for blending
        """
        
        current_face = swapped_face.copy()
        x, y, w, h = face_bbox
        
        # Step 1: CodeFormer Enhancement (if enabled)
        if self.codeformer and self.config.get("use_enhancer"):
            enhanced = self.codeformer.enhance(
                current_face,
                fidelity=self.config.get("codeformer_fidelity", 0.7)
            )
            if enhanced is not None:
                # Resize back to original size if CodeFormer changed dimensions
                if enhanced.shape[:2] != current_face.shape[:2]:
                    enhanced = cv2.resize(
                        enhanced,
                        (current_face.shape[1], current_face.shape[0]),
                        interpolation=cv2.INTER_LANCZOS4
                    )
                current_face = enhanced
                print("   ✓ CodeFormer enhancement applied")
        
        # Step 2: Real-ESRGAN 2x Upscaling (if enabled)
        if self.realesrgan and self.config.get("upscale_factor", 1) > 1:
            upscaled = self.realesrgan.upscale(current_face)
            if upscaled is not None:
                # Resize back to original size (2x upscale then downscale to maintain aspect)
                upscaled = cv2.resize(
                    upscaled,
                    (current_face.shape[1], current_face.shape[0]),
                    interpolation=cv2.INTER_LANCZOS4
                )
                current_face = upscaled
                print("   ✓ Real-ESRGAN 2x upscaling applied")
        
        # Step 3: Color Correction in LAB Space
        if self.config.get("use_color_correction"):
            # Convert BGR to RGB for color correction
            current_rgb = cv2.cvtColor(current_face, cv2.COLOR_BGR2RGB)
            original_rgb = cv2.cvtColor(original_region, cv2.COLOR_BGR2RGB)
            
            # Match color
            color_corrected = self.color_engine.match_color_lab(current_rgb, original_rgb)
            
            # Blend illumination
            illumination_blended = self.color_engine.blend_illumination(
                color_corrected,
                original_rgb,
                alpha=0.3
            )
            
            # Convert back to BGR
            current_face = cv2.cvtColor(illumination_blended, cv2.COLOR_RGB2BGR)
            print("   ✓ Color & illumination correction applied")
        
        # Step 4: Sharpening (if ultra mode)
        if self.config.get("use_sharpening"):
            current_face = self.sharpener.unsharp_mask(current_face, strength=1.2)
            print("   ✓ Sharpening applied")
        
        # Step 5: Prepare for seamless blending
        if self.config.get("use_seamless_blend"):
            # Ensure mask matches swapped face dimensions
            if face_mask.shape[:2] != current_face.shape[:2]:
                face_mask = cv2.resize(
                    face_mask,
                    (current_face.shape[1], current_face.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )
            
            # Calculate center for seamless clone
            center = (x + w // 2, y + h // 2)
            
            # Seamless blend
            result = self.blend_engine.seamless_clone(
                original_image,
                current_face,
                face_mask,
                center
            )
            print("   ✓ Seamless blending applied")
            return result
        
        return current_face
    
    def enhance_image(self, image: np.ndarray, faces_data: list) -> np.ndarray:
        """
        Enhance all faces in an image.
        
        Args:
            image: Original image (BGR)
            faces_data: List of face data {"face": ndarray, "mask": ndarray, "bbox": tuple}
        
        Returns:
            Enhanced image
        """
        if not faces_data:
            return image
        
        result_image = image.copy()
        
        for i, face_data in enumerate(faces_data):
            print(f"\n✨ Enhancing face {i+1}/{len(faces_data)}")
            
            swapped_face = face_data.get("face")
            face_mask = face_data.get("mask")
            face_bbox = face_data.get("bbox")
            original_region = face_data.get("original_region", image.copy())
            
            if swapped_face is None or face_mask is None:
                continue
            
            # Enhance this face
            enhanced = self.enhance_face_region(
                swapped_face,
                original_region,
                face_mask,
                result_image,
                face_bbox
            )
            
            # Update result image (in-place blending)
            x, y, w, h = face_bbox
            result_image[y:y+h, x:x+w] = enhanced[y:y+h, x:x+w]
        
        # Clear cache after processing
        self.device_manager.clear_cache()
        
        return result_image
    
    def cleanup(self):
        """Release model resources."""
        if self.codeformer:
            self.codeformer.close()
        if self.realesrgan:
            del self.realesrgan
        
        self.device_manager.clear_cache()
