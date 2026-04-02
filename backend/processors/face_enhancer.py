"""
Advanced Face Enhancement Pipeline
Provides ultra-realistic face swap post-processing including:
- Super resolution (GFPGAN, CodeFormer, Real-ESRGAN)
- Color & lighting correction
- Seamless blending
- Detail restoration
- Noise matching
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict
import os


class FaceEnhancer:
    """
    Multi-stage face enhancement pipeline for ultra-realistic results.
    
    Processing Order:
    1. Super Resolution (upscale swapped face)
    2. Color & Lighting Correction
    3. Seamless Blending
    4. Detail Restoration
    5. Noise Matching
    """
    
    def __init__(self, quality_mode: str = "high_quality", device: str = "cpu"):
        """
        Args:
            quality_mode: 'fast', 'high_quality', or 'ultra_realistic'
            device: 'cpu' or 'cuda'
        """
        self.quality_mode = quality_mode
        self.device = device
        self.gfpgan_model = None
        self.codeformer_model = None
        self.realesrgan_model = None
        
        # Load models based on quality mode
        self._load_models()
    
    def _load_models(self):
        """Lazy load enhancement models based on quality mode."""
        try:
            if self.quality_mode in ['high_quality', 'ultra_realistic']:
                self._try_load_gfpgan()
            
            if self.quality_mode == 'ultra_realistic':
                self._try_load_codeformer()
                self._try_load_realesrgan()
        except Exception as e:
            print(f"Warning: Could not load all enhancement models: {e}")
            print("Falling back to OpenCV-based enhancement")
    
    def _try_load_gfpgan(self):
        """Load GFPGAN model for face restoration."""
        try:
            from gfpgan import GFPGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            
            model_path = os.path.expanduser('~/.cache/gfpgan/GFPGANv1.4.pth')
            if not os.path.exists(model_path):
                print(f"GFPGAN model not found at {model_path}. Will download on first use.")
            
            # Upscale background with RealESRGAN
            bg_upsampler = None
            if self.quality_mode == 'ultra_realistic':
                bg_upsampler = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                                      num_block=23, num_grow_ch=32, scale=2)
            
            self.gfpgan_model = GFPGANer(
                model_path=model_path,
                upscale=2,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=bg_upsampler,
                device=self.device
            )
            print("✓ GFPGAN loaded successfully")
        except ImportError:
            print("GFPGAN not installed. Install with: pip install gfpgan")
        except Exception as e:
            print(f"Could not load GFPGAN: {e}")
    
    def _try_load_codeformer(self):
        """Load CodeFormer model for face restoration."""
        try:
            from codeformer import CodeFormer
            
            model_path = os.path.expanduser('~/.cache/codeformer/codeformer.pth')
            self.codeformer_model = CodeFormer(
                model_path=model_path,
                device=self.device,
                upscale=2
            )
            print("✓ CodeFormer loaded successfully")
        except ImportError:
            print("CodeFormer not installed. Using GFPGAN instead.")
        except Exception as e:
            print(f"Could not load CodeFormer: {e}")
    
    def _try_load_realesrgan(self):
        """Load Real-ESRGAN for super resolution."""
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                          num_block=23, num_grow_ch=32, scale=4)
            
            model_path = os.path.expanduser('~/.cache/realesrgan/RealESRGAN_x4plus.pth')
            
            self.realesrgan_model = RealESRGANer(
                scale=4,
                model_path=model_path,
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=True if self.device == 'cuda' else False,
                device=self.device
            )
            print("✓ Real-ESRGAN loaded successfully")
        except ImportError:
            print("Real-ESRGAN not installed.")
        except Exception as e:
            print(f"Could not load Real-ESRGAN: {e}")
    
    def enhance_face(self, 
                     swapped_img: np.ndarray,
                     original_img: np.ndarray,
                     face_bbox: Tuple[int, int, int, int],
                     face_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Main enhancement pipeline.
        
        Args:
            swapped_img: Image with face already swapped
            original_img: Original target image for reference
            face_bbox: (x1, y1, x2, y2) bounding box of swapped face
            face_mask: Optional mask of face region
            
        Returns:
            Enhanced image with ultra-realistic face
        """
        result = swapped_img.copy()
        
        # Step 1: Super Resolution
        result = self._apply_super_resolution(result, face_bbox)
        
        # Step 2: Color & Lighting Correction
        result = self._apply_color_correction(result, original_img, face_bbox)
        
        # Step 3: Seamless Blending
        if face_mask is not None:
            result = self._apply_seamless_blend(result, original_img, face_mask, face_bbox)
        
        # Step 4: Detail Restoration
        result = self._apply_detail_restoration(result, face_bbox)
        
        # Step 5: Noise Matching
        result = self._apply_noise_matching(result, original_img, face_bbox)
        
        return result
    
    def _apply_super_resolution(self, img: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Apply super resolution to face region."""
        if self.quality_mode == 'fast':
            return img  # Skip for fast mode
        
        x1, y1, x2, y2 = bbox
        
        # Try GFPGAN first
        if self.gfpgan_model is not None:
            try:
                _, _, restored_img = self.gfpgan_model.enhance(
                    img, 
                    has_aligned=False, 
                    only_center_face=False,
                    paste_back=True
                )
                return restored_img
            except Exception as e:
                print(f"GFPGAN enhancement failed: {e}")
        
        # Try CodeFormer
        if self.codeformer_model is not None:
            try:
                restored_img = self.codeformer_model.enhance(img, fidelity_weight=0.7)
                return restored_img
            except Exception as e:
                print(f"CodeFormer enhancement failed: {e}")
        
        # Fallback: OpenCV super resolution
        return self._opencv_super_resolution(img, bbox)
    
    def _opencv_super_resolution(self, img: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Fallback super resolution using OpenCV."""
        x1, y1, x2, y2 = bbox
        
        # Extract face region
        face = img[y1:y2, x1:x2].copy()
        
        # Upscale using Lanczos interpolation (better than bicubic for faces)
        h, w = face.shape[:2]
        upscaled = cv2.resize(face, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
        
        # Apply bilateral filter to smooth while preserving edges
        upscaled = cv2.bilateralFilter(upscaled, 9, 75, 75)
        
        # Downscale back to original size with enhanced detail
        enhanced_face = cv2.resize(upscaled, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        # Paste back
        result = img.copy()
        result[y1:y2, x1:x2] = enhanced_face
        
        return result
    
    def _apply_color_correction(self, img: np.ndarray, reference: np.ndarray, 
                                bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Advanced color and lighting correction."""
        x1, y1, x2, y2 = bbox
        
        # Extract regions
        swapped_face = img[y1:y2, x1:x2].copy()
        reference_face = reference[y1:y2, x1:x2].copy()
        
        if self.quality_mode == 'ultra_realistic':
            # LAB color space matching (most accurate)
            corrected_face = self._lab_color_transfer(swapped_face, reference_face)
        else:
            # Histogram matching (faster)
            corrected_face = self._histogram_matching(swapped_face, reference_face)
        
        # Apply gamma correction
        corrected_face = self._adaptive_gamma_correction(corrected_face, reference_face)
        
        # Paste back
        result = img.copy()
        result[y1:y2, x1:x2] = corrected_face
        
        return result
    
    def _lab_color_transfer(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Color transfer in LAB space.
        Based on Reinhard et al. "Color Transfer between Images"
        """
        # Convert to LAB
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Calculate means and standard deviations
        source_mean, source_std = cv2.meanStdDev(source_lab)
        target_mean, target_std = cv2.meanStdDev(target_lab)
        
        # Reshape for broadcasting
        source_mean = source_mean.reshape(1, 1, 3)
        source_std = source_std.reshape(1, 1, 3)
        target_mean = target_mean.reshape(1, 1, 3)
        target_std = target_std.reshape(1, 1, 3)
        
        # Transfer color statistics
        result_lab = (source_lab - source_mean) * (target_std / (source_std + 1e-6)) + target_mean
        result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
        
        # Convert back to BGR
        result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def _histogram_matching(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Match histogram of source to target."""
        matched = np.zeros_like(source)
        
        for i in range(3):  # Process each channel
            matched[:, :, i] = self._match_channel_histogram(
                source[:, :, i], 
                target[:, :, i]
            )
        
        return matched
    
    def _match_channel_histogram(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Match single channel histogram."""
        # Calculate histograms
        s_values, s_counts = np.unique(source.ravel(), return_counts=True)
        t_values, t_counts = np.unique(target.ravel(), return_counts=True)
        
        # Calculate CDFs
        s_cdf = np.cumsum(s_counts).astype(np.float64)
        s_cdf /= s_cdf[-1]
        t_cdf = np.cumsum(t_counts).astype(np.float64)
        t_cdf /= t_cdf[-1]
        
        # Create mapping
        mapping = np.interp(s_cdf, t_cdf, t_values)
        
        # Apply mapping
        matched = np.interp(source.ravel(), s_values, mapping).reshape(source.shape)
        
        return matched.astype(np.uint8)
    
    def _adaptive_gamma_correction(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Adaptive gamma correction to match brightness."""
        # Calculate average brightness
        source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        
        source_brightness = np.mean(source_gray)
        target_brightness = np.mean(target_gray)
        
        if source_brightness == 0:
            return source
        
        # Calculate gamma
        gamma = np.log(target_brightness / 255.0) / np.log(source_brightness / 255.0)
        gamma = np.clip(gamma, 0.5, 2.0)  # Limit gamma range
        
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype(np.uint8)
        
        # Apply gamma correction
        corrected = cv2.LUT(source, table)
        
        return corrected
    
    def _apply_seamless_blend(self, img: np.ndarray, original: np.ndarray,
                             mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Apply Poisson blending for seamless integration."""
        x1, y1, x2, y2 = bbox
        
        # Ensure mask is binary
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Feather the mask edges
        feathered_mask = self._feather_mask(mask, kernel_size=15)
        
        # Calculate center point for Poisson blending
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        try:
            # Try Poisson blending (seamlessClone)
            result = cv2.seamlessClone(
                img, 
                original, 
                feathered_mask, 
                center, 
                cv2.MIXED_CLONE
            )
        except Exception as e:
            print(f"Poisson blending failed: {e}, using alpha blending")
            # Fallback to alpha blending
            result = self._alpha_blend(img, original, feathered_mask)
        
        return result
    
    def _feather_mask(self, mask: np.ndarray, kernel_size: int = 15) -> np.ndarray:
        """Create feathered (soft edge) mask."""
        # Gaussian blur for soft edges
        feathered = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        
        # Ensure it's 3-channel for seamlessClone
        if len(feathered.shape) == 2:
            feathered = cv2.cvtColor(feathered, cv2.COLOR_GRAY2BGR)
        
        return feathered
    
    def _alpha_blend(self, foreground: np.ndarray, background: np.ndarray, 
                     mask: np.ndarray) -> np.ndarray:
        """Alpha blending fallback."""
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        alpha = mask.astype(float) / 255.0
        alpha = np.stack([alpha] * 3, axis=-1)
        
        blended = (foreground * alpha + background * (1 - alpha)).astype(np.uint8)
        
        return blended
    
    def _apply_detail_restoration(self, img: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Restore fine details and sharpness."""
        x1, y1, x2, y2 = bbox
        face = img[y1:y2, x1:x2].copy()
        
        if self.quality_mode == 'fast':
            # Basic sharpening
            enhanced = self._unsharp_mask(face, amount=0.5)
        elif self.quality_mode == 'high_quality':
            # Medium sharpening + high-freq enhancement
            enhanced = self._unsharp_mask(face, amount=1.0)
            enhanced = self._enhance_high_frequency(enhanced, strength=0.3)
        else:  # ultra_realistic
            # Strong sharpening + detail enhancement
            enhanced = self._unsharp_mask(face, amount=1.5)
            enhanced = self._enhance_high_frequency(enhanced, strength=0.5)
            enhanced = self._enhance_skin_texture(enhanced)
        
        result = img.copy()
        result[y1:y2, x1:x2] = enhanced
        
        return result
    
    def _unsharp_mask(self, img: np.ndarray, amount: float = 1.0, 
                     radius: float = 2.0, threshold: int = 0) -> np.ndarray:
        """
        Unsharp masking for detail enhancement.
        
        Args:
            amount: Strength of sharpening (0.5-2.0)
            radius: Blur radius (1.0-3.0)
            threshold: Minimum brightness change to sharpen
        """
        # Create blurred version
        blurred = cv2.GaussianBlur(img, (0, 0), radius)
        
        # Calculate difference
        sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
        
        # Apply threshold
        if threshold > 0:
            low_contrast_mask = np.absolute(img - blurred) < threshold
            sharpened = np.where(low_contrast_mask, img, sharpened)
        
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def _enhance_high_frequency(self, img: np.ndarray, strength: float = 0.3) -> np.ndarray:
        """Enhance high-frequency details (fine textures)."""
        # Convert to float
        img_float = img.astype(np.float32) / 255.0
        
        # Extract low-frequency component
        low_freq = cv2.GaussianBlur(img_float, (0, 0), 3)
        
        # Extract high-frequency component
        high_freq = img_float - low_freq
        
        # Enhance high-frequency
        enhanced = img_float + high_freq * strength
        
        # Convert back
        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
        
        return enhanced
    
    def _enhance_skin_texture(self, face: np.ndarray) -> np.ndarray:
        """
        Enhance realistic skin texture using frequency separation.
        """
        # Convert to float
        img_float = face.astype(np.float32)
        
        # Separate into low and high frequency
        low_freq = cv2.GaussianBlur(img_float, (0, 0), 5)
        high_freq = img_float - low_freq
        
        # Enhance texture slightly
        enhanced_high = high_freq * 1.2
        
        # Recombine with subtle enhancement
        result = low_freq + enhanced_high
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _apply_noise_matching(self, img: np.ndarray, reference: np.ndarray,
                             bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Match noise pattern to avoid plastic/artificial look."""
        if self.quality_mode == 'fast':
            return img  # Skip for fast mode
        
        x1, y1, x2, y2 = bbox
        
        # Estimate noise in reference image
        noise_level = self._estimate_noise(reference[y1:y2, x1:x2])
        
        if noise_level > 1.0:  # Only add noise if original has noise
            # Add matching noise to swapped face
            face = img[y1:y2, x1:x2].copy()
            noisy_face = self._add_film_grain(face, noise_level)
            
            result = img.copy()
            result[y1:y2, x1:x2] = noisy_face
            return result
        
        return img
    
    def _estimate_noise(self, img: np.ndarray) -> float:
        """
        Estimate noise level in image.
        Uses Median Absolute Deviation (MAD) method.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Use Laplacian to detect edges
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Calculate MAD
        mad = np.median(np.abs(laplacian - np.median(laplacian)))
        
        # Estimate sigma (noise standard deviation)
        sigma = 1.4826 * mad
        
        return float(sigma)
    
    def _add_film_grain(self, img: np.ndarray, noise_level: float) -> np.ndarray:
        """Add realistic film grain noise."""
        # Generate Gaussian noise
        noise = np.random.normal(0, noise_level * 0.5, img.shape).astype(np.float32)
        
        # Add noise to image
        noisy = img.astype(np.float32) + noise
        
        # Clip and convert back
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        
        return noisy
    
    def get_face_mask(self, face_landmarks: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
        """
        Generate face mask from landmarks for blending.
        
        Args:
            face_landmarks: Face landmark points
            img_shape: (height, width) of image
            
        Returns:
            Binary mask of face region
        """
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        
        if face_landmarks is not None and len(face_landmarks) > 0:
            # Create convex hull around landmarks
            hull = cv2.convexHull(face_landmarks.astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 255)
        
        return mask


class EnhancementConfig:
    """Configuration presets for different quality modes."""
    
    FAST = {
        'super_resolution': False,
        'color_correction': 'histogram',
        'seamless_blend': False,
        'detail_restoration': 'basic',
        'noise_matching': False,
        'upscale_factor': 1
    }
    
    HIGH_QUALITY = {
        'super_resolution': True,
        'color_correction': 'histogram',
        'seamless_blend': True,
        'detail_restoration': 'medium',
        'noise_matching': False,
        'upscale_factor': 2
    }
    
    ULTRA_REALISTIC = {
        'super_resolution': True,
        'color_correction': 'lab',
        'seamless_blend': True,
        'detail_restoration': 'advanced',
        'noise_matching': True,
        'upscale_factor': 2
    }
    
    @classmethod
    def get_config(cls, mode: str) -> Dict:
        """Get configuration for quality mode."""
        configs = {
            'fast': cls.FAST,
            'high_quality': cls.HIGH_QUALITY,
            'ultra_realistic': cls.ULTRA_REALISTIC
        }
        return configs.get(mode, cls.HIGH_QUALITY)
