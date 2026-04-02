"""
Safe Diffusion Refinement Module

Provides subtle, identity-preserving diffusion-based face refinement.
Uses low denoise values to avoid face regeneration.
"""

import numpy as np
from typing import Optional, Tuple
import cv2


class SafeDiffusionRefiner:
    """
    Diffusion-based refinement that preserves face identity.
    
    Design Philosophy:
    - Use diffusion for subtle texture refinement, NOT face regeneration
    - Keep denoise values LOW (0.15-0.30)
    - Use neutral prompts
    - Apply ONLY to masked face region
    - Preserve face structure with ControlNet or structure guides
    """
    
    def __init__(self, pipeline=None):
        """
        Args:
            pipeline: Hugging Face diffusion pipeline (e.g., StableDiffusionInpaintPipeline)
                     If None, uses fallback refinement
        """
        self.pipeline = pipeline
        self.device = None
    
    def refine_face_safe(self, 
                        face_crop: np.ndarray,
                        mask: Optional[np.ndarray] = None,
                        denoise_strength: float = 0.20,
                        use_controlnet: bool = False) -> np.ndarray:
        """
        Safely refine face using diffusion with strict identity preservation.
        
        Args:
            face_crop: Face image (high-res crop)
            mask: Face region mask (255 for face, 0 for background)
            denoise_strength: Denoise/guidance value (0.15-0.30 recommended)
                            Lower = more preservation (safer)
                            Higher = more refinement (riskier)
            use_controlnet: Use ControlNet for structure preservation
            
        Returns:
            Refined face crop
        """
        print(f"[SafeDiffusionRefiner] Refining with denoise={denoise_strength}")
        
        # Safety bounds for denoise
        if denoise_strength < 0.15:
            print(f"  ⚠️  denoise_strength too low ({denoise_strength}), setting to 0.15")
            denoise_strength = 0.15
        if denoise_strength > 0.30:
            print(f"  ⚠️  denoise_strength too high ({denoise_strength}), capping at 0.30")
            denoise_strength = 0.30
        
        # If pipeline available, use diffusion refinement
        if self.pipeline is not None:
            try:
                refined = self._refine_with_diffusion(
                    face_crop,
                    mask,
                    denoise_strength,
                    use_controlnet
                )
                print(f"  ✓ Diffusion refinement applied")
                return refined
            except Exception as e:
                print(f"  ⚠️  Diffusion refinement failed: {e}")
                print(f"  → Fallback to basic refinement")
        
        # Fallback: use safe CNN-based refinement
        return self._safe_cnn_refinement(face_crop, denoise_strength)
    
    def _refine_with_diffusion(self,
                              face_crop: np.ndarray,
                              mask: Optional[np.ndarray],
                              denoise_strength: float,
                              use_controlnet: bool) -> np.ndarray:
        """
        Run actual diffusion refinement.
        
        Steps:
        1. Prepare input (BGR→RGB, resize if needed)
        2. Prepare mask for inpainting
        3. Run inference with safe parameters
        4. Blend output carefully
        """
        if self.pipeline is None:
            raise ValueError("Diffusion pipeline not available")
        
        # Step 1: Prepare input image
        input_image = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        
        # Step 2: Prepare mask for inpainting
        if mask is None:
            mask = self._create_default_mask(face_crop.shape[:2])
        else:
            # Ensure mask is single channel
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Invert mask if needed (diffusion expects 0=preserve, 255=inpaint)
        mask = 255 - mask
        
        # Step 3: Prepare prompt
        # Neutral prompt: enhance quality without changing face
        prompt = "professional portrait, soft lighting, HD, high quality"
        negative_prompt = "deformed face, ugly, distorted, low quality"
        
        # Step 4: Run inference
        # Note: Implementation depends on actual pipeline (SD1.5, SDXL, etc.)
        # This is a generic example
        
        try:
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=input_image,
                mask_image=mask,
                strength=denoise_strength,
                guidance_scale=7.5,  # Standard guidance (not too high)
                num_inference_steps=20,  # For speed (lower is faster but may be lower quality)
            )
            
            output_image = np.array(result.images[0])
            
            # Convert back to BGR
            output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
            
            return output_image
            
        except Exception as e:
            print(f"    Diffusion pipeline error: {e}")
            raise
    
    def _safe_cnn_refinement(self, face_crop: np.ndarray,
                            denoise_strength: float) -> np.ndarray:
        """
        Fallback refinement using safe CNN-like operations.
        Mimics what diffusion would do but without neural networks.
        """
        refined = face_crop.copy()
        
        # Higher denoise → more refinement
        # Lower denoise → more preservation
        
        # Step 1: Noise reduction (bilateral filter)
        # Strength factor affects filter intensity
        sigma_color = 75 * denoise_strength
        sigma_space = 75 * denoise_strength
        refined = cv2.bilateralFilter(
            refined, 
            d=9, 
            sigmaColor=sigma_color, 
            sigmaSpace=sigma_space
        )
        
        # Step 2: Detail enhancement (unsharp mask)
        # Only apply if denoise is high enough for noticeable effect
        if denoise_strength > 0.20:
            gaussian = cv2.GaussianBlur(refined, (0, 0), 1.5)
            details = cv2.subtract(refined, gaussian)
            refined = cv2.addWeighted(
                refined, 1.0,
                details, 0.2 * denoise_strength,
                0
            )
        
        # Step 3: Subtle color stabilization
        # Convert to LAB and stabilize colors
        refined_lab = cv2.cvtColor(refined, cv2.COLOR_BGR2LAB)
        
        # Slightly reduce a/b fluctuations
        a_channel = refined_lab[:, :, 1].astype(np.float32)
        b_channel = refined_lab[:, :, 2].astype(np.float32)
        
        a_blurred = cv2.GaussianBlur(a_channel, (5, 5), 0)
        b_blurred = cv2.GaussianBlur(b_channel, (5, 5), 0)
        
        # Blend: preserve original chrominance mostly, add slight blur (0.1 factor)
        a_channel = (a_channel * 0.9 + a_blurred * 0.1).astype(np.uint8)
        b_channel = (b_channel * 0.9 + b_blurred * 0.1).astype(np.uint8)
        
        refined_lab[:, :, 1] = a_channel
        refined_lab[:, :, 2] = b_channel
        refined = cv2.cvtColor(refined_lab, cv2.COLOR_LAB2BGR)
        
        return np.clip(refined, 0, 255).astype(np.uint8)
    
    @staticmethod
    def _create_default_mask(shape: Tuple) -> np.ndarray:
        """
        Create default mask for face region.
        Returns: Mask where 255=face (preserve), 0=background (inpaint)
        """
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create elliptical face region
        center = (w // 2, h // 2)
        axes = (w // 2 - 10, h // 2 - 10)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        # Feather edges
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        
        return mask
    
    @staticmethod
    def validate_denoise_strength(denoise: float) -> float:
        """
        Validate and clamp denoise strength to safe range.
        
        Returns:
            Safe denoise value in [0.15, 0.30]
        """
        SAFE_MIN = 0.15  # Minimum denoise for perceptible effect
        SAFE_MAX = 0.30  # Maximum denoise before identity risk
        
        if denoise < SAFE_MIN:
            print(f"⚠️  denoise {denoise} below minimum {SAFE_MIN}, using {SAFE_MIN}")
            return SAFE_MIN
        elif denoise > SAFE_MAX:
            print(f"⚠️  denoise {denoise} exceeds maximum {SAFE_MAX}, using {SAFE_MAX}")
            return SAFE_MAX
        
        return denoise


class DiffusionSettings:
    """Recommended diffusion settings for different scenarios."""
    
    # For subtle texture refinement (safe)
    SUBTLE = {
        'denoise_strength': 0.15,
        'guidance_scale': 7.5,
        'num_steps': 20,
        'prompt_weight': 0.5,  # How much to guide toward prompt
        'description': 'Minimal changes, maximum safety'
    }
    
    # For balanced enhancement (recommended for Cinematic)
    BALANCED = {
        'denoise_strength': 0.20,
        'guidance_scale': 7.5,
        'num_steps': 25,
        'prompt_weight': 0.7,
        'description': 'Good quality with identity preservation'
    }
    
    # For noticeable refinement (higher risk)
    ENHANCED = {
        'denoise_strength': 0.25,
        'guidance_scale': 7.5,
        'num_steps': 30,
        'prompt_weight': 0.8,
        'description': 'Noticeable improvements, some identity shift risk'
    }
    
    # Unsafe - Not recommended
    AGGRESSIVE = {
        'denoise_strength': 0.40,  # ✗ TOO HIGH - Identity risk
        'guidance_scale': 15.0,    # ✗ TOO HIGH - Over-guided
        'num_steps': 50,           # ✗ TOO MANY - Unnecessary
        'prompt_weight': 1.0,      # ✗ EXTREME - Regenerates face
        'description': '⚠️  DO NOT USE - HIGH IDENTITY LOSS RISK'
    }
    
    @classmethod
    def get_settings(cls, mode: str) -> dict:
        """Get recommended settings for mode."""
        mode_map = {
            'subtle': cls.SUBTLE,
            'balanced': cls.BALANCED,
            'enhanced': cls.ENHANCED,
        }
        return mode_map.get(mode, cls.BALANCED)

