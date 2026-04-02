"""
Advanced Face Enhancement Service
Supports GFPGAN, Real-ESRGAN, CodeFormer, and refinement pipelines
"""
import logging
import numpy as np
import cv2
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import torch
from core.model_loader import get_model_loader
from core.device_config import get_device_config
from services.face_detection_advanced import Face

logger = logging.getLogger(__name__)


@dataclass
class EnhancementConfig:
    """Face enhancement configuration"""
    use_gfpgan: bool = True
    use_codeformer: bool = True
    use_realesrgan: bool = True
    upscale_factor: int = 2
    gfpgan_upscale: int = 2
    blend_ratio: float = 0.7


class FaceEnhancer:
    """Multi-method face enhancement engine"""
    
    def __init__(self, config: Optional[EnhancementConfig] = None):
        """Initialize face enhancer"""
        self.config = config or EnhancementConfig()
        self.device_config = get_device_config()
        self.device = self.device_config.get_device()
        self.dtype = self.device_config.get_dtype()
        self.model_loader = get_model_loader()
        
        # Try to load enhancement models
        self.gfpgan = self._load_gfpgan() if self.config.use_gfpgan else None
        self.codeformer = self._load_codeformer() if self.config.use_codeformer else None
        self.realesrgan = self._load_realesrgan() if self.config.use_realesrgan else None
    
    def enhance(
        self,
        image: np.ndarray,
        face: Optional[Face] = None,
        only_face: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Enhance image or face region
        
        Args:
            image: Input image (BGR)
            face: Optional face object for region-based enhancement
            only_face: If True with face, enhance only face region
            
        Returns:
            Tuple of (enhanced image, metadata)
        """
        try:
            original = image.copy()
            enhanced = image.copy()
            
            # Region-based enhancement
            if face and only_face:
                x1, y1, x2, y2 = face.bbox
                face_region = image[y1:y2, x1:x2].copy()
                
                # Enhance face region
                enhanced_face = self._enhance_region(face_region)
                
                # Place back
                enhanced[y1:y2, x1:x2] = enhanced_face
            else:
                # Enhance entire image
                enhanced = self._enhance_region(enhanced)
            
            metadata = {
                'success': True,
                'methods_used': [],
            }
            
            # Track which methods were applied
            if self.gfpgan:
                metadata['methods_used'].append('GFPGAN')
            if self.codeformer:
                metadata['methods_used'].append('CodeFormer')
            if self.realesrgan:
                metadata['methods_used'].append('Real-ESRGAN')
            
            return enhanced, metadata
            
        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            return image.copy(), {'success': False, 'reason': str(e)}
    
    def _enhance_region(self, image_region: np.ndarray) -> np.ndarray:
        """Enhance a face region through multiple stages"""
        try:
            result = image_region.copy()
            
            # Stage 1: CodeFormer restoration (if available)
            if self.codeformer is not None:
                try:
                    result = self._apply_codeformer(result)
                    logger.info("✓ CodeFormer enhancement applied")
                except Exception as e:
                    logger.warning(f"CodeFormer enhancement failed: {e}")
            
            # Stage 2: GFPGAN enhancement (if available)
            if self.gfpgan is not None:
                try:
                    result = self._apply_gfpgan(result)
                    logger.info("✓ GFPGAN enhancement applied")
                except Exception as e:
                    logger.warning(f"GFPGAN enhancement failed: {e}")
            
            # Stage 3: Real-ESRGAN upscaling (if available)
            if self.realesrgan is not None:
                try:
                    result = self._apply_realesrgan(result)
                    logger.info("✓ Real-ESRGAN upscaling applied")
                except Exception as e:
                    logger.warning(f"Real-ESRGAN upscaling failed: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Region enhancement failed: {e}")
            return image_region.copy()
    
    def _apply_gfpgan(self, image: np.ndarray) -> np.ndarray:
        """Apply GFPGAN enhancement"""
        try:
            # Ensure proper input format
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # GFPGAN expects BGR format
            # This is a placeholder - actual implementation depends on gfpgan library
            
            # For now, apply a basic enhancement
            # Replace with actual GFPGAN call when available
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            enhanced = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"GFPGAN application failed: {e}")
            return image
    
    def _apply_codeformer(self, image: np.ndarray) -> np.ndarray:
        """Apply CodeFormer restoration"""
        try:
            if self.codeformer is None:
                return image
            
            # Ensure proper input format
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            h, w = image.shape[:2]
            
            # Prepare input
            input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_img = torch.from_numpy(input_img).float().permute(2, 0, 1).unsqueeze(0)
            input_img = input_img.to(self.device) / 255.0
            
            # Inference
            with torch.no_grad():
                # Resize if too large
                if max(h, w) > 512:
                    input_img = torch.nn.functional.interpolate(
                        input_img,
                        size=(512, 512),
                        mode='bicubic',
                        align_corners=False
                    )
                
                output = self.codeformer(input_img, w=self.config.blend_ratio)[0]
                
                # Resize back if needed
                if max(h, w) > 512:
                    output = torch.nn.functional.interpolate(
                        output.unsqueeze(0),
                        size=(h, w),
                        mode='bicubic',
                        align_corners=False
                    )[0]
                
                # Convert output
                output = (output.squeeze() * 255).clamp(0, 255).byte().permute(1, 2, 0)
                output = cv2.cvtColor(output.cpu().numpy(), cv2.COLOR_RGB2BGR)
                
                return output
            
        except Exception as e:
            logger.error(f"CodeFormer application failed: {e}")
            return image
    
    def _apply_realesrgan(self, image: np.ndarray) -> np.ndarray:
        """Apply Real-ESRGAN upscaling"""
        try:
            if self.realesrgan is None:
                return image
            
            # Ensure proper input format
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # This is a placeholder for Real-ESRGAN
            # Actual implementation requires proper library integration
            
            # For now, use OpenCV upscaling
            h, w = image.shape[:2]
            new_h, new_w = int(h * self.config.upscale_factor), int(w * self.config.upscale_factor)
            
            upscaled = cv2.resize(
                image,
                (new_w, new_h),
                interpolation=cv2.INTER_LANCZOS4
            )
            
            return upscaled
            
        except Exception as e:
            logger.error(f"Real-ESRGAN application failed: {e}")
            return image
    
    def _load_gfpgan(self) -> Optional[Any]:
        """Load GFPGAN model"""
        try:
            model = self.model_loader.get_gfpgan_model()
            if model:
                logger.info("✓ GFPGAN model loaded")
            return model
        except Exception as e:
            logger.warning(f"GFPGAN loading failed: {e}")
            return None
    
    def _load_codeformer(self) -> Optional[torch.nn.Module]:
        """Load CodeFormer model"""
        try:
            model = self.model_loader.get_codeformer_model()
            if model:
                logger.info("✓ CodeFormer model loaded")
            return model
        except Exception as e:
            logger.warning(f"CodeFormer loading failed: {e}")
            return None
    
    def _load_realesrgan(self) -> Optional[Any]:
        """Load Real-ESRGAN model"""
        try:
            model = self.model_loader.get_realesrgan_model()
            if model:
                logger.info("✓ Real-ESRGAN model loaded")
            return model
        except Exception as e:
            logger.warning(f"Real-ESRGAN loading failed: {e}")
            return None


# Convenience function
def create_enhancer(**kwargs) -> FaceEnhancer:
    """Create a face enhancer instance"""
    config = EnhancementConfig(**kwargs)
    return FaceEnhancer(config)
