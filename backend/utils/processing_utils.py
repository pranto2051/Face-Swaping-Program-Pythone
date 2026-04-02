"""
Utility Functions and Helpers
Common utilities for face swap pipeline
"""
import logging
import numpy as np
import cv2
from typing import Tuple, List
from pathlib import Path

logger = logging.getLogger(__name__)


class ImageUtils:
    """Image processing utilities"""
    
    @staticmethod
    def resize_image(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
        """Resize image to max size while maintaining aspect ratio"""
        h, w = image.shape[:2]
        
        if max(h, w) <= max_size:
            return image
        
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    @staticmethod
    def normalize_image(image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Normalize image to [0, 1] range"""
        dtype = image.dtype
        
        if image.dtype == np.uint8:
            normalized = image.astype(np.float32) / 255.0
        elif image.dtype == np.float64:
            normalized = image.astype(np.float32)
        else:
            normalized = image.astype(np.float32) / 255.0
        
        return normalized, {'original_dtype': dtype, 'max_value': 255}
    
    @staticmethod
    def denormalize_image(image: np.ndarray, normalized_info: dict) -> np.ndarray:
        """Denormalize image back to original range"""
        original_dtype = normalized_info.get('original_dtype', np.uint8)
        
        if original_dtype == np.uint8:
            denormalized = np.clip(image * 255, 0, 255).astype(np.uint8)
        else:
            denormalized = image.astype(original_dtype)
        
        return denormalized
    
    @staticmethod
    def pad_image(image: np.ndarray, target_size: int) -> Tuple[np.ndarray, dict]:
        """Pad image to target size"""
        h, w = image.shape[:2]
        
        if h >= target_size and w >= target_size:
            return image, {'padded': False}
        
        pad_h = max(0, target_size - h)
        pad_w = max(0, target_size - w)
        
        pad_top = pad_h // 2
        pad_bot = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        padded = cv2.copyMakeBorder(
            image,
            pad_top, pad_bot, pad_left, pad_right,
            cv2.BORDER_REFLECT_101
        )
        
        return padded, {
            'padded': True,
            'pad_top': pad_top,
            'pad_bot': pad_bot,
            'pad_left': pad_left,
            'pad_right': pad_right,
            'original_shape': image.shape,
        }
    
    @staticmethod
    def unpad_image(image: np.ndarray, pad_info: dict) -> np.ndarray:
        """Remove padding from image"""
        if not pad_info.get('padded'):
            return image
        
        h, w = pad_info['original_shape'][:2]
        
        pad_top = pad_info['pad_top']
        pad_bot = pad_info['pad_bot']
        pad_left = pad_info['pad_left']
        pad_right = pad_info['pad_right']
        
        h_end = image.shape[0] - pad_bot
        w_end = image.shape[1] - pad_right
        
        return image[pad_top:h_end, pad_left:w_end]
    
    @staticmethod
    def apply_color_space_conversion(image: np.ndarray, from_space: str, to_space: str) -> np.ndarray:
        """Convert between color spaces"""
        conversions = {
            ('BGR', 'RGB'): cv2.COLOR_BGR2RGB,
            ('RGB', 'BGR'): cv2.COLOR_RGB2BGR,
            ('BGR', 'LAB'): cv2.COLOR_BGR2LAB,
            ('LAB', 'BGR'): cv2.COLOR_LAB2BGR,
            ('BGR', 'HSV'): cv2.COLOR_BGR2HSV,
            ('HSV', 'BGR'): cv2.COLOR_HSV2BGR,
            ('BGR', 'GRAY'): cv2.COLOR_BGR2GRAY,
            ('GRAY', 'BGR'): cv2.COLOR_GRAY2BGR,
        }
        
        key = (from_space.upper(), to_space.upper())
        if key in conversions:
            return cv2.cvtColor(image, conversions[key])
        
        raise ValueError(f"Unsupported conversion: {from_space} -> {to_space}")


class FileUtils:
    """File handling utilities"""
    
    @staticmethod
    def get_safe_path(base_dir: str, filename: str) -> Path:
        """Get safe file path preventing directory traversal"""
        from werkzeug.utils import secure_filename
        
        safe_name = secure_filename(filename)
        safe_path = (Path(base_dir) / safe_name).resolve()
        
        if not str(safe_path).startswith(str(Path(base_dir).resolve())):
            raise ValueError("Invalid file path")
        
        return safe_path
    
    @staticmethod
    def get_file_size(file_path: str) -> int:
        """Get file size in bytes"""
        return Path(file_path).stat().st_size
    
    @staticmethod
    def cleanup_directory(directory: str, keep_files: List[str] = None) -> None:
        """Clean up directory except for specified files"""
        keep_files = keep_files or []
        
        try:
            for file_path in Path(directory).iterdir():
                if file_path.name not in keep_files:
                    if file_path.is_file():
                        file_path.unlink()
                    elif file_path.is_dir():
                        import shutil
                        shutil.rmtree(file_path)
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")


class MetricsUtils:
    """Performance metrics utilities"""
    
    @staticmethod
    def calculate_image_similarity(image1: np.ndarray, image2: np.ndarray) -> float:
        """Calculate similarity between two images (0-1)"""
        if image1.shape != image2.shape:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        
        # Mean Squared Error
        mse = np.mean((image1.astype(np.float32) - image2.astype(np.float32)) ** 2)
        
        # Convert to similarity (0-1)
        max_mse = 255 ** 2
        similarity = 1.0 - (mse / max_mse)
        
        return float(np.clip(similarity, 0, 1))
    
    @staticmethod
    def estimate_processing_time(image_size: Tuple[int, int], mode: str) -> float:
        """Estimate processing time in seconds"""
        h, w = image_size
        pixels = h * w
        
        # Base times (in seconds)
        base_times = {
            'fast': 0.5,
            'studio': 3.0,
            'cinematic': 10.0,
        }
        
        base_time = base_times.get(mode.lower(), 3.0)
        
        # Adjust for image size
        reference_pixels = 512 * 512
        size_multiplier = pixels / reference_pixels
        
        estimated_time = base_time * size_multiplier
        
        return estimated_time
    
    @staticmethod
    def calculate_quality_score(image: np.ndarray) -> float:
        """Estimate image quality (0-100)"""
        # Laplacian variance (focus quality)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to 0-100
        quality = min(100, (laplacian_var / 1000.0) * 100)
        
        return float(quality)


class ValidationUtils:
    """Input validation utilities"""
    
    @staticmethod
    def validate_image(image: np.ndarray, min_size: int = 64, max_size: int = 4096) -> Tuple[bool, str]:
        """Validate image dimensions and format"""
        if image is None or image.size == 0:
            return False, "Image is empty"
        
        h, w = image.shape[:2]
        
        if h < min_size or w < min_size:
            return False, f"Image too small: {w}x{h} (minimum: {min_size}x{min_size})"
        
        if h > max_size or w > max_size:
            return False, f"Image too large: {w}x{h} (maximum: {max_size}x{max_size})"
        
        if len(image.shape) not in [2, 3]:
            return False, f"Invalid image format: {len(image.shape)} channels"
        
        return True, "Valid"
    
    @staticmethod
    def validate_faces(faces: list, expected_count: int = None) -> Tuple[bool, str]:
        """Validate face detection results"""
        if not faces:
            return False, "No faces detected"
        
        if expected_count and len(faces) != expected_count:
            return False, f"Expected {expected_count} faces, found {len(faces)}"
        
        return True, f"{len(faces)} faces detected"
    
    @staticmethod
    def validate_mode(mode: str, available_modes: list) -> Tuple[bool, str]:
        """Validate processing mode"""
        if mode not in available_modes:
            return False, f"Invalid mode: {mode}. Available: {', '.join(available_modes)}"
        
        return True, f"Valid mode: {mode}"


# Logging utilities
def setup_logging(name: str, level: str = 'INFO') -> logging.Logger:
    """Setup logging for a module"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, level.upper()))
    
    return logger
