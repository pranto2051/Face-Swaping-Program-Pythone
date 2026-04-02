"""
Processing Mode Controller
Orchestrates Fast, Studio, and Cinematic modes with different quality levels
"""
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Available processing modes"""
    FAST = 'fast'
    STUDIO = 'studio'
    CINEMATIC = 'cinematic'


@dataclass
class ModeConfig:
    """Configuration for each processing mode"""
    mode: ProcessingMode
    
    # Face detection
    detection_confidence: float
    detection_model: str
    align_faces: bool
    
    # Face swap method
    swap_method: str  # 'insightface', 'deepfacelab', 'hybrid'
    preserve_expression: bool
    identity_strength: float
    
    # Refinement
    apply_refinement: bool
    refinement_blend: float
    denoise_strength: float
    color_correction: bool
    lighting_correction: bool
    
    # Enhancement
    apply_enhancement: bool
    use_gfpgan: bool
    use_codeformer: bool
    use_realesrgan: bool
    enhancement_upscale: int
    
    # Performance
    max_face_size: int
    batch_processing: bool
    gpu_acceleration: bool
    
    # Processing time expectations
    estimated_time_single_image: str  # e.g., "2-3 seconds"
    estimated_time_video_minute: str  # e.g., "2-3 minutes per minute of video"


class ModeController:
    """Controls processing modes and configuration"""
    
    # Predefined mode configurations
    MODES: Dict[ProcessingMode, ModeConfig] = {
        ProcessingMode.FAST: ModeConfig(
            mode=ProcessingMode.FAST,
            
            # Detection: Fast model, low confidence
            detection_confidence=0.4,
            detection_model='buffalo_m',
            align_faces=False,
            
            # Swap: InsightFace only
            swap_method='insightface',
            preserve_expression=False,
            identity_strength=1.0,
            
            # No refinement
            apply_refinement=False,
            refinement_blend=0.5,
            denoise_strength=0.0,
            color_correction=False,
            lighting_correction=False,
            
            # No enhancement
            apply_enhancement=False,
            use_gfpgan=False,
            use_codeformer=False,
            use_realesrgan=False,
            enhancement_upscale=1,
            
            # Performance
            max_face_size=512,
            batch_processing=False,
            gpu_acceleration=True,
            
            estimated_time_single_image='<1 second',
            estimated_time_video_minute='<1 minute per minute'
        ),
        
        ProcessingMode.STUDIO: ModeConfig(
            mode=ProcessingMode.STUDIO,
            
            # Detection: Balanced
            detection_confidence=0.6,
            detection_model='buffalo_l',
            align_faces=True,
            
            # Swap: InsightFace with expression preservation
            swap_method='insightface',
            preserve_expression=True,
            identity_strength=0.9,
            
            # Moderate refinement
            apply_refinement=True,
            refinement_blend=0.6,
            denoise_strength=0.2,
            color_correction=True,
            lighting_correction=True,
            
            # GFPGAN enhancement
            apply_enhancement=True,
            use_gfpgan=True,
            use_codeformer=False,
            use_realesrgan=False,
            enhancement_upscale=1,
            
            # Performance
            max_face_size=768,
            batch_processing=False,
            gpu_acceleration=True,
            
            estimated_time_single_image='3-5 seconds',
            estimated_time_video_minute='3-5 minutes per minute'
        ),
        
        ProcessingMode.CINEMATIC: ModeConfig(
            mode=ProcessingMode.CINEMATIC,
            
            # Detection: High confidence
            detection_confidence=0.8,
            detection_model='buffalo_l',
            align_faces=True,
            
            # Swap: Hybrid method with full expression
            swap_method='hybrid',
            preserve_expression=True,
            identity_strength=0.95,
            
            # Full refinement
            apply_refinement=True,
            refinement_blend=0.7,
            denoise_strength=0.4,
            color_correction=True,
            lighting_correction=True,
            
            # Full enhancement pipeline
            apply_enhancement=True,
            use_gfpgan=True,
            use_codeformer=True,
            use_realesrgan=True,
            enhancement_upscale=2,
            
            # Performance
            max_face_size=1024,
            batch_processing=True,
            gpu_acceleration=True,
            
            estimated_time_single_image='10-20 seconds',
            estimated_time_video_minute='10-20 minutes per minute'
        ),
    }
    
    @classmethod
    def get_mode_config(cls, mode: ProcessingMode) -> ModeConfig:
        """Get configuration for a processing mode"""
        return cls.MODES[mode]
    
    @classmethod
    def get_mode_by_name(cls, mode_name: str) -> Optional[ProcessingMode]:
        """Get mode by name string"""
        try:
            return ProcessingMode(mode_name.lower())
        except ValueError:
            logger.warning(f"Unknown mode: {mode_name}")
            return None
    
    @classmethod
    def get_mode_summary(cls, mode: ProcessingMode) -> Dict[str, Any]:
        """Get human-readable summary of mode"""
        config = cls.get_mode_config(mode)
        
        return {
            'mode': mode.value,
            'quality_level': cls._get_quality_label(mode),
            'processing_speed': cls._get_speed_label(mode),
            'estimated_time_image': config.estimated_time_single_image,
            'estimated_time_video': config.estimated_time_video_minute,
            'features': {
                'face_alignment': config.align_faces,
                'expression_preservation': config.preserve_expression,
                'color_correction': config.color_correction,
                'gfpgan_enhancement': config.use_gfpgan,
                'codeformer_enhancement': config.use_codeformer,
                'upscaling': config.use_realesrgan,
            },
            'description': cls._get_mode_description(mode),
        }
    
    @classmethod
    def _get_quality_label(cls, mode: ProcessingMode) -> str:
        """Get quality label for mode"""
        if mode == ProcessingMode.FAST:
            return 'Good'
        elif mode == ProcessingMode.STUDIO:
            return 'Very Good'
        else:
            return 'Excellent'
    
    @classmethod
    def _get_speed_label(cls, mode: ProcessingMode) -> str:
        """Get speed label for mode"""
        if mode == ProcessingMode.FAST:
            return 'Fastest'
        elif mode == ProcessingMode.STUDIO:
            return 'Moderate'
        else:
            return 'Slowest'
    
    @classmethod
    def _get_mode_description(cls, mode: ProcessingMode) -> str:
        """Get description for mode"""
        descriptions = {
            ProcessingMode.FAST: (
                'Quick processing using InsightFace swap. Best for previews and batch processing. '
                'Lower quality but very fast.'
            ),
            ProcessingMode.STUDIO: (
                'Balanced quality and speed. Uses InsightFace with expression preservation, '
                'GFPGAN enhancement, and color correction. Best for general use.'
            ),
            ProcessingMode.CINEMATIC: (
                'Maximum quality output. Uses hybrid swap with full enhancement pipeline including '
                'DeepFake refinement, CodeFormer restoration, and Real-ESRGAN upscaling. '
                'Best for final output and high-quality results.'
            ),
        }
        return descriptions.get(mode, 'Unknown mode')
    
    @classmethod
    def validate_config(cls, mode: ProcessingMode, custom_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and merge custom configuration with mode defaults
        
        Args:
            mode: Processing mode
            custom_config: Custom configuration overrides
            
        Returns:
            Merged and validated configuration
        """
        base_config = cls.get_mode_config(mode)
        
        # Create mutable copy
        config_dict = vars(base_config).copy()
        
        # Apply custom overrides
        for key, value in custom_config.items():
            if key in config_dict:
                config_dict[key] = value
            else:
                logger.warning(f"Unknown config key: {key}")
        
        return config_dict
    
    @classmethod
    def print_mode_info(cls):
        """Print information about all available modes"""
        print("\n" + "="*70)
        print("FACE SWAP PROCESSING MODES".center(70))
        print("="*70)
        
        for mode in ProcessingMode:
            summary = cls.get_mode_summary(mode)
            
            print(f"\n├─ {mode.value.upper()}")
            print(f"│  Quality:     {summary['quality_level']}")
            print(f"│  Speed:       {summary['processing_speed']}")
            print(f"│  Image Time:  {summary['estimated_time_image']}")
            print(f"│  Video Time:  {summary['estimated_time_video']}")
            print(f"│  Description: {summary['description']}")
            print(f"│  Features:")
            
            features = summary['features']
            for i, (feature, enabled) in enumerate(features.items()):
                is_last = i == len(features) - 1
                prefix = "└─" if is_last else "├─"
                status = "✓" if enabled else "✗"
                print(f"│    {prefix} {feature}: {status}")
        
        print("\n" + "="*70 + "\n")


# Convenience function
def get_mode_controller() -> type:
    """Get mode controller class"""
    return ModeController
