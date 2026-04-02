"""
Hybrid Face Swap Pipeline
Orchestrates the complete multi-stage face swapping process
"""
import logging
import numpy as np
import cv2
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import time
from core.device_config import get_device_config
from core.mode_controller import ModeController, ProcessingMode, ModeConfig
from services.face_detection_advanced import FaceDetector
from services.face_swap_hybrid import FaceSwapper, SwapConfig
from services.face_enhancer_advanced import FaceEnhancer, EnhancementConfig
from services.deepfake_refinement import DeepFakeRefinement, RefinementConfig

logger = logging.getLogger(__name__)


class HybridFaceSwapPipeline:
    """Advanced pipeline for face swapping with multiple processing stages"""
    
    def __init__(self, mode: ProcessingMode = ProcessingMode.STUDIO):
        """
        Initialize pipeline with processing mode
        
        Args:
            mode: Processing mode (FAST, STUDIO, CINEMATIC)
        """
        self.mode = mode
        self.mode_config = ModeController.get_mode_config(mode)
        self.device_config = get_device_config()
        
        # Initialize components
        self.detector = FaceDetector(self.mode_config.detection_model)
        self.swapper = self._create_swapper()
        self.enhancer = self._create_enhancer()
        self.refiner = self._create_refiner()
        
        logger.info(f"✓ Pipeline initialized in {mode.value} mode")
    
    def _create_swapper(self) -> FaceSwapper:
        """Create face swapper with mode config"""
        swap_config = SwapConfig(
            method=self.mode_config.swap_method,
            blend_mode='seamless',
            preserve_expression=self.mode_config.preserve_expression,
            identity_strength=self.mode_config.identity_strength,
        )
        return FaceSwapper(swap_config)
    
    def _create_enhancer(self) -> Optional[FaceEnhancer]:
        """Create face enhancer with mode config"""
        if not self.mode_config.apply_enhancement:
            return None
        
        enhancement_config = EnhancementConfig(
            use_gfpgan=self.mode_config.use_gfpgan,
            use_codeformer=self.mode_config.use_codeformer,
            use_realesrgan=self.mode_config.use_realesrgan,
            upscale_factor=self.mode_config.enhancement_upscale,
            blend_ratio=0.7,
        )
        return FaceEnhancer(enhancement_config)
    
    def _create_refiner(self) -> Optional[DeepFakeRefinement]:
        """Create refinement engine with mode config"""
        if not self.mode_config.apply_refinement:
            return None
        
        refinement_config = RefinementConfig(
            blend_ratio=self.mode_config.refinement_blend,
            denoise_strength=self.mode_config.denoise_strength,
            apply_color_correction=self.mode_config.color_correction,
            apply_lighting_correction=self.mode_config.lighting_correction,
        )
        return DeepFakeRefinement(refinement_config)
    
    def process(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
        source_face_id: Optional[int] = None,
        target_face_id: Optional[int] = None,
        callback: Optional[callable] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Complete face swap pipeline
        
        Args:
            source_image: Source image with face to swap
            target_image: Target image to receive face
            source_face_id: Specific source face to use (None = auto-select)
            target_face_id: Specific target face to use (None = auto-select)
            callback: Progress callback function
            
        Returns:
            Tuple of (swapped image, metadata)
        """
        start_time = time.time()
        metadata = {
            'mode': self.mode.value,
            'stages': {},
            'total_time': 0,
        }
        
        try:
            # Stage 1: Face Detection
            logger.info("→ Stage 1: Face Detection")
            detection_start = time.time()
            
            source_faces, source_annotated = self.detector.detect_faces(
                source_image,
                self.mode_config.detection_confidence
            )
            target_faces, target_annotated = self.detector.detect_faces(
                target_image,
                self.mode_config.detection_confidence
            )
            
            if not source_faces or not target_faces:
                logger.error("No faces detected in source or target")
                return target_image.copy(), {
                    'success': False,
                    'reason': 'No faces detected'
                }
            
            metadata['stages']['detection'] = {
                'time': time.time() - detection_start,
                'source_faces': len(source_faces),
                'target_faces': len(target_faces),
            }
            
            if callback:
                callback(20, "Face detection completed")
            
            # Stage 2: Face Selection
            logger.info("→ Stage 2: Face Selection")
            source_face = self.detector.select_face(source_faces, source_face_id)
            target_face = self.detector.select_face(target_faces, target_face_id)
            
            if not source_face or not target_face:
                return target_image.copy(), {
                    'success': False,
                    'reason': 'Failed to select faces'
                }
            
            metadata['stages']['selection'] = {
                'source_face_id': source_face.id,
                'target_face_id': target_face.id,
            }
            
            if callback:
                callback(30, "Faces selected")
            
            # Stage 3: Face Swap
            logger.info("→ Stage 3: Face Swap")
            swap_start = time.time()
            
            swapped_image, swap_metadata = self.swapper.swap_faces(
                source_image,
                target_image,
                source_face,
                target_face,
            )
            
            if not swap_metadata.get('success'):
                return target_image.copy(), {
                    'success': False,
                    'reason': 'Face swap failed'
                }
            
            metadata['stages']['swap'] = {
                'time': time.time() - swap_start,
                'method': swap_metadata.get('method'),
            }
            
            if callback:
                callback(50, "Face swap completed")
            
            # Stage 4: Refinement (if enabled)
            if self.refiner:
                logger.info("→ Stage 4: Refinement")
                refine_start = time.time()
                
                refined_image, refine_metadata = self.refiner.refine(
                    target_image,
                    swapped_image,
                )
                
                metadata['stages']['refinement'] = {
                    'time': time.time() - refine_start,
                    'applied': refine_metadata.get('refinements_applied', []),
                }
                
                swapped_image = refined_image
                
                if callback:
                    callback(65, "Refinement completed")
            
            # Stage 5: Enhancement (if enabled)
            if self.enhancer:
                logger.info("→ Stage 5: Enhancement")
                enhance_start = time.time()
                
                enhanced_image, enhance_metadata = self.enhancer.enhance(
                    swapped_image,
                    face=target_face,
                    only_face=True,
                )
                
                metadata['stages']['enhancement'] = {
                    'time': time.time() - enhance_start,
                    'methods': enhance_metadata.get('methods_used', []),
                }
                
                swapped_image = enhanced_image
                
                if callback:
                    callback(85, "Enhancement completed")
            
            # Final output
            logger.info("✓ Pipeline completed successfully")
            
            metadata['success'] = True
            metadata['total_time'] = time.time() - start_time
            
            if callback:
                callback(100, "Processing completed")
            
            return swapped_image, metadata
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return target_image.copy(), {
                'success': False,
                'reason': f'Pipeline error: {str(e)}'
            }
    
    def process_batch(
        self,
        source_images: List[np.ndarray],
        target_image: np.ndarray,
        callback: Optional[callable] = None,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Process multiple source images with same target
        
        Args:
            source_images: List of source images
            target_image: Target image
            callback: Progress callback
            
        Returns:
            Tuple of (list of swapped images, metadata)
        """
        results = []
        metadata_list = []
        
        for idx, source_image in enumerate(source_images):
            progress = int((idx / len(source_images)) * 100)
            if callback:
                callback(progress, f"Processing image {idx + 1}/{len(source_images)}")
            
            result, meta = self.process(source_image, target_image, callback)
            results.append(result)
            metadata_list.append(meta)
        
        return results, {
            'total_images': len(source_images),
            'successful': sum(1 for m in metadata_list if m.get('success')),
            'metadata': metadata_list,
        }
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline configuration information"""
        return {
            'mode': self.mode.value,
            'config': vars(self.mode_config),
            'device': str(self.device_config.get_device()),
            'device_info': self.device_config.processor_info,
        }
    
    def change_mode(self, new_mode: ProcessingMode) -> None:
        """Change processing mode"""
        logger.info(f"Changing mode from {self.mode.value} to {new_mode.value}")
        self.mode = new_mode
        self.mode_config = ModeController.get_mode_config(new_mode)
        
        # Recreate components with new configuration
        self.swapper = self._create_swapper()
        self.enhancer = self._create_enhancer()
        self.refiner = self._create_refiner()


# Convenience function
def create_pipeline(mode: ProcessingMode = ProcessingMode.STUDIO) -> HybridFaceSwapPipeline:
    """Create a face swap pipeline instance"""
    return HybridFaceSwapPipeline(mode)
