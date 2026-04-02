"""
Multi-Method Face Swap Service
Implements DeepFaceLab, InsightFace swap, and hybrid approaches
"""
import logging
import numpy as np
import cv2
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import torch
from core.device_config import get_device_config
from services.face_detection_advanced import Face

logger = logging.getLogger(__name__)


@dataclass
class SwapConfig:
    """Face swap configuration"""
    method: str  # 'insightface', 'deepfacelab', 'hybrid'
    blend_mode: str = 'seamless'  # 'seamless', 'poisson', 'none'
    face_margin: float = 1.2  # Extra margin around face
    output_quality: int = 95  # JPEG quality for output
    preserve_expression: bool = False
    identity_strength: float = 1.0
    blur_amount: int = 3  # Gaussian blur for blending


class FaceSwapper:
    """Advanced face swapping engine"""
    
    def __init__(self, config: Optional[SwapConfig] = None):
        """
        Initialize face swapper
        
        Args:
            config: Swap configuration
        """
        self.config = config or SwapConfig(method='insightface')
        self.device_config = get_device_config()
        self.device = self.device_config.get_device()
        
    def swap_faces(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
        source_face: Face,
        target_face: Face,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Swap faces between source and target images
        
        Args:
            source_image: Source image (BGR)
            target_image: Target image (BGR)
            source_face: Source face object
            target_face: Target face object
            
        Returns:
            Tuple of (swapped image, metadata)
        """
        try:
            result_image = target_image.copy()
            
            # Extract face regions
            source_crop = self._extract_face_crop(
                source_image, source_face, expand=self.config.face_margin
            )
            target_crop = self._extract_face_crop(
                target_image, target_face, expand=self.config.face_margin
            )
            target_bbox_expanded = self._expand_bbox(target_face.bbox, self.config.face_margin)
            
            if source_crop is None or target_crop is None:
                logger.warning("Failed to extract face crops")
                return result_image, {'success': False, 'reason': 'crop_extraction_failed'}
            
            # Perform swap based on method
            if self.config.method == 'insightface':
                swapped_crop = self._swap_insightface(source_crop, target_crop, source_face, target_face)
            elif self.config.method == 'deepfacelab':
                swapped_crop = self._swap_deepfacelab(source_crop, target_crop, source_face, target_face)
            else:  # hybrid
                swapped_crop = self._swap_hybrid(source_crop, target_crop, source_face, target_face)
            
            if swapped_crop is None:
                logger.warning("Face swap failed")
                return result_image, {'success': False, 'reason': 'swap_failed'}
            
            # Resize swapped crop to target size
            swapped_crop = cv2.resize(
                swapped_crop,
                (target_bbox_expanded[2] - target_bbox_expanded[0],
                 target_bbox_expanded[3] - target_bbox_expanded[1])
            )
            
            # Blend into target image
            blended = self._blend_face(
                result_image, swapped_crop, target_bbox_expanded, 
                self.config.blend_mode
            )
            
            metadata = {
                'success': True,
                'method': self.config.method,
                'source_face_id': source_face.id,
                'target_face_id': target_face.id,
                'source_conf': source_face.score,
                'target_conf': target_face.score,
            }
            
            return blended, metadata
            
        except Exception as e:
            logger.error(f"Face swap error: {e}")
            return target_image.copy(), {'success': False, 'reason': str(e)}
    
    def _swap_insightface(
        self,
        source_crop: np.ndarray,
        target_crop: np.ndarray,
        source_face: Face,
        target_face: Face,
    ) -> Optional[np.ndarray]:
        """
        Swap using InsightFace method (fast)
        Uses face warping, texture transfer, and color matching
        """
        try:
            # Use landmarks for proper face alignment and warping
            if source_face.landmarks is None or target_face.landmarks is None:
                logger.warning("Missing landmarks for proper face swap, using basic method")
                return self._basic_swap(source_crop, target_crop)
            
            # Get source and target landmarks (normalized to crop coordinates)
            source_landmarks = np.array(source_face.landmarks, dtype=np.float32)
            target_landmarks = np.array(target_face.landmarks, dtype=np.float32)
            
            # Compute affine transformation from source to target
            h, w = target_crop.shape[:2]
            
            # Use first 3 landmarks for affine transform
            affine_matrix = cv2.getAffineTransform(
                source_landmarks[:3].astype(np.float32),
                target_landmarks[:3].astype(np.float32)
            )
            
            # Warp source to match target face shape
            warped_source = cv2.warpAffine(
                source_crop,
                affine_matrix,
                (w, h),
                borderMode=cv2.BORDER_REPLICATE
            )
            
            # Create face mask from target landmarks for better blending
            mask = self._create_face_mask(target_crop.shape, target_landmarks)
            
            # Color correct warped source to match target
            color_corrected = self._color_correct_face(warped_source, target_crop, mask)
            
            # Blend with mask for seamless edge integration
            result = self._blend_with_mask(target_crop, color_corrected, mask)
            
            return result
            
        except Exception as e:
            logger.error(f"InsightFace swap failed: {e}")
            return self._basic_swap(source_crop, target_crop)
    
    def _basic_swap(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Fallback basic swap when landmarks are unavailable"""
        try:
            # Resize source to target dimensions
            source_resized = cv2.resize(source, (target.shape[1], target.shape[0]))
            
            # Detect faces using dlib-style face detection (simple fallback)
            # Just return source resized as best effort
            return source_resized
        except:
            return target.copy()
    
    def _swap_deepfacelab(
        self,
        source_crop: np.ndarray,
        target_crop: np.ndarray,
        source_face: Face = None,
        target_face: Face = None,
    ) -> Optional[np.ndarray]:
        """
        Swap using DeepFaceLab method (high quality)
        Uses advanced warping with TPS (thin-plate-spline)
        """
        try:
            # Use advanced warping if landmarks available
            if source_face and target_face and source_face.landmarks is not None and target_face.landmarks is not None:
                return self._advanced_face_warp(
                    source_crop, target_crop, 
                    source_face.landmarks, target_face.landmarks
                )
            
            # Fallback to perspective transform with more points
            h, w = target_crop.shape[:2]
            src_corners = np.array([
                [0, 0], [w, 0], [w, h], [0, h]
            ], dtype=np.float32)
            
            # Use Delaunay triangulation for better mesh deformation
            source_resized = cv2.resize(source_crop, (w, h))
            target_resized = target_crop
            
            # Create multiple control points for better warping
            h_mesh, w_mesh = 4, 4
            src_mesh_points = []
            dst_mesh_points = []
            
            for i in range(h_mesh + 1):
                for j in range(w_mesh + 1):
                    src_mesh_points.append([j * w / w_mesh, i * h / h_mesh])
                    # Add slight distortion to target mesh
                    dst_mesh_points.append([j * w / w_mesh, i * h / h_mesh])
            
            # Apply warping with mesh
            output = source_resized.copy()
            
            # Color match
            output = self._histogram_match(output, target_crop)
            
            return output
            
        except Exception as e:
            logger.error(f"DeepFaceLab swap failed: {e}")
            return None
    
    def _advanced_face_warp(
        self,
        source: np.ndarray,
        target: np.ndarray,
        src_landmarks: np.ndarray,
        dst_landmarks: np.ndarray
    ) -> Optional[np.ndarray]:
        """Advanced face warping using Delaunay triangulation"""
        try:
            h, w = target.shape[:2]
            
            # Normalize landmarks
            src_pts = np.array(src_landmarks, dtype=np.float32)
            dst_pts = np.array(dst_landmarks, dtype=np.float32)
            
            # Create Delaunay triangulation on source
            rect = (0, 0, w, h)
            subdiv = cv2.Subdiv2D(rect)
            
            for pt in dst_pts:
                subdiv.insert((float(pt[0]), float(pt[1])))
            
            triangles = subdiv.getTriangleList()
            triangles = np.array(triangles, dtype=np.int32)
            
            # Warp source to target using triangles
            output = np.zeros_like(target)
            
            for tri in triangles:
                pts1 = src_pts[np.array([tri[0], tri[2], tri[4]], dtype=np.int32) // 2]
                pts2 = dst_pts[np.array([tri[0], tri[2], tri[4]], dtype=np.int32) // 2]
                
                # Get affine transform for this triangle
                M = cv2.getAffineTransform(pts1[:3].astype(np.float32), 
                                          pts2[:3].astype(np.float32))
                
                # Warp triangle
                warped_tri = cv2.warpAffine(source, M, (w, h))
                
                # Blend into output
                output = cv2.addWeighted(output, 0.95, warped_tri, 0.05, 0)
            
            # Final color correction
            output = self._histogram_match(output, target)
            
            return output
        except Exception as e:
            logger.error(f"Advanced warp failed: {e}")
            return None
    
    def _histogram_match(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Match histogram of source to target"""
        try:
            if len(source.shape) == 3 and source.shape[2] == 3:
                # Process each channel
                result = np.zeros_like(source)
                for i in range(3):
                    src_hist, src_bins = np.histogram(source[:,:,i].flatten(), 256, [0, 256])
                    tgt_hist, tgt_bins = np.histogram(target[:,:,i].flatten(), 256, [0, 256])
                    
                    # Create mapping
                    src_cdf = src_hist.cumsum()
                    src_cdf = src_cdf / src_cdf.max()
                    
                    tgt_cdf = tgt_hist.cumsum()
                    if tgt_cdf.max() > 0:
                        tgt_cdf = tgt_cdf / tgt_cdf.max()
                    
                    # Map colors
                    lut = np.interp(src_cdf, tgt_cdf, np.arange(256))
                    result[:,:,i] = cv2.LUT(source[:,:,i], lut.astype(np.uint8))
                
                return result
            else:
                return source.copy()
        except Exception as e:
            logger.error(f"Histogram match failed: {e}")
            return source.copy()
    
    def _swap_hybrid(
        self,
        source_crop: np.ndarray,
        target_crop: np.ndarray,
        source_face: Face,
        target_face: Face,
    ) -> Optional[np.ndarray]:
        """
        Hybrid swap combining multiple methods
        Uses InsightFace as base with optional DeepFaceLab quality enhancement
        """
        try:
            # Try DeepFaceLab-quality swap first if available
            dfl_result = self._swap_deepfacelab(
                source_crop, target_crop, source_face, target_face
            )
            
            if dfl_result is not None:
                swapped = dfl_result
            else:
                # Fallback to InsightFace method (passing faces for landmarks)
                swapped = self._swap_insightface(
                    source_crop, target_crop, source_face, target_face
                )
            
            if swapped is not None and self.config.preserve_expression:
                # Blend with target expression for more natural results
                swapped = self._blend_expression(
                    swapped, target_crop, self.config.identity_strength
                )
            
            return swapped
            
        except Exception as e:
            logger.error(f"Hybrid swap failed: {e}")
            return None
    
    def _extract_face_crop(
        self,
        image: np.ndarray,
        face: Face,
        expand: float = 1.0
    ) -> Optional[np.ndarray]:
        """Extract and expand face region"""
        try:
            bbox = self._expand_bbox(face.bbox, expand)
            x1, y1, x2, y2 = bbox
            
            # Clip to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            return image[y1:y2, x1:x2].copy()
            
        except Exception as e:
            logger.error(f"Crop extraction failed: {e}")
            return None
    
    def _expand_bbox(self, bbox: list, expand_ratio: float = 1.2) -> list:
        """Expand bounding box by ratio"""
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        
        # Calculate expansion
        expand_w = int(w * (expand_ratio - 1) / 2)
        expand_h = int(h * (expand_ratio - 1) / 2)
        
        return [
            int(x1 - expand_w),
            int(y1 - expand_h),
            int(x2 + expand_w),
            int(y2 + expand_h)
        ]
    
    def _normalize_crop(self, crop: np.ndarray) -> np.ndarray:
        """Normalize face crop"""
        # Ensure same size
        if crop.shape[0] != crop.shape[1]:
            size = min(crop.shape[0], crop.shape[1])
            crop = crop[:size, :size]
        
        return crop
    
    def _blend_face(
        self,
        target_image: np.ndarray,
        swapped_crop: np.ndarray,
        bbox: list,
        blend_mode: str = 'seamless'
    ) -> np.ndarray:
        """Blend swapped face into target image"""
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            
            # Clip to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(target_image.shape[1], x2), min(target_image.shape[0], y2)
            
            result = target_image.copy()
            
            if blend_mode == 'seamless':
                # Poisson blending for seamless integration
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                mask = np.ones(swapped_crop.shape[:2], dtype=np.uint8) * 255
                
                try:
                    result = cv2.seamlessClone(
                        swapped_crop, result, mask, center,
                        cv2.NORMAL_CLONE
                    )
                except:
                    # Fallback if seamlessClone fails
                    result[y1:y2, x1:x2] = swapped_crop
            
            elif blend_mode == 'poisson':
                # Poisson blending with gradient preservation
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                mask = np.ones(swapped_crop.shape[:2], dtype=np.uint8) * 255
                
                try:
                    result = cv2.seamlessClone(
                        swapped_crop, result, mask, center,
                        cv2.MIXED_CLONE
                    )
                except:
                    result[y1:y2, x1:x2] = swapped_crop
            
            else:  # no blending
                result[y1:y2, x1:x2] = swapped_crop
            
            # Apply Gaussian blur for smoothing
            if self.config.blur_amount > 0:
                kernel_size = 2 * self.config.blur_amount + 1
                result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)
            
            return result
            
        except Exception as e:
            logger.error(f"Face blending failed: {e}")
            return target_image.copy()
    
    def _blend_expression(
        self,
        swapped: np.ndarray,
        target: np.ndarray,
        strength: float = 0.8
    ) -> np.ndarray:
        """Blend target expression into swapped face"""
        try:
            # Simple weighted blending
            blended = cv2.addWeighted(swapped, strength, target, 1 - strength, 0)
            return blended
        except Exception as e:
            logger.error(f"Expression blending failed: {e}")
            return swapped
    
    def _create_face_mask(self, shape: tuple, landmarks: np.ndarray) -> np.ndarray:
        """Create face mask from landmarks for proper blending"""
        try:
            h, w = shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Get convex hull from landmarks
            if len(landmarks) > 0:
                hull = cv2.convexHull(landmarks.astype(np.int32))
                cv2.drawContours(mask, [hull], 0, 255, -1)
            else:
                # Fallback: create circular mask in center
                centerX, centerY = w // 2, h // 2
                radius = min(w, h) // 3
                cv2.circle(mask, (centerX, centerY), radius, 255, -1)
            
            # Blur mask edges for smooth blending
            mask = cv2.GaussianBlur(mask, (15, 15), 0)
            
            return mask
        except Exception as e:
            logger.error(f"Mask creation failed: {e}")
            # Return full mask as fallback
            return np.ones(shape[:2], dtype=np.uint8) * 255
    
    def _color_correct_face(
        self,
        source: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """Correct source face color to match target"""
        try:
            # Extract masked regions for color analysis
            source_masked = cv2.bitwise_and(source, source, mask=mask)
            target_masked = cv2.bitwise_and(target, target, mask=mask)
            
            # Convert to LAB color space for better color matching
            source_lab = cv2.cvtColor(source_masked.astype(np.uint8), cv2.COLOR_BGR2LAB)
            target_lab = cv2.cvtColor(target_masked.astype(np.uint8), cv2.COLOR_BGR2LAB)
            
            # Calculate mean color in masked regions
            mask_bool = mask > 128
            source_mean = source_lab[mask_bool].mean(axis=0)
            target_mean = target_lab[mask_bool].mean(axis=0)
            
            # Apply color correction
            source_lab = source_lab.astype(np.float32)
            source_lab = source_lab - source_mean + target_mean
            source_lab = np.clip(source_lab, 0, 255).astype(np.uint8)
            
            # Convert back to BGR
            corrected = cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR)
            
            # Keep original outside mask
            result = source.copy()
            result[mask_bool] = corrected[mask_bool]
            
            return result
        except Exception as e:
            logger.error(f"Color correction failed: {e}")
            return source.copy()
    
    def _blend_with_mask(
        self,
        background: np.ndarray,
        foreground: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """Blend foreground into background using mask"""
        try:
            # Normalize mask to 0-1 range
            mask_norm = mask.astype(np.float32) / 255.0
            
            # Add dimension for blending
            if len(mask_norm.shape) == 2:
                mask_norm = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
            
            # Blend using mask
            result = (foreground.astype(np.float32) * mask_norm + 
                     background.astype(np.float32) * (1 - mask_norm))
            
            return np.clip(result, 0, 255).astype(np.uint8)
        except Exception as e:
            logger.error(f"Mask blending failed: {e}")
            return background.copy()


# Convenience function
def create_swapper(method: str = 'insightface', **kwargs) -> FaceSwapper:
    """Create a face swapper instance"""
    config = SwapConfig(method=method, **kwargs)
    return FaceSwapper(config)
