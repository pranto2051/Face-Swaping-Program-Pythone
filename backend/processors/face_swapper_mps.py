"""
MPS-Optimized Face Swapping Pipeline
Designed to preserve maximum resolution and detail for the enhancement stage
"""

import cv2
import numpy as np
import torch
from typing import Tuple, Dict, Optional, List
import insightface
from core.device_manager import get_device_manager


class MPS_FaceSwapper:
    """
    High-quality face swapping optimized for Apple Silicon.
    Preserves original image resolution and minimizes artifacts.
    """
    
    def __init__(self):
        """Initialize face swapper with optimal settings for MPS."""
        self.device_manager = get_device_manager()
        self.device = self.device_manager.get_device()
        
        # Use insightface with CPU provider for MPS compatibility
        # (GPU providers may not work reliably on Apple Silicon)
        try:
            # Initialize swapper with CPU context for stability
            providers = [('CPUExecutionProvider',)]
            
            self.swapper = insightface.model_zoo.get_model(
                'inswapper_128.onnx',
                providers=providers
            )
            
            self.face_analyser = insightface.app.FaceAnalysis(
                name='buffalo_l',
                providers=providers,
                root=None
            )
            self.face_analyser.prepare(ctx_id=0, det_size=(640, 640))
            
            print("✓ MPS_FaceSwapper initialized with CPU providers (ONNX Runtime)")
            
        except Exception as e:
            print(f"⚠️  Error initializing face swapper: {e}")
            raise
    
    def detect_faces(self, image: np.ndarray) -> List[insightface.utils.Face]:
        """
        Detect all faces in image using buffalo model.
        
        Args:
            image: Input image (BGR)
        
        Returns:
            List of Face objects with embeddings
        """
        try:
            faces = self.face_analyser.get(image)
            return faces
        except Exception as e:
            print(f"Face detection error: {e}")
            return []
    
    def swap_face(self, source_image: np.ndarray, target_image: np.ndarray,
                  source_faces: List[insightface.utils.Face],
                  target_faces: List[insightface.utils.Face],
                  source_face_index: int = 0,
                  target_face_index: int = 0) -> Tuple[np.ndarray, bool]:
        """
        Swap face from source image to target image.
        Preserves original resolution with enhanced size normalization.
        
        Args:
            source_image: Source image (BGR, face source)
            target_image: Target image (BGR, where face will be swapped)
            source_faces: Detected faces in source image
            target_faces: Detected faces in target image
            source_face_index: Which source face to use (0 = first)
            target_face_index: Which target face to replace (0 = first)
        
        Returns:
            Tuple of (swapped_image, success)
        """
        try:
            if not source_faces or not target_faces:
                return target_image, False
            
            source_face = source_faces[min(source_face_index, len(source_faces)-1)]
            target_face = target_faces[min(target_face_index, len(target_faces)-1)]
            
            # Use enhanced alignment if landmarks are available
            if hasattr(source_face, 'kps') and hasattr(target_face, 'kps'):
                if source_face.kps is not None and target_face.kps is not None:
                    swapped = self._swap_with_landmark_alignment(target_image, target_face, source_face)
                    return swapped, True
            
            # Fallback to standard swap
            swapped = self.swapper.get(target_image, target_face, source_face, 
                                       paste_back=True)
            
            return swapped, True
            
        except Exception as e:
            print(f"Face swap failed: {e}")
            return target_image, False
    
    def _swap_with_landmark_alignment(self, target_img: np.ndarray, 
                                     target_face, source_face) -> np.ndarray:
        """
        Enhanced face swap with landmark-based size normalization.
        Prevents face size mismatch and ensures boundaries are respected.
        """
        try:
            # Get landmarks
            src_landmarks = source_face.kps.astype(np.float32)
            tgt_landmarks = target_face.kps.astype(np.float32)
            
            # Calculate eye distance for proper size matching
            src_eye_dist = self._compute_eye_distance(src_landmarks)
            tgt_eye_dist = self._compute_eye_distance(tgt_landmarks)
            
            if src_eye_dist < 1 or tgt_eye_dist < 1:
                return self.swapper.get(target_img, target_face, source_face, paste_back=True)
            
            # If sizes already match closely (within 5%), use standard swap
            size_ratio = tgt_eye_dist / src_eye_dist
            if 0.95 <= size_ratio <= 1.05:
                return self.swapper.get(target_img, target_face, source_face, paste_back=True)
            
            # Perform initial swap
            result = self.swapper.get(target_img, target_face, source_face, paste_back=True)
            
            # Re-detect face in the swapped result to get the actual swapped face landmarks
            from insightface.app import FaceAnalysis
            import onnxruntime
            
            # Quick face detection on result
            app = FaceAnalysis(providers=['CPUExecutionProvider'])
            app.prepare(ctx_id=0)
            result_faces = app.get(result)
            
            if not result_faces:
                return result  # No face detected in result, return as-is
            
            # Find the face closest to target position
            result_face = min(result_faces, 
                            key=lambda f: np.linalg.norm(f.kps.mean(axis=0) - tgt_landmarks.mean(axis=0)))
            
            result_landmarks = result_face.kps.astype(np.float32)
            result_eye_dist = self._compute_eye_distance(result_landmarks)
            
            # Calculate how much we need to scale the result
            scale_factor = tgt_eye_dist / result_eye_dist if result_eye_dist > 0 else 1.0
            
            # If the result already matches (within 3%), return as-is
            if 0.97 <= scale_factor <= 1.03:
                return result
            
            # Extract face region from result
            result_bbox = result_face.bbox.astype(int)
            x1, y1, x2, y2 = result_bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(result.shape[1], x2), min(result.shape[0], y2)
            
            face_region = result[y1:y2, x1:x2]
            
            # Resize the face region
            new_h = int(face_region.shape[0] * scale_factor)
            new_w = int(face_region.shape[1] * scale_factor)
            resized_face = cv2.resize(face_region, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Calculate center positions
            result_center = result_landmarks.mean(axis=0).astype(int)
            
            # Calculate paste position (centered on target)
            paste_x = max(0, result_center[0] - new_w // 2)
            paste_y = max(0, result_center[1] - new_h // 2)
            
            # Ensure we don't paste outside bounds
            paste_x = min(paste_x, result.shape[1] - new_w)
            paste_y = min(paste_y, result.shape[0] - new_h)
            
            # Create a copy of the result
            final_result = result.copy()
            
            # Create blend mask for smooth transition
            mask = np.ones((new_h, new_w), dtype=np.float32)
            feather = min(20, new_w // 10, new_h // 10)
            if feather > 0:
                mask = cv2.GaussianBlur(mask, (feather*2+1, feather*2+1), 0)
            mask = mask[..., np.newaxis]
            
            # Blend the resized face onto the result
            y_end = min(paste_y + new_h, final_result.shape[0])
            x_end = min(paste_x + new_w, final_result.shape[1])
            actual_h = y_end - paste_y
            actual_w = x_end - paste_x
            
            if actual_h > 0 and actual_w > 0:
                region = final_result[paste_y:y_end, paste_x:x_end]
                face_crop = resized_face[:actual_h, :actual_w]
                mask_crop = mask[:actual_h, :actual_w]
                
                blended = region.astype(np.float32) * (1 - mask_crop) + face_crop.astype(np.float32) * mask_crop
                final_result[paste_y:y_end, paste_x:x_end] = blended.astype(np.uint8)
            
            return final_result
            
        except Exception as e:
            print(f"Enhanced alignment failed: {e}")
            import traceback
            traceback.print_exc()
            return self.swapper.get(target_img, target_face, source_face, paste_back=True)
    
    def _compute_eye_distance(self, landmarks: np.ndarray) -> float:
        """
        Compute distance between eyes for scale normalization.
        Landmarks: [left_eye, right_eye, nose, left_mouth, right_mouth]
        """
        if len(landmarks) < 2:
            return 0.0
        
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        distance = np.linalg.norm(right_eye - left_eye)
        return float(distance)
    
    def _create_boundary_mask(self, img_shape: Tuple[int, int], 
                             landmarks: np.ndarray) -> np.ndarray:
        """
        Create convex hull mask from landmarks to ensure face stays within boundaries.
        """
        mask = np.zeros(img_shape, dtype=np.uint8)
        
        # Create convex hull
        hull = cv2.convexHull(landmarks.astype(np.int32))
        
        # Expand hull slightly (15%) for natural blending
        center = hull.mean(axis=0)
        expanded_hull = center + (hull - center) * 1.15
        expanded_hull = expanded_hull.astype(np.int32)
        
        # Fill mask
        cv2.fillConvexPoly(mask, expanded_hull, 255)
        
        return mask
    
    def extract_face_region(self, image: np.ndarray, 
                           face: insightface.utils.Face) -> Tuple[np.ndarray, np.ndarray, Tuple]:
        """
        Extract face region with bounding box and mask.
        
        Args:
            image: Original image (BGR)
            face: Detected face object
        
        Returns:
            Tuple of (face_region, mask, bbox)
        """
        try:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            # Extract face region
            face_region = image[y1:y2, x1:x2].copy()
            
            # Create face mask from landmarks
            mask = self._create_face_mask(face_region, face)
            
            # Return face, mask, and bbox in (x, y, w, h) format
            return face_region, mask, (x1, y1, x2-x1, y2-y1)
            
        except Exception as e:
            print(f"Error extracting face region: {e}")
            return None, None, None
    
    def _create_face_mask(self, face_image: np.ndarray, 
                         face: insightface.utils.Face) -> np.ndarray:
        """
        Create face mask from detected landmarks.
        Mask is white (255) for face, black (0) for background.
        """
        try:
            h, w = face_image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Get landmarks and translate to face region coordinates
            if hasattr(face, 'landmark_3d') and face.landmark_3d is not None:
                landmarks = face.landmark_3d
            else:
                landmarks = face.kps if hasattr(face, 'kps') else None
            
            if landmarks is not None:
                # Translate landmarks to face region
                landmarks_local = landmarks.copy()
                landmarks_local[:, 0] -= face.bbox[0]
                landmarks_local[:, 1] -= face.bbox[1]
                
                # Create convex hull from landmarks
                hull = cv2.convexHull(landmarks_local.astype(np.int32))
                
                # Fill the mask
                cv2.drawContours(mask, [hull], 0, 255, -1)
                
                # Expand mask slightly for better blending
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.dilate(mask, kernel, iterations=1)
            else:
                # Fallback: use entire face region
                mask[:, :] = 255
            
            return mask
            
        except Exception as e:
            print(f"Mask creation error: {e}")
            # Return full white mask as fallback
            return np.ones((face_image.shape[0], face_image.shape[1]), dtype=np.uint8) * 255
    
    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'swapper'):
            del self.swapper
        if hasattr(self, 'face_analyser'):
            del self.face_analyser
