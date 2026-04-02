import os
import cv2
import insightface
from insightface.model_zoo import get_model
import numpy as np

class FaceSwapper:
    """
    Inference Stage:
    1. Load 'inswapper_128' model using InsightFace model_zoo.
    2. Input: source_face (from source image), target_face (from target image).
    3. Output: Swapped face region blended into the target image.
    
    Enhanced with landmark-based size normalization to prevent face size mismatch.
    """
    def __init__(self, model_path=None, providers=None):
        import os
        if model_path is None:
            model_path = os.path.expanduser('~/.insightface/models/inswapper_128.onnx')

        if providers is None:
            # CoreML can fail to compile/save models on some macOS setups.
            providers = ['CPUExecutionProvider']
        
        # Specify providers to avoid issues with GPU on non-supported hardware
        self.model = get_model(model_path, providers=providers)
        self.use_enhanced_alignment = True  # Enable landmark-based size normalization

    def swap(self, target_img, target_face, source_face):
        """
        Performs the face swap with enhanced size normalization.
        target_img: np.ndarray (the full target image)
        target_face: The face object detected in the target image
        source_face: The face object detected in the source image
        """
        if self.use_enhanced_alignment and hasattr(source_face, 'kps') and hasattr(target_face, 'kps'):
            if source_face.kps is not None and target_face.kps is not None:
                return self._swap_with_landmark_alignment(target_img, target_face, source_face)
        
        # Fallback to standard InsightFace swap
        return self.model.get(target_img, target_face, source_face, paste_back=True)
    
    def _swap_with_landmark_alignment(self, target_img, target_face, source_face):
        """
        Enhanced face swap with landmark-based size normalization.
        Ensures the swapped face perfectly matches target face size and stays within boundaries.
        """
        try:
            # Get landmarks
            src_landmarks = source_face.kps.astype(np.float32)
            tgt_landmarks = target_face.kps.astype(np.float32)
            
            # Calculate eye distance for proper size matching
            src_eye_dist = self._compute_eye_distance(src_landmarks)
            tgt_eye_dist = self._compute_eye_distance(tgt_landmarks)
            
            if src_eye_dist < 1 or tgt_eye_dist < 1:
                return self.model.get(target_img, target_face, source_face, paste_back=True)
            
            # If sizes already match closely (within 5%), use standard swap
            size_ratio = tgt_eye_dist / src_eye_dist
            if 0.95 <= size_ratio <= 1.05:
                return self.model.get(target_img, target_face, source_face, paste_back=True)
            
            # Perform initial swap
            result = self.model.get(target_img, target_face, source_face, paste_back=True)
            
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
            print(f"Enhanced alignment failed: {e}, falling back to standard swap")
            import traceback
            traceback.print_exc()
            return self.model.get(target_img, target_face, source_face, paste_back=True)
    
    def _compute_eye_distance(self, landmarks):
        """
        Compute Euclidean distance between eyes using landmarks.
        Landmarks order: [left_eye, right_eye, nose, left_mouth, right_mouth]
        """
        if len(landmarks) < 2:
            return 0
        
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        
        distance = np.linalg.norm(right_eye - left_eye)
        return distance
    
    def _create_face_mask(self, img_shape, landmarks):
        """
        Create a binary mask using convex hull of facial landmarks.
        Ensures the swapped face stays within the target face boundary.
        """
        mask = np.zeros(img_shape, dtype=np.uint8)
        
        # Create convex hull from landmarks
        hull = cv2.convexHull(landmarks.astype(np.int32))
        
        # Expand hull slightly for natural blending
        center = hull.mean(axis=0)
        expanded_hull = center + (hull - center) * 1.15
        expanded_hull = expanded_hull.astype(np.int32)
        
        # Fill the mask
        cv2.fillConvexPoly(mask, expanded_hull, 255)
        
        return mask
