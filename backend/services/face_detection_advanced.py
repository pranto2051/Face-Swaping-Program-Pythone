"""
Advanced Face Detection Service using InsightFace
Supports multi-face detection, alignment, and landmark extraction
Optimized for all device types (MPS, CUDA, CPU)
"""
import logging
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import torch
from core.model_loader import get_model_loader

logger = logging.getLogger(__name__)


@dataclass
class Face:
    """Face detection result"""
    id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    landmarks: List[List[float]]  # 5 or 68 facial landmarks
    angle: Dict[str, float]  # pitch, roll, yaw
    score: float  # confidence score
    embedding: Optional[np.ndarray] = None  # Face embedding for recognition
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'bbox': self.bbox,
            'landmarks': self.landmarks,
            'angle': self.angle,
            'score': self.score,
        }


class FaceDetector:
    """Advanced multi-face detector using InsightFace"""
    
    def __init__(self, model_name: str = 'buffalo_l'):
        """
        Initialize face detector
        
        Args:
            model_name: InsightFace model to use ('buffalo_l', 'buffalo_m', 'buffalo_s')
        """
        self.model_loader = get_model_loader()
        self.app = self.model_loader.get_insightface_model(model_name)
        self.model_name = model_name
        
    def detect_faces(self, image: np.ndarray, conf_threshold: float = 0.5) -> Tuple[List[Face], np.ndarray]:
        """
        Detect all faces in an image
        
        Args:
            image: Input image (BGR format, H x W x C)
            conf_threshold: Confidence threshold for detection (0.0-1.0)
            
        Returns:
            Tuple of (faces list, annotated image)
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided to detector")
            return [], image.copy()
        
        try:
            # Handle image format
            h, w = image.shape[:2]
            
            # Ensure BGR format
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                image_rgb = image.copy()
            
            # Detect faces
            faces = self.app.get(image_rgb)
            
            if not faces:
                logger.info("No faces detected in image")
                return [], image.copy()
            
            # Process detections
            detected_faces = []
            for idx, face in enumerate(faces):
                # Filter by confidence
                if face.det_score < conf_threshold:
                    continue
                
                # Extract bounding box
                bbox = face.bbox.astype(int).tolist()
                bbox = [max(0, bbox[0]), max(0, bbox[1]), 
                       min(w, bbox[2]), min(h, bbox[3])]
                
                # Extract landmarks
                landmarks = face.kps.tolist() if hasattr(face, 'kps') else []
                
                # Extract head pose angles
                angle = {
                    'pitch': getattr(face, 'pose', [0, 0, 0])[0],
                    'roll': getattr(face, 'pose', [0, 0, 0])[1],
                    'yaw': getattr(face, 'pose', [0, 0, 0])[2],
                }
                
                detected_face = Face(
                    id=idx,
                    bbox=bbox,
                    landmarks=landmarks,
                    angle=angle,
                    score=float(face.det_score),
                    embedding=face.embedding if hasattr(face, 'embedding') else None
                )
                detected_faces.append(detected_face)
            
            logger.info(f"✓ Detected {len(detected_faces)} faces in image")
            
            # Create annotated image
            annotated = self._annotate_faces(image.copy(), detected_faces)
            
            return detected_faces, annotated
            
        except Exception as e:
            logger.error(f"✗ Face detection failed: {e}")
            return [], image.copy()
    
    def detect_and_align_faces(self, image: np.ndarray, conf_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Detect faces and prepare aligned crops
        
        Args:
            image: Input image
            conf_threshold: Confidence threshold
            
        Returns:
            Dictionary with detection results and aligned crops
        """
        faces, annotated = self.detect_faces(image, conf_threshold)
        
        result = {
            'faces': [face.to_dict() for face in faces],
            'count': len(faces),
            'image_shape': image.shape[:2],
            'annotated_image': annotated,
            'aligned_faces': []
        }
        
        # Generate aligned face crops
        for face in faces:
            try:
                aligned = self._align_face(image, face)
                if aligned is not None:
                    result['aligned_faces'].append({
                        'face_id': face.id,
                        'aligned_image': aligned,
                        'face': face
                    })
            except Exception as e:
                logger.warning(f"Failed to align face {face.id}: {e}")
        
        return result
    
    def _align_face(self, image: np.ndarray, face: Face, output_size: int = 512) -> Optional[np.ndarray]:
        """
        Align face using landmarks
        
        Args:
            image: Source image
            face: Face object with landmarks
            output_size: Size of aligned output
            
        Returns:
            Aligned face crop
        """
        try:
            if not face.landmarks or len(face.landmarks) < 2:
                # Use bounding box for simple alignment
                x1, y1, x2, y2 = face.bbox
                return cv2.resize(image[y1:y2, x1:x2], (output_size, output_size))
            
            # Use landmarks for perspective transform
            landmarks = np.array(face.landmarks[:2])  # Use first 2 points
            
            # Reference landmarks for aligned face
            ref_landmarks = np.array([
                [0.3, 0.3],
                [0.7, 0.3]
            ]) * output_size
            
            # Calculate transformation matrix
            M = cv2.getAffineTransform(
                landmarks.astype(np.float32),
                ref_landmarks.astype(np.float32)
            )
            
            # Apply transformation
            aligned = cv2.warpAffine(
                image,
                M,
                (output_size, output_size),
                borderMode=cv2.BORDER_REFLECT_101
            )
            
            return aligned
            
        except Exception as e:
            logger.debug(f"Alignment failed: {e}")
            return None
    
    def _annotate_faces(self, image: np.ndarray, faces: List[Face]) -> np.ndarray:
        """
        Draw face bounding boxes and landmarks on image
        
        Args:
            image: Input image
            faces: List of detected faces
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        for face in faces:
            # Draw bounding box
            x1, y1, x2, y2 = face.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw face ID and confidence
            label = f"Face {face.id} ({face.score:.2f})"
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw landmarks
            if face.landmarks:
                landmarks = np.array(face.landmarks, dtype=np.int32)
                for point in landmarks:
                    cv2.circle(annotated, tuple(point), 2, (255, 0, 0), -1)
        
        return annotated
    
    def select_face(self, faces: List[Face], face_id: Optional[int] = None) -> Optional[Face]:
        """
        Select a single face from detection results
        
        Args:
            faces: List of detected faces
            face_id: ID of face to select (None = highest confidence)
            
        Returns:
            Selected face or None
        """
        if not faces:
            return None
        
        if face_id is not None:
            for face in faces:
                if face.id == face_id:
                    return face
            logger.warning(f"Face ID {face_id} not found")
            return None
        
        # Select highest confidence face
        return max(faces, key=lambda f: f.score)
    
    def get_face_embedding(self, image: np.ndarray, face: Face) -> Optional[np.ndarray]:
        """
        Get face embedding for recognition
        
        Args:
            image: Source image
            face: Face object
            
        Returns:
            Face embedding vector
        """
        try:
            if face.embedding is not None:
                return face.embedding
            
            # Extract face region
            x1, y1, x2, y2 = face.bbox
            face_crop = image[y1:y2, x1:x2]
            
            # Get embedding through InsightFace
            faces = self.app.get(image)
            if faces:
                return faces[0].embedding
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get face embedding: {e}")
            return None
    
    def compare_faces(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compare two face embeddings
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Similarity score (0-1, higher = more similar)
        """
        try:
            # Normalize embeddings
            emb1 = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
            emb2 = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
            
            # Cosine similarity
            similarity = np.dot(emb1, emb2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Face comparison failed: {e}")
            return 0.0


# Convenience function
def create_detector(model_name: str = 'buffalo_l') -> FaceDetector:
    """Create a face detector instance"""
    return FaceDetector(model_name)
