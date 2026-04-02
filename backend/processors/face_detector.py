import numpy as np
from typing import List, NamedTuple
import insightface
from insightface.app import FaceAnalysis

class DetectedFace(NamedTuple):
    id: int
    bbox: List[float]
    confidence: float
    landmarks: np.ndarray
    embedding: np.ndarray

class MultiFaceDetector:
    """
    Pipeline:
    1. Run InsightFace FaceAnalysis with det_size=(640,640)
    2. For each detected face, extract:
         - bbox (x1, y1, x2, y2) in image pixel coordinates
       - det_score (confidence 0-1)
       - landmarks_2d_106 → reduce to 68-point for alignment
       - embedding (512-d vector for cache)
    3. Sort by bbox area descending (largest = primary)
    4. Return all faces with IDs for frontend selection
    """
    def __init__(self, name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider']):
        self.app = FaceAnalysis(name=name, providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        faces = self.app.get(image)
        # Sort by bbox area descending
        sorted_faces = sorted(faces, key=lambda f: self._bbox_area(f.bbox), reverse=True)
        
        return [
            DetectedFace(
                id=i,
                bbox=self._pixel_bbox(f.bbox, image.shape),
                confidence=float(f.det_score),
                landmarks=f.landmark_2d_106, # 106 points
                embedding=f.embedding
            )
            for i, f in enumerate(sorted_faces)
        ]

    def _bbox_area(self, bbox):
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    def _pixel_bbox(self, bbox, shape):
        h, w = shape[:2]
        x1 = int(max(0, min(w - 1, bbox[0])))
        y1 = int(max(0, min(h - 1, bbox[1])))
        x2 = int(max(0, min(w - 1, bbox[2])))
        y2 = int(max(0, min(h - 1, bbox[3])))
        return [
            x1,
            y1,
            x2,
            y2,
        ]
