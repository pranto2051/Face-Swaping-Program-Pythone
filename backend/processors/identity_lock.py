"""
Identity Lock & Preservation Module

Ensures face identity is preserved during enhancement by:
1. Storing original embedding from InsightFace swap
2. Verifying enhancement doesn't drift identity
3. Falling back if similarity drops below threshold
"""

import numpy as np
import insightface
from typing import Tuple, Optional, Dict
from scipy.spatial.distance import cosine


class IdentityLocker:
    """Manages identity preservation throughout enhancement pipeline."""
    
    def __init__(self, similarity_threshold: float = 0.90):
        """
        Args:
            similarity_threshold: Min cosine similarity (0-1) to preserve identity
                                Higher = stricter identity lock
                                0.90+ = very strict (recommended)
        """
        self.similarity_threshold = similarity_threshold
        self.locked_embedding = None
        self.locked_bbox = None
        self.locked_landmarks = None
        self.locked_face_info = {}
        
    def lock_identity(self, image: np.ndarray, face_object, 
                     face_analyzer) -> Dict:
        """
        Extract and lock identity info immediately after swap.
        
        Args:
            image: Swapped image (BGR)
            face_object: Face object from InsightFace detector
            face_analyzer: InsightFace analyzer
            
        Returns:
            Dict with locked identity info
        """
        # Store embedding (this is the target identity we want to preserve)
        self.locked_embedding = face_object.embedding.copy()
        self.locked_bbox = face_object.bbox.copy()
        self.locked_landmarks = face_object.kps.copy() if hasattr(face_object, 'kps') else None
        
        # Store detailed face info
        self.locked_face_info = {
            'embedding': self.locked_embedding,
            'bbox': self.locked_bbox,
            'landmarks': self.locked_landmarks,
            'gender': face_object.gender if hasattr(face_object, 'gender') else None,
            'age': face_object.age if hasattr(face_object, 'age') else None,
            'expression': face_object.expression if hasattr(face_object, 'expression') else None,
        }
        
        print(f"✓ Identity locked: embedding={self.locked_embedding.shape}, "
              f"bbox={self.locked_bbox}, similarity_threshold={self.similarity_threshold}")
        
        return self.locked_face_info
    
    def verify_identity(self, enhanced_image: np.ndarray, 
                       face_object, verbose: bool = True) -> Tuple[bool, float]:
        """
        Check if enhanced face still matches original identity.
        
        Args:
            enhanced_image: Enhanced face image (cropped, same size as original)
            face_object: Face object from detector on enhanced image
            verbose: Print verification results
            
        Returns:
            (is_preserved, similarity_score)
        """
        if self.locked_embedding is None:
            raise ValueError("Identity not locked yet. Call lock_identity() first.")
        
        # Compare embeddings
        enhanced_embedding = face_object.embedding
        similarity = self._cosine_similarity(
            self.locked_embedding, 
            enhanced_embedding
        )
        
        is_preserved = similarity >= self.similarity_threshold
        
        if verbose:
            status = "✓ PASS" if is_preserved else "✗ FAIL"
            print(f"{status} Identity verification: similarity={similarity:.4f} "
                  f"(threshold={self.similarity_threshold})")
            
            if not is_preserved:
                drift = self.similarity_threshold - similarity
                print(f"    ⚠️  Identity drift detected: {drift:.4f} below threshold")
                print(f"    Action: Fallback to basic enhancement")
        
        return is_preserved, similarity
    
    def verify_identity_on_crop(self, face_crop: np.ndarray,
                               face_analyzer) -> Tuple[bool, float]:
        """
        Verify identity directly from enhanced face crop (no need to detect first).
        
        Args:
            face_crop: Enhanced face crop (already extracted)
            face_analyzer: InsightFace analyzer
            
        Returns:
            (is_preserved, similarity_score)
        """
        if self.locked_embedding is None:
            raise ValueError("Identity not locked yet. Call lock_identity() first.")
        
        # Detect face in crop
        faces = face_analyzer.get(face_crop)
        if not faces:
            print("✗ No face detected in enhanced crop!")
            return False, 0.0
        
        face = faces[0]
        return self.verify_identity(face_crop, face, verbose=True)
    
    def get_locked_info(self) -> Dict:
        """Get reference to locked identity info."""
        return self.locked_face_info.copy()
    
    def reset(self):
        """Reset lock for next image."""
        self.locked_embedding = None
        self.locked_bbox = None
        self.locked_landmarks = None
        self.locked_face_info = {}
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings (0-1)."""
        # Cosine distance returns 0-2, similarity is 1 - distance
        distance = cosine(vec1, vec2)
        similarity = 1 - distance
        return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
    
    @staticmethod
    def batch_verify_identities(locked_embedding: np.ndarray,
                               enhanced_embeddings: np.ndarray) -> np.ndarray:
        """
        Verify multiple enhanced embeddings against one locked embedding.
        
        Args:
            locked_embedding: Reference embedding (shape: N,)
            enhanced_embeddings: Multiple embeddings (shape: M, N)
            
        Returns:
            Array of similarities (shape: M,)
        """
        similarities = []
        for enhanced_emb in enhanced_embeddings:
            sim = 1 - cosine(locked_embedding, enhanced_emb)
            similarities.append(max(0.0, min(1.0, sim)))
        return np.array(similarities)


class IdentityPreservationMonitor:
    """Tracks identity preservation across entire pipeline for debugging."""
    
    def __init__(self):
        self.history = []
    
    def record(self, stage: str, similarity: float, status: str = ""):
        """Record identity check at each stage."""
        self.history.append({
            'stage': stage,
            'similarity': similarity,
            'status': status,
        })
    
    def print_report(self):
        """Print identity preservation report."""
        print("\n" + "="*60)
        print("IDENTITY PRESERVATION REPORT")
        print("="*60)
        for record in self.history:
            sim_str = f"{record['similarity']:.4f}" if isinstance(record['similarity'], (int, float)) else "N/A"
            print(f"{record['stage']:30} | Sim: {sim_str:8} | {record['status']}")
        print("="*60 + "\n")
    
    def reset(self):
        self.history = []

