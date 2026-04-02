"""
Safe Face Enhancement Module

Identity-preserving enhancement that:
1. Extracts face crop at original resolution (NO downscale)
2. Applies enhancement only to face region
3. Verifies identity preserved after enhancement
4. Blends back using proper masking
5. Returns to baseline if identity is lost
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from processors.identity_lock import IdentityLocker


class SafeFaceEnhancer:
    """
    Enhances swapped face while strictly preserving identity.
    Works with CodeFormer, GFPGAN, or custom enhancement functions.
    """
    
    def __init__(self, enhancement_model=None, face_analyzer=None, 
                 similarity_threshold: float = 0.90):
        """
        Args:
            enhancement_model: CodeFormer, GFPGAN, or custom model
            face_analyzer: InsightFace analyzer for identity verification
            similarity_threshold: Min cosine similarity to preserve identity
        """
        self.enhancement_model = enhancement_model
        self.face_analyzer = face_analyzer
        self.identity_locker = IdentityLocker(similarity_threshold)
        self.monitor = None
    
    def enhance_after_swap(self, 
                          swapped_image: np.ndarray,
                          swapped_face_object,
                          original_image: np.ndarray,
                          quality_mode: str = 'studio',
                          use_fallback_on_failure: bool = True) -> np.ndarray:
        """
        Safely enhance swapped face while preserving identity.
        
        Args:
            swapped_image: Image with face already swapped (full resolution)
            swapped_face_object: Face object from InsightFace detector
            original_image: Original target image (for color reference)
            quality_mode: 'studio' or 'cinematic'
            use_fallback_on_failure: Use basic enhancement if identity check fails
            
        Returns:
            Enhanced image with preserved identity
        """
        print(f"\n[SafeFaceEnhancer] Starting enhancement (mode={quality_mode})")
        
        # Step 1: Lock identity immediately after swap
        self.identity_locker.lock_identity(
            swapped_image, 
            swapped_face_object, 
            self.face_analyzer
        )
        
        # Step 2: Extract face crop at ORIGINAL resolution
        face_crop, face_bbox, face_mask = self._extract_face_crop(
            swapped_image, 
            swapped_face_object
        )
        print(f"  → Face crop extracted: {face_crop.shape}")
        
        # Step 3: Apply enhancement to CROP ONLY
        enhanced_crop = self._enhance_face_crop(
            face_crop, 
            quality_mode
        )
        print(f"  → Face enhancement applied: {enhanced_crop.shape}")
        
        # Step 4: Verify identity preserved
        is_preserved, similarity = self.identity_locker.verify_identity_on_crop(
            enhanced_crop,
            self.face_analyzer
        )
        
        if not is_preserved:
            print(f"  ✗ Identity not preserved (sim={similarity:.4f})")
            if use_fallback_on_failure:
                print(f"  → Using fallback basic enhancement")
                enhanced_crop = self._apply_basic_enhancement(face_crop)
            else:
                print(f"  → Skipping enhancement")
                enhanced_crop = face_crop
        else:
            print(f"  ✓ Identity preserved (sim={similarity:.4f})")
        
        # Step 5: Blend enhanced crop back into full image
        result = self._blend_face_back(
            swapped_image,
            enhanced_crop,
            face_bbox,
            face_mask
        )
        print(f"  → Face blended back: {result.shape}")
        
        # Step 6: Color correction
        result = self._apply_color_correction(
            result,
            original_image,
            face_bbox
        )
        print(f"  → Color correction applied")
        
        # Step 7: Soft global sharpening (if cinematic)
        if quality_mode == 'cinematic':
            result = self._apply_soft_sharpening(result)
            print(f"  → Soft sharpening applied")
        
        return result
    
    def _extract_face_crop(self, image: np.ndarray, face_object) -> Tuple[np.ndarray, Tuple, np.ndarray]:
        """
        Extract face crop at original resolution.
        
        Returns:
            (face_crop, face_bbox, face_mask)
        """
        # Get bounding box
        bbox = face_object.bbox.astype(int)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        
        # Ensure within bounds
        h, w = image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Extract crop (NO downscaling)
        face_crop = image[y1:y2, x1:x2].copy()
        
        # Generate face mask from landmarks
        face_mask = self._generate_face_mask(
            face_object,
            face_crop.shape[:2]
        )
        
        return face_crop, (x1, y1, x2, y2), face_mask
    
    def _generate_face_mask(self, face_object, crop_shape: Tuple) -> np.ndarray:
        """
        Generate face mask from landmarks.
        Mask is in crop coordinates (0,0 = top-left of crop).
        """
        h, w = crop_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Use landmarks if available
        if hasattr(face_object, 'kps') and face_object.kps is not None:
            # Landmarks are in original image coordinates
            # Convert to crop coordinates
            bbox = face_object.bbox.astype(int)
            x1, y1 = bbox[0], bbox[1]
            
            landmarks = (face_object.kps - np.array([x1, y1])).astype(np.int32)
            
            # Create convex hull
            hull = cv2.convexHull(landmarks)
            
            # Expand hull for better blending (10% expansion)
            center = hull.mean(axis=0)
            expanded_hull = center + (hull - center) * 1.1
            expanded_hull = np.clip(expanded_hull, 0, [w-1, h-1]).astype(np.int32)
            
            # Fill the hull
            cv2.fillConvexPoly(mask, expanded_hull, 255)
        else:
            # Fallback: elliptical mask
            center = (w // 2, h // 2)
            axes = (w // 2, h // 2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        return mask
    
    def _enhance_face_crop(self, face_crop: np.ndarray, quality_mode: str) -> np.ndarray:
        """
        Enhance face crop using model.
        Important: Model receives ONLY the face crop, not full image.
        This prevents identity regeneration on background.
        """
        # Fast mode: skip enhancement
        if quality_mode == 'fast':
            return face_crop
        
        # Try using provided model (CodeFormer, GFPGAN, etc.)
        if self.enhancement_model is not None:
            try:
                enhanced = self._model_enhance_crop(face_crop, quality_mode)
                return enhanced
            except Exception as e:
                print(f"    ⚠️  Model enhancement failed: {e}")
                print(f"    → Fallback to basic enhancement")
        
        # Fallback: basic OpenCV enhancement
        return self._apply_basic_enhancement(face_crop)
    
    def _model_enhance_crop(self, face_crop: np.ndarray, quality_mode: str) -> np.ndarray:
        """
        Use enhancement model on face crop.
        Model name should be in self.enhancement_model.
        """
        # Get enhancement params based on mode
        if quality_mode == 'studio':
            fidelity = 0.7  # Balanced
        else:  # cinematic
            fidelity = 0.8  # Higher fidelity (less change)
        
        # Try CodeFormer first (preferred for identity preservation)
        try:
            import torch
            from models.codeformer import CodeFormer
            
            if isinstance(self.enhancement_model, str) and 'codeformer' in self.enhancement_model.lower():
                # CodeFormer enhancement
                model = CodeFormer(model_path=self.enhancement_model, dim_embd=512)
                model.eval()
                
                # Prepare input
                face_tensor = torch.from_numpy(
                    cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                ).float() / 255.0
                face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0)
                
                with torch.no_grad():
                    # fidelity_weight: 0=high quality, 1=preserve details
                    enhanced_tensor = model(face_tensor, fidelity_weight=1.0 - fidelity)
                
                # Convert back to BGR
                enhanced = enhanced_tensor.squeeze(0).permute(1, 2, 0).numpy()
                enhanced = (enhanced * 255).astype(np.uint8)
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
                
                return enhanced
        except Exception as e:
            print(f"    CodeFormer error: {e}")
        
        # Try GFPGAN (less ideal, applies restoration)
        try:
            if self.enhancement_model is not None and hasattr(self.enhancement_model, 'enhance'):
                # GFPGAN with face_crop only
                # Set has_aligned=True if possible to skip auto-alignment
                _, _, enhanced = self.enhancement_model.enhance(
                    face_crop,
                    has_aligned=True,  # Already aligned face
                    only_center_face=True,  # Only process center
                    paste_back=False  # Don't paste, return only
                )
                return enhanced
        except Exception as e:
            print(f"    GFPGAN error: {e}")
        
        raise RuntimeError("No valid enhancement model available")
    
    def _blend_face_back(self, full_image: np.ndarray, enhanced_crop: np.ndarray,
                        face_bbox: Tuple, face_mask: np.ndarray) -> np.ndarray:
        """
        Blend enhanced face crop back into full image using Poisson blending.
        """
        x1, y1, x2, y2 = face_bbox
        
        # Ensure crop matches bbox size
        crop_h, crop_w = enhanced_crop.shape[:2]
        bbox_h, bbox_w = y2 - y1, x2 - x1
        
        if (crop_h, crop_w) != (bbox_h, bbox_w):
            enhanced_crop = cv2.resize(enhanced_crop, (bbox_w, bbox_h))
        
        # Create feathered mask for smooth blending
        feathered_mask = self._feather_mask(face_mask, kernel_size=15)
        
        # Ensure mask is 3-channel
        if len(feathered_mask.shape) == 2:
            feathered_mask = cv2.cvtColor(feathered_mask, cv2.COLOR_GRAY2BGR)
        feathered_mask = feathered_mask.astype(np.float32) / 255.0
        
        # Calculate center for Poisson blending
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        try:
            # Use Poisson blending for seamless integration
            result = cv2.seamlessClone(
                enhanced_crop,
                full_image,
                (feathered_mask[:, :, 0] * 255).astype(np.uint8),
                center,
                cv2.NORMAL_CLONE
            )
            return result
        except Exception as e:
            print(f"    Poisson blending failed: {e}")
            print(f"    → Fallback to alpha blending")
        
        # Fallback: alpha blending
        result = full_image.copy()
        result[y1:y2, x1:x2] = (
            enhanced_crop * feathered_mask +
            full_image[y1:y2, x1:x2] * (1 - feathered_mask)
        ).astype(np.uint8)
        
        return result
    
    def _feather_mask(self, mask: np.ndarray, kernel_size: int = 15) -> np.ndarray:
        """Apply Gaussian blur to mask edges for smooth blending."""
        # Ensure mask is single channel
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to feather edges
        feathered = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        
        return feathered
    
    def _apply_color_correction(self, result: np.ndarray, 
                               reference: np.ndarray,
                               face_bbox: Tuple) -> np.ndarray:
        """
        Match color and lighting of swapped face to target image.
        """
        x1, y1, x2, y2 = face_bbox
        
        # Extract face regions
        result_face = result[y1:y2, x1:x2]
        ref_face = reference[y1:y2, x1:x2]
        
        # LAB color transfer
        corrected_face = self._lab_color_transfer(result_face, ref_face)
        
        # Apply gamma correction for lighting
        corrected_face = self._adaptive_gamma_correction(corrected_face, ref_face)
        
        # Paste back
        result = result.copy()
        result[y1:y2, x1:x2] = corrected_face
        
        return result
    
    @staticmethod
    def _lab_color_transfer(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Transfer color from target to source in LAB space."""
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Calculate statistics
        s_mean, s_std = source_lab.mean(axis=(0, 1)), source_lab.std(axis=(0, 1))
        t_mean, t_std = target_lab.mean(axis=(0, 1)), target_lab.std(axis=(0, 1))
        
        # Transfer
        result_lab = (source_lab - s_mean) * (t_std / (s_std + 1e-6)) + t_mean
        result_lab = np.clip(result_lab, [0, -128, -128], [255, 127, 127])
        
        return cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    @staticmethod
    def _adaptive_gamma_correction(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Adapt brightness/contrast of source to match target."""
        s_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        t_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        
        s_brightness = np.mean(s_gray)
        t_brightness = np.mean(t_gray)
        
        if s_brightness == 0:
            return source
        
        gamma = np.log(t_brightness / 255.0) / np.log(s_brightness / 255.0)
        gamma = np.clip(gamma, 0.5, 2.0)
        
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 
                         for i in range(256)]).astype(np.uint8)
        
        return cv2.LUT(source, table)
    
    @staticmethod
    def _apply_soft_sharpening(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """Apply subtle unsharp masking for detail enhancement."""
        # Create unsharp mask
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        sharpened = cv2.addWeighted(image, 1.0 + strength, gaussian, -strength, 0)
        
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    @staticmethod
    def _apply_basic_enhancement(face_crop: np.ndarray) -> np.ndarray:
        """
        Fallback basic enhancement when model fails.
        Safe operations: bilateral filter + slight sharpening.
        """
        # Bilateral filter: smooth while preserving edges
        enhanced = cv2.bilateralFilter(face_crop, 9, 75, 75)
        
        # Slight unsharp mask for clarity
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        enhanced = cv2.addWeighted(enhanced, 1.1, gaussian, -0.1, 0)
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)

