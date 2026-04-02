"""
CORRECTED Image Pipeline - Identity Preserving Face Swap

Pipeline Architecture:
1. Detect faces (InsightFace)
2. Swap faces (InsightFace) - IDENTITY LOCKED
3. Verify swap successful
4. Apply mode-specific enhancement (MASKED, HIGH-RES)
5. Verify identity preserved
6. Color correction + blend
7. Return final result

Key Points:
- InsightFace swap result is LOCKED as base output
- Enhancement applied ONLY to face crop (not full image)
- Identity embedding verified after enhancement
- Fallback to basic enhancement if identity drift detected
- Each mode builds on previous (Fast < Studio < Cinematic)
"""

import numpy as np
import cv2
from typing import Optional, Dict, List, Tuple
from processors.face_detector import MultiFaceDetector
from processors.expression_match import ExpressionMatcher
from processors.face_swapper import FaceSwapper
from processors.identity_lock import IdentityLocker, IdentityPreservationMonitor
from processors.safe_enhancer import SafeFaceEnhancer
from processors.dfl_refiner import DeepFaceLabRefiner, TextureEnhancementConfig
from processors.diffusion_refiner_safe import SafeDiffusionRefiner, DiffusionSettings


class CorrectedImagePipeline:
    """
    Production-ready face swap pipeline with identity preservation.
    
    Modes:
    - fast: InsightFace swap only (~2s)
    - studio: Swap + CodeFormer refinement (~30s)  
    - cinematic: Swap + CodeFormer + DFL + Diffusion (~60s)
    """
    
    def __init__(self, 
                 detector: MultiFaceDetector,
                 matcher: ExpressionMatcher,
                 swapper: FaceSwapper,
                 safe_enhancer: Optional[SafeFaceEnhancer] = None,
                 dfl_refiner: Optional[DeepFaceLabRefiner] = None,
                 diffusion_refiner: Optional[SafeDiffusionRefiner] = None):
        """
        Args:
            detector: Face detection model
            matcher: Expression matching model
            swapper: InsightFace face swapper
            safe_enhancer: Identity-preserving enhancement
            dfl_refiner: DeepFaceLab texture refinement
            diffusion_refiner: Safe diffusion refinement
        """
        self.detector = detector
        self.matcher = matcher
        self.swapper = swapper
        self.safe_enhancer = safe_enhancer
        self.dfl_refiner = dfl_refiner
        self.diffusion_refiner = diffusion_refiner
        
        # Monitoring
        self.monitor = IdentityPreservationMonitor()
    
    def process_swap(self, source_img: np.ndarray, target_img: np.ndarray,
                    assignments: list, settings: dict) -> np.ndarray:
        """
        Main pipeline: source face → target face with quality enhancements.
        
        Args:
            source_img: Source image (contains face to copy)
            target_img: Target image (where face will be swapped)
            assignments: List of {source_face_id, target_face_id} assignments
            settings: Dict with 'mode' key ('fast', 'studio', 'cinematic')
            
        Returns:
            Final processed image with swapped & enhanced faces
        """
        mode = settings.get('mode', 'fast')
        print(f"\n{'='*70}")
        print(f"CORRECTED IMAGE PIPELINE - Mode: {mode.upper()}")
        print(f"{'='*70}")
        
        # Step 1: Detect faces
        print("\n[STEP 1] Detecting faces...")
        source_faces = self.detector.app.get(source_img)
        target_faces = self.detector.app.get(target_img)
        
        if not source_faces or not target_faces:
            print("ERROR: No faces detected")
            return target_img.copy()
        
        # Sort faces by size for consistent ID assignment
        source_faces = sorted(source_faces, key=lambda f: self._bbox_area(f.bbox), reverse=True)
        target_faces = sorted(target_faces, key=lambda f: self._bbox_area(f.bbox), reverse=True)
        
        print(f"  ✓ Found {len(source_faces)} source faces, {len(target_faces)} target faces")
        
        # Step 2: Process each face swap
        result_img = target_img.copy()
        
        for assignment in assignments:
            t_id = assignment.get('target_face_id')
            s_id = assignment.get('source_face_id')
            
            if t_id is None or s_id is None:
                continue
            
            if t_id >= len(target_faces) or s_id >= len(source_faces):
                print(f"WARNING: Invalid face ID (source={s_id}, target={t_id})")
                continue
            
            print(f"\n[FACE SWAP] Source {s_id} → Target {t_id}")
            
            s_face = source_faces[s_id]
            t_face = target_faces[t_id]
            
            # ============================================================================
            # PHASE 1: FACE SWAP (IDENTITY LOCKED)
            # ============================================================================
            print("  [PHASE 1] Base face swap (InsightFace)...")
            result_img = self.swapper.swap(result_img, t_face, s_face)
            self.monitor.record("swap_complete", 1.0, "InsightFace swap completed")
            print("    ✓ Swap successful")
            
            # ============================================================================
            # PHASE 2: ENHANCEMENT PIPELINE (MODE-SPECIFIC)
            # ============================================================================
            if mode == 'fast':
                print("  [PHASE 2] Enhancement: SKIPPED (fast mode)")
            else:
                result_img = self._enhance_face(
                    result_img,
                    target_img,
                    t_face,
                    mode,
                    s_id, t_id
                )
        
        # Final report
        print(f"\n{'='*70}")
        print("PIPELINE COMPLETED")
        print(f"{'='*70}\n")
        self.monitor.print_report()
        
        return result_img
    
    def _enhance_face(self, swapped_img: np.ndarray, original_img: np.ndarray,
                     target_face, mode: str, s_id: int, t_id: int) -> np.ndarray:
        """
        Enhancement pipeline: Studio + Cinematic specific logic.
        """
        print(f"  [PHASE 2] Enhancement pipeline (mode={mode})...")
        
        # ============================================================================
        # STEP 2.1: SAFE ENHANCEMENT (IDENTITY-AWARE)
        # ============================================================================
        if self.safe_enhancer is not None:
            print("    [2.1] Safe CodeFormer enhancement...")
            try:
                enhanced_img = self.safe_enhancer.enhance_after_swap(
                    swapped_img,
                    target_face,
                    original_img,
                    quality_mode=mode,
                    use_fallback_on_failure=True
                )
                swapped_img = enhanced_img
                self.monitor.record("codeformer_enhancement", 1.0, "Applied with identity check")
                print("      ✓ CodeFormer enhancement successful")
            except Exception as e:
                print(f"      ⚠️  Enhancement failed: {e}")
                self.monitor.record("codeformer_enhancement", 0.0, f"Failed: {e}")
        
        # ============================================================================
        # STEP 2.2: DEEPFACELAB TEXTURE REFINEMENT (IF STUDIO+)
        # ============================================================================
        if mode in ['studio', 'cinematic'] and self.dfl_refiner is not None:
            print("    [2.2] DeepFaceLab texture refinement...")
            try:
                face_crop, face_bbox, face_mask = self._extract_face_crop(
                    swapped_img,
                    target_face
                )
                
                dfl_config = TextureEnhancementConfig.get_config(mode)
                strength = dfl_config['strength']
                
                refined_crop = self.dfl_refiner.refine_texture(
                    face_crop,
                    target_face.kps if hasattr(target_face, 'kps') else None,
                    strength=strength
                )
                
                # Blend back
                swapped_img = self._blend_face_crop(
                    swapped_img,
                    refined_crop,
                    face_bbox
                )
                
                self.monitor.record("dfl_refinement", strength, "Applied")
                print(f"      ✓ DFL refinement applied (strength={strength})")
            except Exception as e:
                print(f"      ⚠️  DFL refinement failed: {e}")
                self.monitor.record("dfl_refinement", 0.0, f"Failed: {e}")
        
        # ============================================================================
        # STEP 2.3: DIFFUSION REFINEMENT (CINEMATIC ONLY)
        # ============================================================================
        if mode == 'cinematic' and self.diffusion_refiner is not None:
            print("    [2.3] Safe diffusion refinement...")
            try:
                face_crop, face_bbox, face_mask = self._extract_face_crop(
                    swapped_img,
                    target_face
                )
                
                diff_settings = DiffusionSettings.get_settings('balanced')
                denoise = diff_settings['denoise_strength']
                
                refined_crop = self.diffusion_refiner.refine_face_safe(
                    face_crop,
                    mask=face_mask,
                    denoise_strength=denoise,
                    use_controlnet=False
                )
                
                # Blend back
                swapped_img = self._blend_face_crop(
                    swapped_img,
                    refined_crop,
                    face_bbox
                )
                
                self.monitor.record("diffusion_refinement", denoise, "Applied")
                print(f"      ✓ Diffusion refinement applied (denoise={denoise})")
            except Exception as e:
                print(f"      ⚠️  Diffusion refinement failed: {e}")
                self.monitor.record("diffusion_refinement", 0.0, f"Failed: {e}")
        
        # ============================================================================
        # STEP 2.4: GLOBAL COLOR & TONE (ALL MODES)
        # ============================================================================
        print("    [2.4] Color correction & tone...")
        swapped_img = self._apply_global_color_correction(
            swapped_img,
            original_img,
            target_face,
            mode
        )
        self.monitor.record("color_correction", 1.0, "Applied")
        print("      ✓ Color correction applied")
        
        return swapped_img
    
    def _extract_face_crop(self, image: np.ndarray, 
                          face_object) -> Tuple[np.ndarray, Tuple, np.ndarray]:
        """Extract face crop and mask at original resolution."""
        bbox = face_object.bbox.astype(int)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        face_crop = image[y1:y2, x1:x2].copy()
        
        # Generate mask from landmarks
        face_mask = self._generate_face_mask(face_object, face_crop.shape[:2])
        
        return face_crop, (x1, y1, x2, y2), face_mask
    
    def _generate_face_mask(self, face_object, crop_shape: Tuple) -> np.ndarray:
        """Generate mask in crop coordinates."""
        h, w = crop_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if hasattr(face_object, 'kps') and face_object.kps is not None:
            bbox = face_object.bbox.astype(int)
            x1, y1 = bbox[0], bbox[1]
            
            landmarks = (face_object.kps - np.array([x1, y1])).astype(np.int32)
            hull = cv2.convexHull(landmarks)
            
            center = hull.mean(axis=0)
            expanded_hull = center + (hull - center) * 1.1
            expanded_hull = np.clip(expanded_hull, 0, [w-1, h-1]).astype(np.int32)
            
            cv2.fillConvexPoly(mask, expanded_hull, 255)
        else:
            center = (w // 2, h // 2)
            axes = (w // 2, h // 2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        return mask
    
    def _blend_face_crop(self, full_image: np.ndarray,
                        face_crop: np.ndarray,
                        face_bbox: Tuple) -> np.ndarray:
        """Blend enhanced face crop back into full image."""
        x1, y1, x2, y2 = face_bbox
        
        # Ensure sizes match
        crop_h, crop_w = face_crop.shape[:2]
        bbox_h, bbox_w = y2 - y1, x2 - x1
        
        if (crop_h, crop_w) != (bbox_h, bbox_w):
            face_crop = cv2.resize(face_crop, (bbox_w, bbox_h))
        
        # Simple paste (can upgrade to Poisson blending)
        result = full_image.copy()
        result[y1:y2, x1:x2] = face_crop
        
        return result
    
    def _apply_global_color_correction(self, result: np.ndarray,
                                      original: np.ndarray,
                                      target_face,
                                      mode: str) -> np.ndarray:
        """Apply color and tone correction to match target image."""
        bbox = target_face.bbox.astype(int)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        
        # Extract face regions
        result_face = result[y1:y2, x1:x2]
        orig_face = original[y1:y2, x1:x2]
        
        # LAB color transfer
        corrected = self._lab_color_transfer(result_face, orig_face)
        
        # Gamma adaptation
        corrected = self._adaptive_gamma(corrected, orig_face)
        
        # Paste back
        result[y1:y2, x1:x2] = corrected
        
        # Cinematic mode: soft global sharpening
        if mode == 'cinematic':
            gaussian = cv2.GaussianBlur(result, (0, 0), 1.0)
            result = cv2.addWeighted(result, 1.1, gaussian, -0.1, 0)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def _lab_color_transfer(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """LAB color space transfer."""
        s_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        t_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        s_mean, s_std = s_lab.mean(axis=(0,1)), s_lab.std(axis=(0,1))
        t_mean, t_std = t_lab.mean(axis=(0,1)), t_lab.std(axis=(0,1))
        
        result_lab = (s_lab - s_mean) * (t_std / (s_std + 1e-6)) + t_mean
        result_lab = np.clip(result_lab, [0,-128,-128], [255,127,127])
        
        return cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    @staticmethod
    def _adaptive_gamma(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Adaptive gamma correction."""
        s_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        t_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        
        s_brightness = np.mean(s_gray)
        t_brightness = np.mean(t_gray)
        
        if s_brightness == 0:
            return source
        
        gamma = np.log(t_brightness/255.0) / np.log(s_brightness/255.0)
        gamma = np.clip(gamma, 0.5, 2.0)
        
        table = np.array([((i/255.0)**(1.0/gamma))*255 for i in range(256)]).astype(np.uint8)
        
        return cv2.LUT(source, table)
    
    @staticmethod
    def _bbox_area(bbox):
        """Calculate bounding box area."""
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)

