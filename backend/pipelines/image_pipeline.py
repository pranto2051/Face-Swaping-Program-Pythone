import numpy as np
from processors.face_detector import MultiFaceDetector
from processors.expression_match import ExpressionMatcher
from processors.face_swapper import FaceSwapper
from processors.face_enhancer import FaceEnhancer, EnhancementConfig
from typing import Optional, Dict, Any

class ImagePipeline:
    def __init__(self, detector: MultiFaceDetector, matcher: ExpressionMatcher, 
                 swapper: FaceSwapper, enhancer: Optional[FaceEnhancer] = None):
        self.detector = detector
        self.matcher = matcher
        self.swapper = swapper
        self.enhancer = enhancer
        
    def _get_quality_mode(self, mode: str) -> str:
        """Map legacy modes to quality modes."""
        mode_mapping = {
            'fast': 'fast',
            'studio': 'high_quality',
            'cinematic': 'ultra_realistic'
        }
        return mode_mapping.get(mode, 'high_quality')

    def process_swap(self, source_img: np.ndarray, target_img: np.ndarray, 
                     assignments: list, settings: dict):
        """
        Enhanced Face Swap Pipeline with Ultra-Realistic Post-Processing
        
        Quality Modes:
        - fast: Standard InsightFace swap (no enhancement)
        - studio/high_quality: Super resolution + color correction + seamless blending
        - cinematic/ultra_realistic: Full pipeline with all enhancements
        
        Processing Order:
        1. Detect faces
        2. Swap faces (InsightFace)
        3. Apply enhancement pipeline (if enabled):
           - Super Resolution (GFPGAN/CodeFormer)
           - Color & Lighting Correction
           - Seamless Blending (Poisson)
           - Detail Restoration
           - Noise Matching
        """
        import cv2

        mode = settings.get('mode', 'fast')
        swap_mode = self._resolve_swap_mode(settings)
        quality_mode = self._get_quality_mode(mode)
        
        # Detect faces
        source_faces = self.detector.app.get(source_img)
        target_faces = self.detector.app.get(target_img)
        
        if not source_faces or not target_faces:
            print("DEBUG: No faces detected in one of the images")
            return target_img.copy()

        # Sort faces by size for consistent ID assignment
        source_faces = sorted(source_faces, key=lambda f: self.detector._bbox_area(f.bbox), reverse=True)
        target_faces = sorted(target_faces, key=lambda f: self.detector._bbox_area(f.bbox), reverse=True)
        
        result_img = target_img.copy()

        # Process each face swap assignment
        for assignment in assignments:
            t_id = assignment.get('target_face_id', assignment.get('targetFaceId'))
            s_id = assignment.get('source_face_id', assignment.get('sourceFaceId'))
            
            if t_id is None or s_id is None:
                continue
                
            if t_id >= len(target_faces) or s_id >= len(source_faces):
                print(f"Warning: Invalid face ID (target: {t_id}/{len(target_faces)}, source: {s_id}/{len(source_faces)})")
                continue
            
            t_face = target_faces[t_id]
            s_face = source_faces[s_id]

            if swap_mode == 'raw_copy':
                # Strict mode: affine-only placement, no expression/color/diffusion refinement.
                result_img = self._swap_raw_copy(result_img, source_img, s_face, t_face, settings)
                continue

            # Adaptive mode: regular face swap behavior.
            result_img = self.swapper.swap(result_img, t_face, s_face)

            # Enhancement is allowed only in adaptive mode.
            if quality_mode != 'fast' and self.enhancer is not None:
                bbox = self._get_face_bbox(t_face)
                face_mask = self._get_face_mask(t_face, result_img.shape)
                result_img = self.enhancer.enhance_face(
                    swapped_img=result_img,
                    original_img=target_img,
                    face_bbox=bbox,
                    face_mask=face_mask
                )
            elif quality_mode != 'fast':
                result_img = self._apply_basic_enhancement(result_img, target_img, mode)

        return result_img

    def _resolve_swap_mode(self, settings: Dict[str, Any]) -> str:
        """Resolve adaptive vs strict raw-copy behavior from request settings."""
        mode_value = str(settings.get('mode', 'fast')).lower()
        explicit_swap_mode = str(settings.get('swap_mode', '')).lower()
        use_target_expression = settings.get('use_target_expression')

        if mode_value in ('adaptive', 'raw_copy'):
            return mode_value
        if explicit_swap_mode in ('adaptive', 'raw_copy'):
            return explicit_swap_mode
        if isinstance(use_target_expression, bool):
            return 'adaptive' if use_target_expression else 'raw_copy'
        return 'adaptive'

    def _swap_raw_copy(self, base_img: np.ndarray, source_img: np.ndarray, source_face, target_face,
                       settings: Dict[str, Any]) -> np.ndarray:
        """
        Raw copy mode: rigid affine transform + feather blend only.
        No expression transfer, color matching, or non-linear warping.
        """
        import cv2

        source_kps = self._get_five_point_landmarks(source_face)
        target_kps = self._get_five_point_landmarks(target_face)
        if source_kps is None or target_kps is None:
            return base_img

        src_tri = np.float32(source_kps[:3])
        dst_tri = np.float32(target_kps[:3])
        affine_matrix = cv2.getAffineTransform(src_tri, dst_tri)

        h, w = base_img.shape[:2]
        warped_source = cv2.warpAffine(
            source_img,
            affine_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        source_mask = self._build_source_face_mask(source_face, source_img.shape)
        warped_mask = cv2.warpAffine(
            source_mask,
            affine_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        feather = int(settings.get('mask_feather', 10))
        feather = max(1, min(feather, 50))
        blur_kernel = max(3, feather * 2 + 1)
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        warped_mask = cv2.GaussianBlur(warped_mask, (blur_kernel, blur_kernel), 0)

        alpha = (warped_mask.astype(np.float32) / 255.0)[..., None]
        blended = (warped_source.astype(np.float32) * alpha) + (base_img.astype(np.float32) * (1.0 - alpha))
        return np.clip(blended, 0, 255).astype(np.uint8)

    @staticmethod
    def _get_five_point_landmarks(face) -> Optional[np.ndarray]:
        """Return InsightFace 5-point landmarks when available."""
        if not hasattr(face, 'kps') or face.kps is None:
            return None
        kps = np.asarray(face.kps, dtype=np.float32)
        if kps.shape[0] < 5:
            return None
        return kps[:5]

    def _build_source_face_mask(self, face, source_shape) -> np.ndarray:
        """Create a source-space binary mask to support rigid copy blending."""
        import cv2

        mask = np.zeros(source_shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = self._get_face_bbox(face)

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        rx = max(1, int((x2 - x1) * 0.55))
        ry = max(1, int((y2 - y1) * 0.62))
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

        return mask
    
    def _get_face_bbox(self, face) -> tuple:
        """Extract bounding box from face object."""
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        
        # Ensure bbox is within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        
        return (x1, y1, x2, y2)
    
    def _get_face_mask(self, face, img_shape) -> np.ndarray:
        """Generate face mask from landmarks."""
        import cv2
        
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        
        # Use face landmarks if available
        if hasattr(face, 'kps') and face.kps is not None:
            landmarks = face.kps.astype(np.int32)
            
            # Create convex hull around landmarks
            hull = cv2.convexHull(landmarks)
            
            # Expand hull slightly for better blending
            center = hull.mean(axis=0)
            expanded_hull = center + (hull - center) * 1.1
            expanded_hull = expanded_hull.astype(np.int32)
            
            # Fill the hull
            cv2.fillConvexPoly(mask, expanded_hull, 255)
        else:
            # Fallback: use bounding box
            bbox = self._get_face_bbox(face)
            x1, y1, x2, y2 = bbox
            
            # Create elliptical mask (more natural than rectangle)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            axes = ((x2 - x1) // 2, (y2 - y1) // 2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        return mask
    
    def _apply_basic_enhancement(self, result_img: np.ndarray, 
                                 target_img: np.ndarray, mode: str) -> np.ndarray:
        """
        Fallback basic enhancement when FaceEnhancer is not available.
        This is the legacy enhancement logic.
        """
        import cv2
        
        if mode in ['studio', 'cinematic']:
            # Basic sharpening (Unsharp Mask)
            gaussian_blur = cv2.GaussianBlur(result_img, (0, 0), 3)
            result_img = cv2.addWeighted(result_img, 1.5, gaussian_blur, -0.5, 0)
        
        if mode == 'cinematic':
            # Basic color matching
            target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB)
            result_lab = cv2.cvtColor(result_img, cv2.COLOR_BGR2LAB)
            
            l_t, a_t, b_t = cv2.split(target_lab)
            l_r, a_r, b_r = cv2.split(result_lab)
            
            l_match = cv2.addWeighted(l_r, 0.7, l_t, 0.3, 0)
            result_lab = cv2.merge([l_match, a_r, b_r])
            result_img = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
        
        return result_img
