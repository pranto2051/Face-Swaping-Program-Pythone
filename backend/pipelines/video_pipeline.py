from processors.face_detector import MultiFaceDetector
from processors.expression_match import ExpressionMatcher
from processors.diffusion_refiner import DiffusionRefiner
from processors.face_enhancer import FaceEnhancer
from services.video_optimizer import VideoOptimizer
from typing import Optional

class CinematicVideoPipeline:
    """
    V3 Video Pipeline with Ultra-Realistic Enhancement:
    1. Temporal Coherence: Track faces and smooth parameters.
    2. Ultra-Realistic Enhancement: Super-resolution, color correction, detail restoration.
    3. Cinematic Color Grade: Apply LUTs and grain.
    4. Multi-Pass Rendering: Detect -> Swap -> Enhance -> Grade -> Encode.
    """
    
    def __init__(self, detector, matcher, swapper, enhancer: Optional[FaceEnhancer] = None):
        self.detector = detector
        self.matcher = matcher
        self.swapper = swapper
        self.enhancer = enhancer
        # Optional: Initialize refiner and optimizer if they are ready
        # self.optimizer = VideoOptimizer()

    def process(self, video_path, source_img, face_assignments, params, progress_callback=None):
        """
        High-Performance Video Swapping with Enhanced Quality Modes:
        - fast: Standard InsightFace (no enhancement)
        - studio: Fast + Unsharp Mask + color correction
        - cinematic: Full enhancement pipeline (slower but ultra-realistic)
        
        Note: For video, 'studio' mode is recommended for best quality/speed balance.
        'cinematic' mode may be slow for long videos.
        """
        import cv2
        import numpy as np
        import os
        import uuid
        from moviepy.editor import VideoFileClip, ImageSequenceClip
        
        mode = params.get('mode', 'fast')
        swap_mode = self._resolve_swap_mode(params)
        
        # Map video modes to enhancer quality modes
        quality_mode_map = {
            'fast': 'fast',
            'studio': 'high_quality',
            'cinematic': 'ultra_realistic'
        }
        quality_mode = quality_mode_map.get(mode, 'fast')
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Could not open video file")
            
        fps = cap.get(cv2.CAP_PROP_FPS) if hasattr(cv2, 'CAP_PROP_FPS') else cap.get(5)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"🎬 VIDEO PROCESSING STARTED")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Total Frames: {total_frames}")
        print(f"   Mode: {mode}")
        
        frame_skip = params.get('frame_skip', 1)
        
        source_faces = self.detector.app.get(source_img)
        if not source_faces:
            raise Exception("No face detected in source image")
        source_faces = sorted(source_faces, key=lambda f: self.detector._bbox_area(f.bbox), reverse=True)
        
        count = 0
        processed_count = 0
        
        temp_dir = f"temp_frames_{uuid.uuid4()}"
        os.makedirs(temp_dir, exist_ok=True)
        
        bridge = params.get('bridge')
        job_id = params.get('job_id')
        stop_event = params.get('stop_event')
        
        original_clip = None
        new_clip = None

        try:
            while cap.isOpened():
                if stop_event and stop_event.is_set():
                    return None
                    
                ret, frame = cap.read()
                if not ret:
                    break
                
                if count % frame_skip == 0:
                    try:
                        target_faces = self.detector.app.get(frame)
                        
                        if not target_faces:
                            swapped_frame = frame
                        else:
                            target_faces = sorted(target_faces, key=lambda f: self.detector._bbox_area(f.bbox), reverse=True)
                            swapped_frame = frame.copy()
                            
                            for target_id, source_id in face_assignments:
                                if target_id >= len(target_faces) or source_id >= len(source_faces):
                                    continue
                                    
                                target_face = target_faces[target_id]
                                source_face = source_faces[source_id]
                                
                                if swap_mode == 'raw_copy':
                                    swapped_frame = self._swap_raw_copy_frame(
                                        swapped_frame,
                                        source_img,
                                        source_face,
                                        target_face,
                                        params,
                                    )
                                else:
                                    # 1. BASE SWAP
                                    swapped_frame = self.swapper.swap(
                                        swapped_frame,
                                        target_face,
                                        source_face
                                    )

                                    # 2. ENHANCEMENT PIPELINE (if enhancer available and mode != fast)
                                    if quality_mode != 'fast' and self.enhancer is not None:
                                        try:
                                            # Extract face bbox for enhancement
                                            bbox = self._get_face_bbox(target_face)
                                            face_mask = self._get_face_mask(target_face, swapped_frame.shape)

                                            # Apply enhancement (lightweight for video)
                                            swapped_frame = self.enhancer.enhance_face(
                                                swapped_img=swapped_frame,
                                                original_img=frame,
                                                face_bbox=bbox,
                                                face_mask=face_mask
                                            )
                                        except Exception as e:
                                            print(f"Enhancement failed on frame {count}, using fallback: {e}")
                                            # Fallback to basic enhancement
                                            if mode in ['studio', 'cinematic']:
                                                gb = cv2.GaussianBlur(swapped_frame, (0, 0), 2)
                                                swapped_frame = cv2.addWeighted(swapped_frame, 1.3, gb, -0.3, 0)

                                    # 3. LEGACY FALLBACK (if no enhancer)
                                    elif quality_mode != 'fast':
                                        # STUDIO ENHANCEMENT
                                        if mode in ['studio', 'cinematic']:
                                            gb = cv2.GaussianBlur(swapped_frame, (0, 0), 2)
                                            swapped_frame = cv2.addWeighted(swapped_frame, 1.3, gb, -0.3, 0)

                                        # CINEMATIC ENHANCEMENT
                                        if mode == 'cinematic':
                                            t_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                                            s_lab = cv2.cvtColor(swapped_frame, cv2.COLOR_BGR2LAB)
                                            l_t, a_t, b_t = cv2.split(t_lab)
                                            l_s, a_s, b_s = cv2.split(s_lab)
                                            l_match = cv2.addWeighted(l_s, 0.8, l_t, 0.2, 0)
                                            res_lab = cv2.merge([l_match, a_s, b_s])
                                            swapped_frame = cv2.cvtColor(res_lab, cv2.COLOR_LAB2BGR)

                        frame_path = os.path.join(temp_dir, f"frame_{processed_count:06d}.jpg")
                        cv2.imwrite(frame_path, swapped_frame)
                        processed_count += 1
                    except Exception as e:
                        frame_path = os.path.join(temp_dir, f"frame_{processed_count:06d}.jpg")
                        cv2.imwrite(frame_path, frame)
                        processed_count += 1
                
                count += 1
                if bridge and job_id and count % 20 == 0:
                    percent = int((count / total_frames) * 100)
                    bridge.emit_progress(job_id, "swapping", percent, current=count, total=total_frames)
                    print(f"   ⏳ Processed {count}/{total_frames} frames ({percent}%)")
            
            cap.release()
            
            if stop_event and stop_event.is_set():
                return None

            if bridge and job_id:
                bridge.emit_progress(job_id, "encoding", 95)
            
            print(f"   🎞️  Encoding video with {processed_count} frames...")

            # --- Reassemble video ---
            output_filename = f"video_result_{uuid.uuid4()}.mp4"
            output_path = os.path.join('static', output_filename)
            
            frame_files = [os.path.join(temp_dir, f) for f in sorted(os.listdir(temp_dir)) if f.endswith('.jpg')]
            new_clip = ImageSequenceClip(frame_files, fps=fps)
            
            original_clip = VideoFileClip(video_path)
            if original_clip.audio is not None:
                new_clip = new_clip.set_audio(original_clip.audio)
                
            new_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
            
            print(f"   ✅ VIDEO COMPLETE: {output_filename}")
            
            return output_filename
        finally:
            # Cleanup temp frames
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            if original_clip:
                original_clip.close()
            if 'new_clip' in locals():
                new_clip.close()
    
    def _get_face_bbox(self, face) -> tuple:
        """Extract bounding box from face object."""
        bbox = face.bbox.astype(int) if hasattr(face.bbox, 'astype') else face.bbox
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        # Ensure bbox is within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        
        return (x1, y1, x2, y2)

    def _resolve_swap_mode(self, params) -> str:
        mode_value = str(params.get('mode', 'fast')).lower()
        explicit_swap_mode = str(params.get('swap_mode', '')).lower()
        use_target_expression = params.get('use_target_expression')

        if mode_value in ('adaptive', 'raw_copy'):
            return mode_value
        if explicit_swap_mode in ('adaptive', 'raw_copy'):
            return explicit_swap_mode
        if isinstance(use_target_expression, bool):
            return 'adaptive' if use_target_expression else 'raw_copy'
        return 'adaptive'

    def _swap_raw_copy_frame(self, base_frame, source_img, source_face, target_face, params):
        import cv2
        import numpy as np

        if not hasattr(source_face, 'kps') or source_face.kps is None:
            return base_frame
        if not hasattr(target_face, 'kps') or target_face.kps is None:
            return base_frame

        source_kps = np.asarray(source_face.kps, dtype=np.float32)
        target_kps = np.asarray(target_face.kps, dtype=np.float32)
        if source_kps.shape[0] < 5 or target_kps.shape[0] < 5:
            return base_frame

        affine_matrix = cv2.getAffineTransform(source_kps[:3], target_kps[:3])
        h, w = base_frame.shape[:2]
        warped_source = cv2.warpAffine(
            source_img,
            affine_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        mask = np.zeros(source_img.shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = self._get_face_bbox(source_face)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        rx, ry = max(1, int((x2 - x1) * 0.55)), max(1, int((y2 - y1) * 0.62))
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

        warped_mask = cv2.warpAffine(mask, affine_matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)

        feather = int(params.get('mask_feather', 10))
        feather = max(1, min(feather, 50))
        blur_kernel = max(3, feather * 2 + 1)
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        warped_mask = cv2.GaussianBlur(warped_mask, (blur_kernel, blur_kernel), 0)

        alpha = (warped_mask.astype(np.float32) / 255.0)[..., None]
        out = warped_source.astype(np.float32) * alpha + base_frame.astype(np.float32) * (1.0 - alpha)
        return np.clip(out, 0, 255).astype(np.uint8)
    
    def _get_face_mask(self, face, img_shape) -> 'np.ndarray':
        """Generate face mask from landmarks."""
        import cv2
        import numpy as np
        
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
