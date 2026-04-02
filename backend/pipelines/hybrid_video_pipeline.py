"""
Hybrid Video Face Swap Pipeline
Processes video frames with the hybrid face swap pipeline
"""
import logging
import numpy as np
import cv2
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import tempfile
import subprocess
import os
from core.mode_controller import ProcessingMode
from pipelines.hybrid_face_swap import HybridFaceSwapPipeline

logger = logging.getLogger(__name__)


class HybridVideoFaceSwapPipeline:
    """Video processing pipeline for face swapping"""
    
    def __init__(self, mode: ProcessingMode = ProcessingMode.STUDIO):
        """
        Initialize video pipeline
        
        Args:
            mode: Processing mode
        """
        self.mode = mode
        self.pipeline = HybridFaceSwapPipeline(mode)
        self.temp_dir = Path(tempfile.gettempdir()) / 'face_swap_video'
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def process_video(
        self,
        video_path: str,
        source_image_path: str,
        output_path: str,
        source_face_id: Optional[int] = None,
        target_face_id: Optional[int] = None,
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Process video file with face swapping
        
        Args:
            video_path: Path to input video
            source_image_path: Path to source image with face
            output_path: Path for output video
            source_face_id: Specific source face
            target_face_id: Specific target face
            callback: Progress callback
            
        Returns:
            Processing metadata
        """
        try:
            # Load source image
            source_image = cv2.imread(source_image_path)
            if source_image is None:
                return {'success': False, 'reason': 'Failed to load source image'}
            
            # Extract video information
            video_info = self._get_video_info(video_path)
            if not video_info['success']:
                return video_info
            
            total_frames = video_info['frame_count']
            fps = video_info['fps']
            width = video_info['width']
            height = video_info['height']
            has_audio = video_info['has_audio']
            audio_path = None
            
            if callback:
                callback(0, "Extracting video information")
            
            logger.info(f"Processing video: {total_frames} frames at {fps} FPS ({width}x{height})")
            
            # Stage 1: Extract frames
            frames_dir = self.temp_dir / f'frames_{os.urandom(8).hex()}'
            frames_dir.mkdir(parents=True, exist_ok=True)
            
            frame_list = self._extract_frames(video_path, frames_dir, total_frames, callback)
            
            if not frame_list:
                return {'success': False, 'reason': 'Failed to extract frames'}
            
            if callback:
                callback(15, "Processing frames with face swap")
            
            # Stage 2: Process frames
            processed_frames = []
            for idx, frame_path in enumerate(frame_list):
                progress = 15 + int((idx / len(frame_list)) * 70)
                if callback:
                    callback(progress, f"Processing frame {idx + 1}/{len(frame_list)}")
                
                frame = cv2.imread(frame_path)
                if frame is None:
                    logger.warning(f"Failed to read frame: {frame_path}")
                    processed_frames.append(frame_path)
                    continue
                
                # Swap faces
                swapped, meta = self.pipeline.process(
                    source_image,
                    frame,
                    source_face_id,
                    target_face_id,
                    callback=None,  # Don't use callback for individual frames
                )
                
                if meta.get('success'):
                    # Save processed frame
                    output_frame_path = frames_dir / f'processed_{idx:06d}.png'
                    cv2.imwrite(str(output_frame_path), swapped)
                    processed_frames.append(str(output_frame_path))
                else:
                    # Use original frame if swap failed
                    processed_frames.append(frame_path)
            
            if callback:
                callback(85, "Rebuilding video from frames")
            
            # Stage 3: Extract audio (if exists)
            if has_audio:
                audio_path = self._extract_audio(video_path, self.temp_dir)
            
            # Stage 4: Rebuild video
            success = self._rebuild_video(
                processed_frames,
                output_path,
                fps,
                width,
                height,
                audio_path,
            )
            
            if callback:
                callback(95, "Finalizing video")
            
            # Cleanup
            self._cleanup(frames_dir, audio_path if audio_path else None)
            
            if callback:
                callback(100, "Video processing completed")
            
            return {
                'success': success,
                'output_path': output_path,
                'total_frames': len(frame_list),
                'fps': fps,
                'resolution': f'{width}x{height}',
            }
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            return {'success': False, 'reason': str(e)}
    
    def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video file information"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return {'success': False, 'reason': 'Cannot open video file'}
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            cap.release()
            
            # Check for audio using ffprobe
            has_audio = self._check_audio_stream(video_path)
            
            return {
                'success': True,
                'frame_count': frame_count,
                'fps': fps,
                'width': width,
                'height': height,
                'has_audio': has_audio,
            }
            
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            return {'success': False, 'reason': str(e)}
    
    def _check_audio_stream(self, video_path: str) -> bool:
        """Check if video has audio stream"""
        try:
            import subprocess
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                 '-show_entries', 'stream=codec_type', '-of', 'default=noprint_wrappers=1:nokey=1:0',
                 video_path],
                capture_output=True,
                text=True,
                timeout=5
            )
            return 'audio' in result.stdout.lower()
        except:
            return False
    
    def _extract_frames(
        self,
        video_path: str,
        output_dir: Path,
        total_frames: int,
        callback: Optional[Callable] = None,
    ) -> list:
        """Extract frames from video"""
        try:
            cap = cv2.VideoCapture(video_path)
            frame_list = []
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if callback and frame_idx % 10 == 0:
                    progress = int((frame_idx / total_frames) * 15)
                    callback(progress, f"Extracting frame {frame_idx}/{total_frames}")
                
                frame_path = output_dir / f'frame_{frame_idx:06d}.png'
                cv2.imwrite(str(frame_path), frame)
                frame_list.append(str(frame_path))
                
                frame_idx += 1
            
            cap.release()
            logger.info(f"✓ Extracted {len(frame_list)} frames")
            return frame_list
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            return []
    
    def _extract_audio(self, video_path: str, output_dir: Path) -> Optional[str]:
        """Extract audio from video using ffmpeg"""
        try:
            audio_path = output_dir / 'audio.aac'
            
            subprocess.run([
                'ffmpeg', '-i', video_path,
                '-q:a', '9',
                '-n',  # Don't overwrite
                str(audio_path)
            ], capture_output=True, timeout=60)
            
            if audio_path.exists():
                logger.info("✓ Audio extracted")
                return str(audio_path)
            
            return None
            
        except Exception as e:
            logger.warning(f"Audio extraction failed: {e}")
            return None
    
    def _rebuild_video(
        self,
        frame_paths: list,
        output_path: str,
        fps: float,
        width: int,
        height: int,
        audio_path: Optional[str] = None,
    ) -> bool:
        """Rebuild video from frames"""
        try:
            temp_video = Path(output_path).parent / f'temp_{os.urandom(8).hex()}.mp4'
            
            # Create video from frames without audio first
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(temp_video), fourcc, fps, (width, height))
            
            for frame_path in frame_paths:
                frame = cv2.imread(frame_path)
                if frame is not None:
                    # Ensure correct size
                    if frame.shape[:2] != (height, width):
                        frame = cv2.resize(frame, (width, height))
                    out.write(frame)
            
            out.release()
            
            # Add audio if available
            if audio_path and os.path.exists(audio_path):
                try:
                    subprocess.run([
                        'ffmpeg', '-i', str(temp_video),
                        '-i', audio_path,
                        '-c:v', 'copy',
                        '-c:a', 'aac',
                        '-map', '0:v:0',
                        '-map', '1:a:0',
                        '-y',
                        output_path
                    ], capture_output=True, timeout=300)
                except:
                    # Fallback: use video without audio
                    import shutil
                    shutil.move(str(temp_video), output_path)
            else:
                import shutil
                shutil.move(str(temp_video), output_path)
            
            logger.info(f"✓ Video rebuilt: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Video rebuild failed: {e}")
            return False
    
    def _cleanup(self, frames_dir: Path, audio_path: Optional[str] = None) -> None:
        """Cleanup temporary files"""
        try:
            import shutil
            if frames_dir.exists():
                shutil.rmtree(frames_dir)
            
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
            
            logger.info("✓ Temporary files cleaned up")
            
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")


# Convenience function
def create_video_pipeline(mode: ProcessingMode = ProcessingMode.STUDIO) -> HybridVideoFaceSwapPipeline:
    """Create a video processing pipeline instance"""
    return HybridVideoFaceSwapPipeline(mode)
