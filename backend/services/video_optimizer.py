import cv2
import numpy as np

class VideoOptimizer:
    """
    Algorithm:

    1. SCENE DETECTION: Mark scene boundary when correlation < threshold (0.3).
    2. KEYFRAME SELECTION: Process every Nth frame + scene boundaries.
    3. INTERPOLATION: Use optical flow for intermediate frames.
    """

    def __init__(self, job_id=None, socketio=None):
        self.job_id = job_id
        self.socketio = socketio

    def detect_scenes(self, video_path) -> list:
        cap = cv2.VideoCapture(video_path)
        scenes = [0]
        prev_hist = None
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            curr_hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            curr_hist = cv2.normalize(curr_hist, curr_hist).flatten()
            
            if prev_hist is not None:
                correlation = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
                if correlation < 0.3:
                    scenes.append(frame_idx)
            
            prev_hist = curr_hist
            frame_idx += 1
            
        cap.release()
        return scenes

    def process_frame_progress(self, frame_idx, total):
        if self.socketio and self.job_id:
            self.socketio.emit('progress', {
                'job_id': self.job_id,
                'stage': 'video_processing',
                'current': frame_idx,
                'total': total,
                'percent': round(frame_idx / total * 100, 1)
            })

    def interpolate_frames(self, prev_frame, next_frame, steps):
        """
        Generate intermediate frames using Farneback Optical Flow.
        """
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        interpolated = []
        for i in range(1, steps + 1):
            alpha = i / (steps + 1)
            # Warp frames based on flow
            # Simplified: actual implementation would use more complex warping
            warped = self._warp_frame(prev_frame, flow * alpha)
            interpolated.append(warped)
        return interpolated

    def _warp_frame(self, img, flow):
        h, w = img.shape[:2]
        flow = -flow
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.reshape(np.arange(h), (h, 1))
        return cv2.remap(img, flow, None, cv2.INTER_LINEAR)
