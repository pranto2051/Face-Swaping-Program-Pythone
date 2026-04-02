import json
import redis
import threading

class ProgressBridge:
    def __init__(self, socketio):
        self.socketio = socketio

    def start(self):
        # No separate thread needed for direct emit
        pass

    def emit_progress(self, job_id, stage, percent, current=None, total=None, eta=0):
        data = {
            "job_id": job_id,
            "stage": stage,
            "percent": percent,
            "current_item": current,
            "total_items": total,
            "eta_seconds": eta
        }
        self.socketio.emit('progress', data)
        # Also log to console for debugging
        print(f"JOB {job_id} PROGRESS: {stage} {percent}%")

    def emit_complete(self, job_id, output_url, quality_score=None):
        self.socketio.emit('complete', {
            "job_id": job_id,
            "output_url": output_url,
            "quality_score": quality_score
        })
        print(f"JOB {job_id} COMPLETE: {output_url}")

    def emit_error(self, job_id, message):
        self.socketio.emit('error', {
            "job_id": job_id,
            "message": message
        })
        print(f"JOB {job_id} ERROR: {message}")

    def emit_canceled(self, job_id):
        self.socketio.emit('progress', {
            "job_id": job_id,
            "stage": "idle",
            "percent": 0,
            "status": "cancelled"
        })
        print(f"JOB {job_id} CANCELLED")
