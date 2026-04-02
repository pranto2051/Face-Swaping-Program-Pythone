from flask import Flask, request, jsonify, send_file
from flask_socketio import SocketIO
from flask_cors import CORS
from api.ws.bridge import ProgressBridge
from core.database import db, Job
from services.gpu_monitor import GPUMonitor
from services.storage import StorageService
from processors.face_detector import MultiFaceDetector
from processors.face_swapper import FaceSwapper
from processors.expression_match import ExpressionMatcher
from processors.face_enhancer import FaceEnhancer
from pipelines.image_pipeline import ImagePipeline
from pipelines.video_pipeline import CinematicVideoPipeline
import torch
import cv2
import psutil
import os
import uuid
import json
import threading
import mimetypes
import zipfile
import gc
from werkzeug.utils import safe_join

class JobManager:
    def __init__(self):
        self._jobs = {}  # job_id -> {"thread": Thread, "stop_event": Event, "status": str}
        self._lock = threading.Lock()

    def create_job(self, job_id):
        with self._lock:
            self._jobs[job_id] = {
                "stop_event": threading.Event(),
                "status": "running"
            }
            return self._jobs[job_id]["stop_event"]

    def set_thread(self, job_id, thread):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["thread"] = thread

    def cancel_job(self, job_id):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["stop_event"].set()
                self._jobs[job_id]["status"] = "canceled"
                return True
        return False

    def get_status(self, job_id):
        with self._lock:
            return self._jobs.get(job_id, {}).get("status", "not_found")

    def cleanup(self, job_id):
        with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]

def create_app():
    # Use absolute path for static folder to ensure consistency regardless of working directory
    static_dir = os.path.abspath('static')
    os.makedirs(static_dir, exist_ok=True)
    
    app = Flask(__name__, static_folder=static_dir, static_url_path='/static')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///deepfake_v3.db')
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
    app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024 # 2GB
    
    CORS(app, resources={r"/*": {"origins": "*"}})
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
    storage = StorageService(static_folder=static_dir)
    detector = MultiFaceDetector()
    swapper = FaceSwapper()
    matcher = ExpressionMatcher()
    
    # Initialize Face Enhancer with GPU support if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        enhancer = FaceEnhancer(quality_mode='high_quality', device=device)
        print(f"✓ FaceEnhancer initialized with device: {device}")
    except Exception as e:
        print(f"Warning: FaceEnhancer initialization failed: {e}")
        print("Face swap will work but without advanced enhancement features")
        enhancer = None
    
    image_pipeline = ImagePipeline(detector, matcher, swapper, enhancer)
    video_pipeline = CinematicVideoPipeline(detector, matcher, swapper, enhancer)

    # Initialize DB
    db.init_app(app)
    
    # Start Progress Bridge
    bridge = ProgressBridge(socketio)
    bridge.start()
    
    # Initialize Job Manager
    job_manager = JobManager()

    def _parse_bool(raw_value, default=False):
        if raw_value is None:
            return default
        return str(raw_value).strip().lower() in ('1', 'true', 'yes', 'on')

    def _resolve_swap_behavior():
        """
        Resolve adaptive vs raw_copy behavior from incoming form fields.
        Supports both legacy expression_match and new use_target_expression/swap_mode.
        """
        explicit_swap_mode = str(request.form.get('swap_mode', '')).strip().lower()
        mode_field = str(request.form.get('mode', '')).strip().lower()

        if explicit_swap_mode not in ('adaptive', 'raw_copy') and mode_field in ('adaptive', 'raw_copy'):
            explicit_swap_mode = mode_field

        expr_field = request.form.get('use_target_expression')
        if expr_field is None:
            expr_field = request.form.get('expression_match')

        if expr_field is not None:
            use_target_expression = _parse_bool(expr_field, True)
        elif explicit_swap_mode in ('adaptive', 'raw_copy'):
            use_target_expression = explicit_swap_mode == 'adaptive'
        else:
            use_target_expression = True

        if explicit_swap_mode not in ('adaptive', 'raw_copy'):
            explicit_swap_mode = 'adaptive' if use_target_expression else 'raw_copy'

        # Raw copy always disables target-expression adaptation.
        if explicit_swap_mode == 'raw_copy':
            use_target_expression = False

        return explicit_swap_mode, use_target_expression

    def _get_mask_feather(default=10):
        try:
            value = int(request.form.get('mask_feather', default))
        except (TypeError, ValueError):
            value = default
        return max(0, min(50, value))
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        return jsonify({
            "status": "healthy",
            "gpu": "available",
            "cuda": torch.cuda.is_available() if 'torch' in globals() else False,
            "models_loaded": True
        }), 200

    @app.route('/api/jobs', methods=['GET'])
    def get_jobs():
        jobs = Job.query.all()
        return jsonify([{"id": j.id, "status": j.status} for j in jobs])
    
    @app.route('/api/detect/faces', methods=['POST'])
    def detect_faces():
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        
        file = request.files['image']
        path = storage.save_upload(file)
        
        import cv2
        img = cv2.imread(path)
        if img is None:
            return jsonify({"error": "Failed to decode image"}), 400

        faces = detector.detect(img)
        faces_count = len(faces)
        auto_select = faces_count == 1
        auto_selected_face_id = 0 if auto_select else None
        
        return jsonify({
            "faces_count": faces_count,
            "auto_select": auto_select,
            "face_count": faces_count,
            "requires_selection": faces_count > 1,
            "auto_selected_face_id": auto_selected_face_id,
            "suggested_face_id": 0 if faces_count > 1 else auto_selected_face_id,
            "image_width": int(img.shape[1]),
            "image_height": int(img.shape[0]),
            "faces": [
                {
                    "id": f.id,
                    "bbox": f.bbox,
                    "confidence": f.confidence,
                    "landmarks": f.landmarks.tolist() if hasattr(f.landmarks, 'tolist') else f.landmarks,
                    "embedding": f.embedding.tolist() if hasattr(f.embedding, 'tolist') else f.embedding
                } for f in faces
            ]
        })

    @app.route('/api/swap/image', methods=['POST'])
    def swap_image():
        source = request.files.get('source')
        target = request.files.get('target')
        mode = request.form.get('mode', 'fast')
        swap_mode, use_target_expression = _resolve_swap_behavior()
        mask_feather = _get_mask_feather()
        
        if not source or not target:
            return jsonify({"error": "Missing source or target"}), 400
            
        source_path = storage.save_upload(source)
        target_path = storage.save_upload(target)
        raw_face_assignments = request.form.get('face_assignments')
        raw_selected_face_id = request.form.get('selected_face_id')

        face_assignments = None
        if raw_face_assignments:
            try:
                face_assignments = json.loads(raw_face_assignments)
            except json.JSONDecodeError:
                return jsonify({"error": "Invalid face_assignments JSON"}), 400

        selected_face_id = None
        if raw_selected_face_id is not None and raw_selected_face_id != "":
            try:
                selected_face_id = int(raw_selected_face_id)
            except ValueError:
                return jsonify({"error": "selected_face_id must be an integer"}), 400

        img_target_preview = cv2.imread(target_path)
        if img_target_preview is None:
            return jsonify({"error": "Failed to decode target image"}), 400

        detected_target_faces = detector.detect(img_target_preview)
        target_face_count = len(detected_target_faces)
        if target_face_count == 0:
            return jsonify({"error": "No faces detected in target image"}), 400

        if selected_face_id is None and target_face_count == 1:
            selected_face_id = 0

        if selected_face_id is None and target_face_count > 1 and not face_assignments:
            return jsonify({"error": "Please select a face to continue."}), 400

        resolved_assignments = []
        if selected_face_id is not None:
            if selected_face_id < 0 or selected_face_id >= target_face_count:
                return jsonify({"error": "selected_face_id is out of range"}), 400
            resolved_assignments = [{"target_face_id": selected_face_id, "source_face_id": 0}]
        elif face_assignments:
            if not isinstance(face_assignments, list) or len(face_assignments) == 0:
                return jsonify({"error": "face_assignments must be a non-empty list"}), 400

            first_assignment = face_assignments[0]
            target_id = first_assignment.get('target_face_id', first_assignment.get('targetFaceId'))
            source_id = first_assignment.get('source_face_id', first_assignment.get('sourceFaceId', 0))
            if target_id is None:
                return jsonify({"error": "face assignment must include target face id"}), 400

            try:
                target_id = int(target_id)
                source_id = int(source_id)
            except ValueError:
                return jsonify({"error": "face assignment ids must be integers"}), 400

            if target_id < 0 or target_id >= target_face_count:
                return jsonify({"error": "target face id is out of range"}), 400

            resolved_assignments = [{"target_face_id": target_id, "source_face_id": source_id}]

        job_id = str(uuid.uuid4())
        stop_event = job_manager.create_job(job_id)

        def run_image_swap_task():
            try:
                bridge.emit_progress(job_id, "detecting", 10)
                img_source = cv2.imread(source_path)
                img_target = cv2.imread(target_path)
                
                if stop_event.is_set():
                    bridge.emit_canceled(job_id)
                    return

                bridge.emit_progress(job_id, "swapping", 40)
                result = image_pipeline.process_swap(
                    img_source, img_target, 
                    resolved_assignments,
                    {
                        'mode': mode,
                        'swap_mode': swap_mode,
                        'use_target_expression': use_target_expression,
                        'mask_feather': mask_feather,
                    }
                )
                
                if stop_event.is_set():
                    bridge.emit_canceled(job_id)
                    return

                bridge.emit_progress(job_id, "encoding", 90)
                out_filename = f"result_{job_id}.jpg"
                out_path = storage.get_static_path(out_filename)
                cv2.imwrite(out_path, result)
                
                bridge.emit_complete(job_id, storage.get_url(out_filename))
            except Exception as e:
                bridge.emit_error(job_id, str(e))
            finally:
                job_manager.cleanup(job_id)

        thread = threading.Thread(target=run_image_swap_task)
        job_manager.set_thread(job_id, thread)
        thread.start()

        return jsonify({"success": True, "job_id": job_id}), 202

    @app.route('/api/swap/batch', methods=['POST'])
    def swap_batch():
        source = request.files.get('source')
        targets = request.files.getlist('targets')
        mode = request.form.get('mode', 'fast')
        swap_mode, use_target_expression = _resolve_swap_behavior()
        mask_feather = _get_mask_feather()
        
        if not source or not targets:
            return jsonify({"error": "Missing source or targets"}), 400

        source_path = storage.save_upload(source)
        target_paths = [storage.save_upload(t) for t in targets]

        job_id = str(uuid.uuid4())
        stop_event = job_manager.create_job(job_id)

        def run_batch_swap_task():
            try:
                img_source = cv2.imread(source_path)
                results = []
                total = len(target_paths)
                
                for i, t_path in enumerate(target_paths):
                    if stop_event.is_set():
                        bridge.emit_canceled(job_id)
                        return
                        
                    bridge.emit_progress(job_id, "swapping", int((i/total)*100), current=i+1, total=total)
                    img_target = cv2.imread(t_path)
                    res = image_pipeline.process_swap(
                        img_source, img_target, 
                        [{"target_face_id": 0, "source_face_id": 0}],
                        {
                            'mode': mode,
                            'swap_mode': swap_mode,
                            'use_target_expression': use_target_expression,
                            'mask_feather': mask_feather,
                        }
                    )
                    
                    out_name = f"batch_{job_id}_{i}.jpg"
                    out_path = storage.get_static_path(out_name)
                    cv2.imwrite(out_path, res)
                    results.append({"filename": os.path.basename(t_path), "output_url": storage.get_url(out_name)})

                bridge.emit_complete(job_id, results)
            except Exception as e:
                bridge.emit_error(job_id, str(e))
            finally:
                job_manager.cleanup(job_id)

        thread = threading.Thread(target=run_batch_swap_task)
        job_manager.set_thread(job_id, thread)
        thread.start()

        return jsonify({"success": True, "job_id": job_id}), 202

    @app.route('/api/swap/pairs', methods=['POST'])
    def swap_pairs():
        """
        Multi-pair image swap endpoint.
        Expects multipart form-data:
          - pair_count: integer
          - source_<idx>: source image file
          - target_<idx>: target image file
          - selected_face_id_<idx>: optional selected face id for target
          - mode: fast|studio|cinematic
        """
        mode = request.form.get('mode', 'fast')
        swap_mode, use_target_expression = _resolve_swap_behavior()
        mask_feather = _get_mask_feather()

        try:
            pair_count = int(request.form.get('pair_count', '0'))
        except ValueError:
            return jsonify({"error": "pair_count must be an integer"}), 400

        if pair_count <= 0:
            return jsonify({"error": "At least one pair is required"}), 400

        pairs = []
        for idx in range(pair_count):
            source_file = request.files.get(f'source_{idx}')
            target_file = request.files.get(f'target_{idx}')

            if not source_file or not target_file:
                return jsonify({"error": f"Missing source/target for pair {idx + 1}"}), 400

            selected_face_raw = request.form.get(f'selected_face_id_{idx}')
            selected_face_id = None
            if selected_face_raw is not None and selected_face_raw != "":
                try:
                    selected_face_id = int(selected_face_raw)
                except ValueError:
                    return jsonify({"error": f"selected_face_id_{idx} must be an integer"}), 400

            pairs.append({
                "pair_id": idx + 1,
                "source_path": storage.save_upload(source_file),
                "target_path": storage.save_upload(target_file),
                "selected_face_id": selected_face_id,
                "status": "waiting",
                "progress": 0,
                "message": "Waiting",
                "result_url": None,
                "filename": None,
            })

        job_id = str(uuid.uuid4())
        stop_event = job_manager.create_job(job_id)

        def _clear_torch_cache():
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
            except Exception:
                pass
            gc.collect()

        def _emit_pairs_progress(stage, percent):
            bridge.emit_progress(
                job_id,
                stage,
                percent,
                current=sum(1 for p in pairs if p["status"] in ["completed", "failed"]),
                total=len(pairs)
            )
            socketio.emit('progress', {
                "job_id": job_id,
                "stage": stage,
                "percent": percent,
                "current_item": sum(1 for p in pairs if p["status"] in ["completed", "failed"]),
                "total_items": len(pairs),
                "pairs": pairs,
            })

        def run_pairs_swap_task():
            try:
                _emit_pairs_progress("detecting", 2)
                results = []
                total_pairs = len(pairs)

                for index, pair in enumerate(pairs):
                    if stop_event.is_set():
                        bridge.emit_canceled(job_id)
                        return

                    pair["status"] = "processing"
                    pair["progress"] = 10
                    pair["message"] = f"Processing Pair {index + 1} of {total_pairs}"
                    overall_before = int((index / total_pairs) * 100)
                    _emit_pairs_progress("detecting", max(5, overall_before))

                    try:
                        img_source = cv2.imread(pair["source_path"])
                        img_target = cv2.imread(pair["target_path"])

                        if img_source is None:
                            raise ValueError("Failed to decode source image")
                        if img_target is None:
                            raise ValueError("Failed to decode target image")

                        target_faces = detector.detect(img_target)
                        face_count = len(target_faces)

                        if face_count == 0:
                            raise ValueError("No face detected in target image.")

                        resolved_face_id = pair["selected_face_id"]
                        if face_count == 1 and resolved_face_id is None:
                            resolved_face_id = 0

                        if face_count > 1 and resolved_face_id is None:
                            raise ValueError("Multiple faces detected. Select one before processing.")

                        if resolved_face_id is None or resolved_face_id < 0 or resolved_face_id >= face_count:
                            raise ValueError("Selected face is out of range for target image.")

                        pair["progress"] = 45
                        pair["message"] = f"Swapping face for Pair {index + 1}"
                        overall_mid = int(((index + 0.5) / total_pairs) * 100)
                        _emit_pairs_progress("swapping", max(10, overall_mid))

                        result = image_pipeline.process_swap(
                            img_source,
                            img_target,
                            [{"target_face_id": int(resolved_face_id), "source_face_id": 0}],
                            {
                                'mode': mode,
                                'swap_mode': swap_mode,
                                'use_target_expression': use_target_expression,
                                'mask_feather': mask_feather,
                            }
                        )

                        out_filename = f"pair_{job_id}_{pair['pair_id']}.jpg"
                        out_path = storage.get_static_path(out_filename)
                        cv2.imwrite(out_path, result)

                        pair["status"] = "completed"
                        pair["progress"] = 100
                        pair["message"] = "Completed"
                        pair["result_url"] = storage.get_url(out_filename)
                        pair["filename"] = out_filename

                        results.append({
                            "pair_id": pair["pair_id"],
                            "filename": out_filename,
                            "output_url": pair["result_url"],
                            "status": "done"
                        })
                    except Exception as pair_error:
                        pair["status"] = "failed"
                        pair["progress"] = 100
                        pair["message"] = str(pair_error)
                        results.append({
                            "pair_id": pair["pair_id"],
                            "filename": f"pair_{pair['pair_id']}.jpg",
                            "status": "error",
                            "message": str(pair_error)
                        })
                    finally:
                        _clear_torch_cache()

                    overall_after = int(((index + 1) / total_pairs) * 100)
                    _emit_pairs_progress("swapping", overall_after)

                zip_filename = None
                successful = [r for r in results if r.get("status") == "done" and r.get("output_url")]
                if successful:
                    zip_filename = f"batch_{job_id}.zip"
                    zip_path = storage.get_static_path(zip_filename)

                    # ZIP_STORED keeps image bytes untouched (no recompression loss).
                    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_STORED) as zf:
                        for item in successful:
                            safe_name = os.path.basename(item["filename"]) or f"pair_{item['pair_id']}.jpg"
                            src_path = storage.get_static_path(safe_name)
                            if os.path.isfile(src_path):
                                zf.write(src_path, arcname=safe_name)

                _emit_pairs_progress("done", 100)
                socketio.emit('complete', {
                    "job_id": job_id,
                    "output_url": results,
                    "zip_url": storage.get_url(zip_filename) if zip_filename else None,
                })
            except Exception as e:
                bridge.emit_error(job_id, str(e))
            finally:
                job_manager.cleanup(job_id)

        thread = threading.Thread(target=run_pairs_swap_task)
        job_manager.set_thread(job_id, thread)
        thread.start()

        return jsonify({"success": True, "job_id": job_id}), 202

    @app.route('/api/swap/video', methods=['POST'])
    def swap_video():
        source = request.files.get('source')
        target_video = request.files.get('target_video')
        mode = request.form.get('mode', 'fast')
        swap_mode, use_target_expression = _resolve_swap_behavior()
        mask_feather = _get_mask_feather()
        
        if not source or not target_video:
            return jsonify({"error": "Missing source or video"}), 400
            
        source_path = storage.save_upload(source)
        target_v_path = storage.save_upload(target_video)
        
        source_img = cv2.imread(source_path)
        if source_img is None:
            return jsonify({"success": False, "error": "Failed to load source image"}), 400

        face_assignments = request.form.get('face_assignments')
        if face_assignments:
            face_assignments = json.loads(face_assignments)
            # Convert from dict format to tuple format expected by pipeline
            face_assignments = [(fa["target_face_id"], fa["source_face_id"]) for fa in face_assignments]
        else:
            face_assignments = [(0, 0)]  # Default: swap first face in target to first face in source

        frame_skip = int(request.form.get('frame_skip', 1))
        keyframe_detection = request.form.get('keyframe_detection', 'false').lower() == 'true'

        job_id = str(uuid.uuid4())
        stop_event = job_manager.create_job(job_id)
        
        def run_video_swap_task():
            try:
                bridge.emit_progress(job_id, "detecting", 5)
                # Video pipeline handles its own progress via bridge
                result_filename = video_pipeline.process(
                    target_v_path, 
                    source_img, 
                    face_assignments, 
                    {
                        "mode": mode, 
                        "swap_mode": swap_mode,
                        "use_target_expression": use_target_expression,
                        "mask_feather": mask_feather,
                        "frame_skip": frame_skip,
                        "keyframe_detection": keyframe_detection,
                        "job_id": job_id,
                        "bridge": bridge,
                        "stop_event": stop_event
                    }
                )
                
                if stop_event.is_set():
                    bridge.emit_canceled(job_id)
                    return

                if result_filename:
                    bridge.emit_complete(job_id, storage.get_url(result_filename))
                else:
                    bridge.emit_error(job_id, "Video processing failed or was canceled")
            except Exception as e:
                import traceback
                traceback.print_exc()
                bridge.emit_error(job_id, str(e))
            finally:
                job_manager.cleanup(job_id)

        thread = threading.Thread(target=run_video_swap_task)
        job_manager.set_thread(job_id, thread)
        thread.start()

        return jsonify({"success": True, "job_id": job_id}), 202


    @app.route('/api/jobs/<job_id>/cancel', methods=['POST'])
    def cancel_job(job_id):
        success = job_manager.cancel_job(job_id)
        if success:
            bridge.emit_canceled(job_id)
            return jsonify({"status": "canceled"}), 200
        return jsonify({"error": "Job not found or already finished"}), 404

    @app.route('/api/reset/state', methods=['POST'])
    def reset_state():
        """Cleanup temporary pair-batch artifacts without reloading the app/socket."""
        payload = request.get_json(silent=True) or {}
        job_id = payload.get("job_id")

        removed = []
        if job_id:
            candidates = [
                f"batch_{job_id}.zip",
            ]

            static_dir = os.path.abspath(storage.static_folder)
            for name in os.listdir(static_dir):
                if name.startswith(f"pair_{job_id}_") and name.endswith(".jpg"):
                    candidates.append(name)

            for name in set(candidates):
                path = storage.get_static_path(name)
                if os.path.isfile(path):
                    try:
                        os.remove(path)
                        removed.append(name)
                    except Exception:
                        pass

        return jsonify({"success": True, "removed": removed, "removed_count": len(removed)})

    @app.route('/api/gpu', methods=['GET'])
    def get_gpu():
        monitor = GPUMonitor()
        stats = monitor.get_stats()
        # Transform to exact format expected by GpuStats in types.ts
        v_used = stats.get("vram_used_mb", 0)
        v_total = stats.get("vram_total_mb", 1) # Avoid div by zero
        return jsonify({
            "gpu_name": stats.get("gpu_name", "Unknown GPU"),
            "vram_used_mb": v_used,
            "vram_total_mb": v_total,
            "vram_percent": round((v_used / v_total) * 100, 1),
            "gpu_util_percent": stats.get("gpu_util_percent", 0),
            "temperature_c": stats.get("temperature_c", 0),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A"
        })

    @app.route('/api/download/<path:filename>')
    def download_file(filename):
        # Restrict downloads to files inside static folder.
        base_dir = os.path.abspath(storage.static_folder)
        safe_path = safe_join(base_dir, filename)

        if not safe_path:
            return jsonify({"error": "Invalid file path"}), 400

        if not os.path.isfile(safe_path):
            return jsonify({"error": "File not found"}), 404

        download_name = os.path.basename(safe_path)
        guessed_mimetype, _ = mimetypes.guess_type(download_name)
        mimetype = guessed_mimetype or "application/octet-stream"

        response = send_file(
            safe_path,
            as_attachment=True,
            download_name=download_name,
            mimetype=mimetype,
            conditional=True,
            etag=True,
            max_age=0,
        )
        # Prevent MIME sniffing and stale cache when serving sensitive outputs.
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.route('/api/cpu', methods=['GET'])
    def get_cpu():
        import platform
        freq = psutil.cpu_freq()
        return jsonify({
            "cpu_name": platform.processor() or "Generic CPU",
            "cpu_percent": psutil.cpu_percent(),
            "core_count": psutil.cpu_count(logical=False),
            "thread_count": psutil.cpu_count(logical=True),
            "frequency_mhz": freq.current if freq else 0,
            "temperature_c": 0, # psutil sensors_temperatures() often empty on Mac/Win
            "per_core_percent": psutil.cpu_percent(percpu=True)
        })

    @app.route('/api/ram', methods=['GET'])
    def get_ram():
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return jsonify({
            "total_gb": round(mem.total / (1024**3), 2),
            "used_gb": round(mem.used / (1024**3), 2),
            "available_gb": round(mem.available / (1024**3), 2),
            "percent": mem.percent,
            "swap_total_gb": round(swap.total / (1024**3), 2),
            "swap_used_gb": round(swap.used / (1024**3), 2),
            "swap_percent": swap.percent
        })

    @app.route('/api/gpu/stats', methods=['GET'])
    def gpu_stats():
        monitor = GPUMonitor()
        return jsonify(monitor.get_stats())

    @app.after_request
    def add_cors_headers(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
        response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
        return response

    @app.errorhandler(Exception)
    def handle_exception(e):
        import traceback
        print(f"GLOBAL ERROR: {str(e)}")
        traceback.print_exc()
        response = jsonify({"success": False, "error": str(e)})
        response.status_code = 500
        return add_cors_headers(response)

    return app, socketio

if __name__ == '__main__':
    app, socketio = create_app()
    socketio.run(app, debug=True, host='0.0.0.0', port=5001)
