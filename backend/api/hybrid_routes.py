"""
Hybrid Face Swap API Routes
Endpoints for advanced face swapping with multiple modes and features
"""
import logging
import os
import uuid
import json
from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from pathlib import Path
from core.mode_controller import ModeController, ProcessingMode
from pipelines.hybrid_face_swap import HybridFaceSwapPipeline
from pipelines.hybrid_video_pipeline import HybridVideoFaceSwapPipeline
from services.gpu_monitor import GPUMonitor

logger = logging.getLogger(__name__)

# Create blueprint
hybrid_swap_bp = Blueprint('hybrid_swap', __name__, url_prefix='/api/hybrid')

# Temporary storage
TEMP_DIR = Path('./temp_hybrid_processing')
TEMP_DIR.mkdir(exist_ok=True)
UPLOAD_FOLDER = Path('./uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True)

# File validation
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4', 'mov', 'avi', 'mkv'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB


def allowed_file(filename: str) -> bool:
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@hybrid_swap_bp.route('/modes', methods=['GET'])
def get_available_modes():
    """Get information about all available processing modes"""
    try:
        modes_info = {}
        for mode in ProcessingMode:
            modes_info[mode.value] = ModeController.get_mode_summary(mode)
        
        return jsonify({
            'success': True,
            'modes': modes_info
        })
    except Exception as e:
        logger.error(f"Error getting modes: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@hybrid_swap_bp.route('/mode/<mode_name>', methods=['GET'])
def get_mode_info(mode_name: str):
    """Get detailed information about a specific mode"""
    try:
        mode = ModeController.get_mode_by_name(mode_name)
        if not mode:
            return jsonify({'success': False, 'error': 'Invalid mode'}), 400
        
        summary = ModeController.get_mode_summary(mode)
        
        return jsonify({
            'success': True,
            'mode': summary
        })
    except Exception as e:
        logger.error(f"Error getting mode info: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@hybrid_swap_bp.route('/swap/image', methods=['POST'])
def swap_image():
    """
    Swap faces in images with hybrid pipeline
    
    Request:
    - source_image: Image file with face to swap
    - target_image: Image file to receive face
    - mode: Processing mode (fast, studio, cinematic)
    - source_face_id: Optional source face ID
    - target_face_id: Optional target face ID
    
    Response:
    - swapped_image: Base64 encoded result
    - metadata: Processing information
    """
    try:
        # Validate request
        if 'source_image' not in request.files or 'target_image' not in request.files:
            return jsonify({'success': False, 'error': 'Missing image files'}), 400
        
        source_file = request.files['source_image']
        target_file = request.files['target_image']
        mode_name = request.form.get('mode', 'studio')
        source_face_id = request.form.get('source_face_id', type=int)
        target_face_id = request.form.get('target_face_id', type=int)
        
        if not allowed_file(source_file.filename) or not allowed_file(target_file.filename):
            return jsonify({'success': False, 'error': 'Invalid file format'}), 400
        
        # Get processing mode
        mode = ModeController.get_mode_by_name(mode_name)
        if not mode:
            return jsonify({'success': False, 'error': 'Invalid processing mode'}), 400
        
        # Load images
        source_data = np.frombuffer(source_file.read(), np.uint8)
        target_data = np.frombuffer(target_file.read(), np.uint8)
        
        source_image = cv2.imdecode(source_data, cv2.IMREAD_COLOR)
        target_image = cv2.imdecode(target_data, cv2.IMREAD_COLOR)
        
        if source_image is None or target_image is None:
            return jsonify({'success': False, 'error': 'Failed to decode images'}), 400
        
        # Process images
        logger.info(f"Processing image swap in {mode.value} mode")
        pipeline = HybridFaceSwapPipeline(mode)
        
        swapped_image, metadata = pipeline.process(
            source_image,
            target_image,
            source_face_id,
            target_face_id,
        )
        
        if not metadata.get('success'):
            return jsonify({
                'success': False,
                'error': metadata.get('reason', 'Processing failed')
            }), 400
        
        # Encode output
        success, encoded = cv2.imencode('.png', swapped_image)
        if not success:
            return jsonify({'success': False, 'error': 'Failed to encode output'}), 500
        
        # Save for download
        output_id = str(uuid.uuid4())
        output_path = TEMP_DIR / f'{output_id}.png'
        cv2.imwrite(str(output_path), swapped_image)
        
        return jsonify({
            'success': True,
            'output_id': output_id,
            'metadata': metadata,
            'download_url': f'/api/hybrid/download/{output_id}',
        })
        
    except Exception as e:
        logger.error(f"Image swap error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@hybrid_swap_bp.route('/swap/batch', methods=['POST'])
def swap_batch():
    """
    Batch process multiple images with same target
    
    Request:
    - source_images: Multiple image files
    - target_image: Target image file
    - mode: Processing mode
    
    Response:
    - batch_id: Unique ID for batch processing
    - output_ids: List of output image IDs
    """
    try:
        if 'source_images' not in request.files or 'target_image' not in request.files:
            return jsonify({'success': False, 'error': 'Missing images'}), 400
        
        source_files = request.files.getlist('source_images')
        target_file = request.files['target_image']
        mode_name = request.form.get('mode', 'studio')
        
        if not source_files:
            return jsonify({'success': False, 'error': 'No source images'}), 400
        
        # Get processing mode
        mode = ModeController.get_mode_by_name(mode_name)
        if not mode:
            return jsonify({'success': False, 'error': 'Invalid mode'}), 400
        
        # Load target image
        target_data = np.frombuffer(target_file.read(), np.uint8)
        target_image = cv2.imdecode(target_data, cv2.IMREAD_COLOR)
        
        if target_image is None:
            return jsonify({'success': False, 'error': 'Failed to load target'}), 400
        
        # Load source images
        source_images = []
        for source_file in source_files:
            if allowed_file(source_file.filename):
                source_data = np.frombuffer(source_file.read(), np.uint8)
                source_image = cv2.imdecode(source_data, cv2.IMREAD_COLOR)
                if source_image is not None:
                    source_images.append(source_image)
        
        if not source_images:
            return jsonify({'success': False, 'error': 'No valid source images'}), 400
        
        # Process batch
        logger.info(f"Batch processing {len(source_images)} images")
        pipeline = HybridFaceSwapPipeline(mode)
        
        swapped_images, batch_metadata = pipeline.process_batch(
            source_images,
            target_image,
        )
        
        # Save results
        batch_id = str(uuid.uuid4())
        output_ids = []
        
        for idx, swapped in enumerate(swapped_images):
            output_id = f"{batch_id}_{idx}"
            output_path = TEMP_DIR / f'{output_id}.png'
            cv2.imwrite(str(output_path), swapped)
            output_ids.append(output_id)
        
        return jsonify({
            'success': True,
            'batch_id': batch_id,
            'output_ids': output_ids,
            'metadata': batch_metadata,
        })
        
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@hybrid_swap_bp.route('/swap/video', methods=['POST'])
def swap_video():
    """
    Process video with face swapping
    
    Request:
    - video_file: Video file to process
    - source_image: Image with face to swap
    - mode: Processing mode
    
    Response:
    - job_id: Processing job ID
    - status_url: URL to check processing status
    """
    try:
        if 'video_file' not in request.files or 'source_image' not in request.files:
            return jsonify({'success': False, 'error': 'Missing files'}), 400
        
        video_file = request.files['video_file']
        source_file = request.files['source_image']
        mode_name = request.form.get('mode', 'studio')
        
        if not allowed_file(video_file.filename) or not allowed_file(source_file.filename):
            return jsonify({'success': False, 'error': 'Invalid file format'}), 400
        
        # Get processing mode
        mode = ModeController.get_mode_by_name(mode_name)
        if not mode:
            return jsonify({'success': False, 'error': 'Invalid mode'}), 400
        
        # Save uploaded files
        job_id = str(uuid.uuid4())
        video_path = UPLOAD_FOLDER / f'{job_id}_input.mp4'
        source_path = UPLOAD_FOLDER / f'{job_id}_source.png'
        
        video_file.save(str(video_path))
        source_file.save(str(source_path))
        
        # Note: Actual video processing should be done async
        # For now, return job info
        return jsonify({
            'success': True,
            'job_id': job_id,
            'status_url': f'/api/hybrid/video/status/{job_id}',
            'message': 'Video processing queued. Check status URL for updates.'
        })
        
    except Exception as e:
        logger.error(f"Video upload error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@hybrid_swap_bp.route('/detect-faces', methods=['POST'])
def detect_faces():
    """
    Detect faces in an image for selection
    
    Request:
    - image_file: Image to analyze
    
    Response:
    - faces: List of detected faces with bounding boxes
    - annotated_image: Image with face annotations
    """
    try:
        if 'image_file' not in request.files:
            return jsonify({'success': False, 'error': 'Missing image'}), 400
        
        image_file = request.files['image_file']
        
        if not allowed_file(image_file.filename):
            return jsonify({'success': False, 'error': 'Invalid file format'}), 400
        
        # Load image
        image_data = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'success': False, 'error': 'Failed to decode image'}), 400
        
        # Detect faces
        from services.face_detection_advanced import FaceDetector
        detector = FaceDetector()
        faces, annotated = detector.detect_faces(image)
        
        # Convert annotated image to base64
        success, encoded = cv2.imencode('.png', annotated)
        if success:
            import base64
            annotated_b64 = base64.b64encode(encoded).decode()
        else:
            annotated_b64 = None
        
        # Convert faces to JSON-serializable format
        faces_data = [face.to_dict() for face in faces]
        
        return jsonify({
            'success': True,
            'faces': faces_data,
            'face_count': len(faces),
            'annotated_image_b64': annotated_b64,
        })
        
    except Exception as e:
        logger.error(f"Face detection error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@hybrid_swap_bp.route('/download/<output_id>', methods=['GET'])
def download_result(output_id: str):
    """Download processed image"""
    try:
        output_path = TEMP_DIR / f'{output_id}.png'
        
        if not output_path.exists():
            return jsonify({'success': False, 'error': 'Output not found'}), 404
        
        return send_file(
            str(output_path),
            mimetype='image/png',
            as_attachment=True,
            download_name=f'swapped_{output_id}.png'
        )
        
    except Exception as e:
        logger.error(f"Download error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@hybrid_swap_bp.route('/gpu-info', methods=['GET'])
def get_gpu_info():
    """Get GPU and device information"""
    try:
        from core.device_config import get_device_config
        
        device_config = get_device_config()
        
        return jsonify({
            'success': True,
            'device_type': device_config.device_type,
            'processor_info': device_config.processor_info,
            'memory_limit_gb': device_config.memory_limit / (1024**3),
        })
        
    except Exception as e:
        logger.error(f"GPU info error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@hybrid_swap_bp.route('/health', methods=['GET'])
def health_check():
    """Check pipeline health"""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'service': 'hybrid-face-swap',
        'version': '1.0.0'
    })


# Export blueprint
def init_hybrid_routes(app):
    """Initialize hybrid routes in Flask app"""
    app.register_blueprint(hybrid_swap_bp)
    logger.info("✓ Hybrid face swap routes initialized")
