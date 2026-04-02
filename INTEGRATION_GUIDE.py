"""
Integration Guide: MPS Face Enhancement System into Existing Flask App

This file explains how to integrate the new MPS-optimized face enhancement
modules into your existing Face Swap Flask application.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Update app.py - Device Detection & Initialization
# ═══════════════════════════════════════════════════════════════════════════════

# CURRENT CODE (OLD):
# ──────────────────
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# enhancer = FaceEnhancer(quality_mode='high_quality', device=device)

# NEW CODE (MPS-OPTIMIZED):
# ─────────────────────────
"""
from core.device_manager import get_device_manager
from processors.face_enhancer_mps import MPS_FaceEnhancer
from processors.face_swapper_mps import MPS_FaceSwapper

# Initialize device manager once
device_mgr = get_device_manager()

# Initialize from environment or defaults
quality_mode = os.getenv('QUALITY_MODE', 'high_quality')
enhancer = MPS_FaceEnhancer(quality_mode=quality_mode)

# Initialize face swapper
swapper = MPS_FaceSwapper()

print(f"✓ MPS_FaceEnhancer ready - Mode: {quality_mode}")
print(f"✓ MPS_FaceSwapper ready - Device: {device_mgr.device_name}")
"""

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Update Image Pipeline - Use MPS Enhancement
# ═══════════════════════════════════════════════════════════════════════════════

# FILE: pipelines/image_pipeline.py
# ──────────────────────────────────────────────────────────────────────────────

"""
from processors.face_enhancer_mps import MPS_FaceEnhancer
from processors.face_swapper_mps import MPS_FaceSwapper

class ImagePipeline:
    def __init__(self, quality_mode='high_quality'):
        self.quality_mode = quality_mode
        self.enhancer = MPS_FaceEnhancer(quality_mode=quality_mode)
        self.swapper = MPS_FaceSwapper()
    
    def process_image(self, source_path, target_path, quality_mode=None):
        '''
        Complete image processing pipeline.
        
        Args:
            source_path: Path to source image (face to swap from)
            target_path: Path to target image (face to swap to)
            quality_mode: Override quality mode if specified
        
        Returns:
            Enhanced swapped image
        '''
        mode = quality_mode or self.quality_mode
        
        # Load images
        source_image = cv2.imread(source_path)
        target_image = cv2.imread(target_path)
        
        # Step 1: Detect faces
        source_faces = self.swapper.detect_faces(source_image)
        target_faces = self.swapper.detect_faces(target_image)
        
        if not source_faces or not target_faces:
            raise ValueError("Could not detect faces in one or both images")
        
        # Step 2: Swap faces at original resolution
        swapped_image, success = self.swapper.swap_face(
            source_image, target_image,
            source_faces, target_faces
        )
        
        if not success:
            raise RuntimeError("Face swap failed")
        
        # Step 3: Prepare faces data for enhancement
        faces_data = []
        for i, face in enumerate(target_faces):
            # Extract region
            face_region, mask, bbox = self.swapper.extract_face_region(
                swapped_image, face
            )
            
            if face_region is not None:
                # Get original region for color reference
                x, y, w, h = bbox
                original_region = target_image[y:y+h, x:x+w].copy()
                
                faces_data.append({
                    "face": face_region,
                    "mask": mask,
                    "bbox": bbox,
                    "original_region": original_region,
                })
        
        # Step 4: Enhance all faces
        if faces_data and mode != "fast":
            enhanced_image = self.enhancer.enhance_image(swapped_image, faces_data)
        else:
            enhanced_image = swapped_image
        
        # Clear cache
        self.enhancer.device_manager.clear_cache()
        
        return enhanced_image
    
    def cleanup(self):
        '''Release model resources.'''
        self.enhancer.cleanup()
"""

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Flask API Endpoints - Add Quality Mode Selection
# ═══════════════════════════════════════════════════════════════════════════════

# In app.py, add endpoint for quality mode configuration:

"""
@app.route('/api/config/quality-mode', methods=['GET', 'POST'])
def quality_mode_config():
    '''Get or set quality mode for face enhancement.'''
    from core.quality_config import QualityModeConfig
    
    if request.method == 'POST':
        data = request.get_json()
        mode = data.get('mode', 'high_quality')
        
        if not QualityModeConfig.validate_mode(mode):
            return jsonify({"error": "Invalid mode"}), 400
        
        # Update enhancement mode
        # This typically involves recreating the enhancer
        # Or having a mode setting that's read each time
        
        return jsonify({
            "current_mode": mode,
            "config": QualityModeConfig.get_mode_config(mode)
        })
    
    else:  # GET request
        current = os.getenv('QUALITY_MODE', 'high_quality')
        modes = QualityModeConfig.get_all_modes()
        
        return jsonify({
            "current_mode": current,
            "available_modes": {k: v for k, v in modes.items()},
            "recommendations": {
                "m1_pro_16gb": "high_quality",
                "m2_pro_16gb": "high_quality",
                "m2_max_32gb": "ultra",
            }
        })

@app.route('/api/diagnostics', methods=['GET'])
def get_diagnostics():
    '''Return system diagnostics and optimization recommendations.'''
    from core.device_manager import get_device_manager
    
    mgr = get_device_manager()
    budget = mgr.get_memory_budget()
    
    return jsonify({
        "device": mgr.device_name,
        "device_type": mgr.device.type,
        "pytorch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "memory": {
            "total_gb": mgr.memory_info['total'],
            "available_gb": mgr.memory_info['available'],
            "used_percent": mgr.memory_info['percent'],
        },
        "recommendations": {
            "safe_allocation_gb": budget['safe_model_gb'],
            "max_image_size": budget['max_image_size'],
            "batch_size": budget['batch_size'],
        }
    })
"""

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: Video Pipeline - Same Approach
# ═══════════════════════════════════════════════════════════════════════════════

# FILE: pipelines/video_pipeline.py
# Same pattern - pass quality_mode to enhancer

"""
from processors.face_enhancer_mps import MPS_FaceEnhancer

class CinematicVideoPipeline:
    def __init__(self, quality_mode='high_quality'):
        self.quality_mode = quality_mode
        self.enhancer = MPS_FaceEnhancer(quality_mode=quality_mode)
    
    def process_frame(self, frame, faces_data):
        '''Apply enhancement to a single video frame.'''
        if not faces_data:
            return frame
        
        # Enhance swapped faces
        enhanced = self.enhancer.enhance_image(frame, faces_data)
        
        # Clear cache after every frame (or every N frames for batch processing)
        self.enhancer.device_manager.clear_cache()
        
        return enhanced
"""

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: Frontend Integration - Quality Mode Selector
# ═══════════════════════════════════════════════════════════════════════════════

# Add to upload/processing form:

"""
// React component for quality mode selection

import React, { useState } from 'react';

export const QualityModeSelector = ({ onModeChange }) => {
  const [mode, setMode] = useState('high_quality');
  
  const modes = {
    'fast': {
      name: 'Fast',
      description: 'CPU only, <10 seconds',
      icon: '⚡'
    },
    'high_quality': {
      name: 'High Quality',
      description: 'MPS enhanced, 40-60 seconds (recommended)',
      icon: '✓'
    },
    'ultra': {
      name: 'Ultra',
      description: 'Maximum quality, 60-90 seconds',
      icon: '✨'
    }
  };
  
  return (
    <div className="quality-mode-selector">
      <label>Enhancement Quality:</label>
      {Object.entries(modes).map(([key, value]) => (
        <label key={key}>
          <input
            type="radio"
            value={key}
            checked={mode === key}
            onChange={(e) => {
              setMode(e.target.value);
              onModeChange(e.target.value);
            }}
          />
          <span>{value.icon} {value.name}</span>
          <small>{value.description}</small>
        </label>
      ))}
    </div>
  );
};
"""

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: Environment Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# Add to .env file:
"""
# MPS Face Enhancement Configuration
QUALITY_MODE=high_quality
DEVICE_TYPE=mps
USE_FLOAT16=0
BATCH_SIZE=1
MAX_MEMORY_GB=14
CLEAR_CACHE_INTERVAL=1

# Model Paths (optional - defaults to ~/.cache)
CODEFORMER_MODEL_PATH=~/.cache/codeformer/codeformer.pth
REALESRGAN_MODEL_PATH=~/.cache/realesrgan/RealESRGAN_x2_compact.pth
"""

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7: Testing the Integration
# ═══════────────────────────────────────────────────────────────────────────────

"""
# Test script: backend/test_integration.py

import cv2
from processors.face_enhancer_mps import MPS_FaceEnhancer
from processors.face_swapper_mps import MPS_FaceSwapper
from pipelines.image_pipeline import ImagePipeline

# Test 1: Individual components
print("Testing individual components...")

swapper = MPS_FaceSwapper()
print("✓ MPS_FaceSwapper initialized")

enhancer = MPS_FaceEnhancer(quality_mode="high_quality")
print("✓ MPS_FaceEnhancer initialized")

# Test 2: Pipeline
print("\nTesting pipeline...")

pipeline = ImagePipeline(quality_mode="high_quality")
print("✓ ImagePipeline initialized")

# Test 3: Process test images
print("\nProcessing test images...")
try:
    source = cv2.imread("test_source.jpg")
    target = cv2.imread("test_target.jpg")
    
    result = pipeline.process_image(
        "test_source.jpg",
        "test_target.jpg",
        quality_mode="high_quality"
    )
    
    cv2.imwrite("test_output.png", result)
    print("✓ End-to-end processing successful")
    print(f"✓ Output saved to test_output.png")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    pipeline.cleanup()
    print("\n✓ Cleanup complete")
"""

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8: Production Deployment Checklist
# ═══════────────────────────────────────────────────────────────────────────────

"""
✅ DEPLOYMENT CHECKLIST:

Code Integration:
  [ ] Updated app.py with MPS device detection
  [ ] Updated image_pipeline.py with MPS_FaceEnhancer
  [ ] Added quality mode API endpoints
  [ ] Added diagnostics endpoint
  [ ] Frontend quality mode selector implemented

Models:
  [ ] CodeFormer model downloaded to ~/.cache/codeformer/
  [ ] Real-ESRGAN model downloaded to ~/.cache/realesrgan/
  [ ] Model paths verified and accessible

Configuration:
  [ ] .env file updated with quality mode settings
  [ ] QUALITY_MODE set to 'high_quality' (recommended)
  [ ] USE_FLOAT16 set to 0 (for stability)
  [ ] BATCH_SIZE set to 1

Testing:
  [ ] Test diagnostics endpoint responds correctly
  [ ] Test with FAST mode (quick test)
  [ ] Test with HIGH_QUALITY mode (full test)
  [ ] Verify memory usage stays under 8GB
  [ ] Check processing time (should be 40-60s)
  [ ] Verify output quality (sharp, natural lighting)

Performance:
  [ ] Run diagnostics.py to verify MPS support
  [ ] Monitor thermal - should not exceed 85°C
  [ ] Check memory during batch processing
  [ ] Verify seamless blending with reference image

Documentation:
  [ ] Share MPS_SETUP_GUIDE.md with team
  [ ] Share BEST_PRACTICES_MPS.md with team
  [ ] Document quality mode selection logic
  [ ] Create troubleshooting guide for users
"""

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 9: Migration from Old Enhancer (if applicable)
# ═══════────────────────────────────────────────────────────────────────────────

"""
# Old import:
from processors.face_enhancer import FaceEnhancer

# New import:
from processors.face_enhancer_mps import MPS_FaceEnhancer

# Old usage:
enhancer = FaceEnhancer(quality_mode='high_quality', device='cuda')
result = enhancer.enhance_face_region(...)

# New usage:
enhancer = MPS_FaceEnhancer(quality_mode='high_quality')
result = enhancer.enhance_image(image, faces_data)

# Key differences:
# 1. Old module: per-face enhancement with manual management
# 2. New module: batch face enhancement with automatic resource management
# 3. New module: integrated color correction & blending
# 4. New module: automatic MPS device detection (no manual device selection)
"""

print(__doc__)
