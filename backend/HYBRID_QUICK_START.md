# Quick Start Guide - Hybrid Face Swap Pipeline

## For Developers

### 1. Quick Setup (5 minutes)

```bash
cd backend

# Activate virtual environment
source .venv/bin/activate

# Install/update requirements
pip install -r requirements.txt

# Verify installation
python -c "from pipelines.hybrid_face_swap import HybridFaceSwapPipeline; print('✓ Ready to use')"
```

### 2. Test with CLI Script

Create `test_pipeline.py`:

```python
#!/usr/bin/env python3
"""Quick test of hybrid face swap pipeline"""

from pathlib import Path
import cv2
from core.mode_controller import ProcessingMode
from pipelines.hybrid_face_swap import HybridFaceSwapPipeline

# Load test images
source = cv2.imread('test_source.jpg')
target = cv2.imread('test_target.jpg')

if source is None or target is None:
    print("Error: Could not load test images")
    print("Place test_source.jpg and test_target.jpg in backend directory")
    exit(1)

# Test all modes
modes = [ProcessingMode.FAST, ProcessingMode.STUDIO, ProcessingMode.CINEMATIC]

for mode in modes:
    print(f"\n{'='*50}")
    print(f"Testing {mode.value.upper()} Mode")
    print(f"{'='*50}")
    
    pipeline = HybridFaceSwapPipeline(mode)
    
    result, metadata = pipeline.process(source, target)
    
    if metadata['success']:
        # Save result
        output_file = f'result_{mode.value}.jpg'
        cv2.imwrite(output_file, result)
        print(f"✓ Success! Saved to: {output_file}")
        print(f"  Time: {metadata['total_time']:.2f}s")
        print(f"  Stages: {list(metadata['stages'].keys())}")
    else:
        print(f"✗ Failed: {metadata.get('reason')}")

print(f"\n{'='*50}")
print("Testing Complete!")
print(f"{'='*50}\n")
```

Run it:
```bash
python test_pipeline.py
```

### 3. API Server Test

Start backend server:
```bash
python app.py
```

Test endpoints:
```bash
# Check modes
curl http://localhost:5001/api/hybrid/modes | python -m json.tool

# Check GPU info
curl http://localhost:5001/api/hybrid/gpu-info | python -m json.tool

# Test health
curl http://localhost:5001/api/hybrid/health
```

### 4. Frontend Integration

In your frontend code:

```typescript
// Get available modes
const response = await fetch('/api/hybrid/modes');
const modesData = await response.json();

// Build form for mode selection
const modes = Object.keys(modesData.modes);
console.log('Available modes:', modes); // ['fast', 'studio', 'cinematic']

// Swap faces
const formData = new FormData();
formData.append('source_image', sourceFile);
formData.append('target_image', targetFile);
formData.append('mode', 'studio');

const result = await fetch('/api/hybrid/swap/image', {
    method: 'POST',
    body: formData
});

const output = await result.json();
console.log('Output ID:', output.output_id);

// Download result
window.location.href = output.download_url;
```

## For End Users

### Using the Application

1. **Select Mode**
   - **Fast**: Quick preview (~1 sec)
   - **Studio**: Good quality (~3-5 sec)
   - **Cinematic**: Best quality (~10-20 sec)

2. **Upload Images**
   - Upload source image with source face
   - Upload target image to receive swap
   - System detects faces automatically

3. **Optional: Select Specific Faces**
   - Click "Detect Faces" to see all detected faces
   - Select which face from each image to use
   - Leave blank for auto-selection

4. **Process**
   - Click "Swap Faces"
   - Monitor progress bar
   - Download result when complete

### Tips for Best Results

**Image Quality**
- Use high-quality, well-lit images
- Ensure faces are clear and frontal
- Minimum resolution: 256x256
- Recommended: 512x512 or higher

**Face Requirements**
- Both images should contain clear faces
- Faces should be roughly similar size
- Good lighting helps overall quality
- High expression differences work in all modes

**Mode Selection**
- Use **Fast** for batch processing or previews
- Use **Studio** for everyday use (good balance)
- Use **Cinematic** for final, high-quality output

**Video Processing**
- Use lower resolution for faster processing
- Studio mode is recommended for videos (good balance)
- Cinematic mode produces best video quality but is slower
- Ensure video file is standard format (MP4, MOV)

## Troubleshooting

### Nothing happens when I click "Swap"
1. Check browser console for errors
2. Verify both images are valid and contain faces
3. Try a different image with more visible face
4. Check backend is running: `curl http://localhost:5001/api/hybrid/health`

### "No faces detected"
- Make sure image has clear, visible face
- Try rotating image if face is at angle
- Increase image brightness/contrast
- Use high-quality image

### Very slow processing
- Try **Fast** mode instead
- Reduce image size before uploading
- Check system resources (RAM, CPU)
- Upgrade to better hardware

### Memory errors
- Use **Fast** mode
- Disable enhancement features
- Process smaller images
- Restart application

## Command Reference

### Python CLI Usage

```python
# Import pipeline
from pipelines.hybrid_face_swap import HybridFaceSwapPipeline
from core.mode_controller import ProcessingMode
import cv2

# Create pipeline
pipeline = HybridFaceSwapPipeline(ProcessingMode.STUDIO)

# Load images
source = cv2.imread('source.jpg')
target = cv2.imread('target.jpg')

# Process
swapped, metadata = pipeline.process(source, target)

# Save
cv2.imwrite('result.jpg', swapped)

# Print info
print(f"Success: {metadata['success']}")
print(f"Time: {metadata['total_time']:.2f}s")
```

## API Quick Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/hybrid/modes` | GET | List available modes |
| `/api/hybrid/mode/<name>` | GET | Get mode details |
| `/api/hybrid/swap/image` | POST | Swap faces (image) |
| `/api/hybrid/swap/batch` | POST | Batch process images |
| `/api/hybrid/swap/video` | POST | Process video |
| `/api/hybrid/detect-faces` | POST | Detect faces in image |
| `/api/hybrid/download/<id>` | GET | Download result |
| `/api/hybrid/gpu-info` | GET | GPU information |
| `/api/hybrid/health` | GET | Service health |

## Performance Summary

### Speed (Single Image, 512x512)
- **Fast Mode**: < 1 second
- **Studio Mode**: 3-5 seconds
- **Cinematic Mode**: 10-20 seconds

### Quality
- **Fast**: Good (basic swap)
- **Studio**: Very Good (good balance)
- **Cinematic**: Excellent (maximum quality)

### Resource Usage
- **Fast**: 1.2GB RAM, minimal GPU
- **Studio**: 2.5GB RAM, moderate GPU
- **Cinematic**: 4GB RAM, high GPU

## Examples

### Example 1: Swap a Celebrity Face

```python
from pipelines.hybrid_face_swap import HybridFaceSwapPipeline
from core.mode_controller import ProcessingMode
import cv2

# Load images
me = cv2.imread('me.jpg')  # My photo
celebrity = cv2.imread('celebrity.jpg')  # Celebrity face

# Use Studio mode for balanced quality
pipeline = HybridFaceSwapPipeline(ProcessingMode.STUDIO)

# Swap: Put celebrity face on my body
result, _ = pipeline.process(celebrity, me)

# Save
cv2.imwrite('swapped.jpg', result)
```

### Example 2: Batch Process Family Photos

```python
from pipelines.hybrid_face_swap import HybridFaceSwapPipeline
from core.mode_controller import ProcessingMode
from pathlib import Path
import cv2

pipeline = HybridFaceSwapPipeline(ProcessingMode.FAST)

# Load my face
my_face = cv2.imread('my_face.jpg')

# Process all family photos
for photo in Path('family_photos').glob('*.jpg'):
    family_member = cv2.imread(str(photo))
    result, _ = pipeline.process(my_face, family_member)
    cv2.imwrite(f'swapped_{photo.stem}.jpg', result)
```

### Example 3: High-Quality Portrait

```python
from pipelines.hybrid_face_swap import HybridFaceSwapPipeline
from core.mode_controller import ProcessingMode
import cv2

# Use Cinematic mode for best quality
pipeline = HybridFaceSwapPipeline(ProcessingMode.CINEMATIC)

source = cv2.imread('source_portrait.jpg')
target = cv2.imread('target_portrait.jpg')

result, metadata = pipeline.process(source, target)

print(f"Processing took {metadata['total_time']:.1f} seconds")
print(f"Enhanced: {metadata['stages'].get('enhancement', {}).get('methods', [])}")

cv2.imwrite('high_quality_swapped.jpg', result)
```

## Getting Help

1. Check logs in backend terminal
2. Enable debug logging: `logging.basicConfig(level=logging.DEBUG)`
3. Test components individually
4. Review error messages carefully
5. Check system resources

## Next Steps

1. ✓ Read this quick start
2. ✓ Run `test_pipeline.py` to verify setup
3. ✓ Test with your own images
4. ✓ Try different modes
5. ✓ Integrate with frontend
6. ✓ Deploy to production

---

**Need more help?** Check `HYBRID_PIPELINE_GUIDE.md` for detailed documentation.
