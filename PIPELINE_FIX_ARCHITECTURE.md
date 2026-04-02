# Face Swap Pipeline Architecture Fix

## 🎯 Problem Analysis

### Current Issues:
1. **Identity Loss** - GFPGAN/CodeFormer applied to full image, restoring original identity
2. **Full-Image Processing** - Enhancement shouldn't touch background or full frame
3. **No Identity Lock** - InsightFace embedding not preserved during enhancement
4. **Wrong Diffusion Settings** - High denoise values allow face regeneration
5. **No DeepFaceLab Integration** - Only placeholder comment exists
6. **Resolution Downscaling** - Models force 256x256 or other small sizes
7. **Mask Misalignment** - Mask generated from landmarks but not properly preserved

### Why Fast Mode Works:
- Simple InsightFace swap (preserves identity by design)
- No enhancement layers that can introduce artifacts
- No models attempting to restore original face

### Why Studio/Cinematic Fail:
- GFPGAN with `has_aligned=False, only_center_face=False` processes entire image
- GFPGAN can apply face restoration, rewriting the swapped identity
- No embedding match check after enhancement
- Diffusion refiner is placeholder only

---

## ✅ Correct Pipeline Architecture

### Phase 1: Face Swap (LOCKED)
```
Input Image
    ↓
Detect Face (InsightFace)
    ↓
Swap Face (InsightFace inswapper_128)
    ↓
LOCK: Save embedding + bbox + landmarks
    ↓
Output: Base swapped image (InsightFace final)
```

### Phase 2: Extract Swapped Face Crop (HIGH RESOLUTION)
```
Swapped Image + Locked Bbox
    ↓
Extract Face Crop at ORIGINAL resolution
    ↓
Do NOT downscale
    ↓
Face Crop (high-res, locked identity inside)
```

### Phase 3: Enhance Face Crop ONLY (MASKED)
```
Face Crop (high-res)
    ↓
├─ Option A: CodeFormer Refinement
│  ├─ Input: Face crop only (NOT full image)
│  ├─ Fidelity: 0.8 (prefer quality over changes)
│  ├─ No face restoration on entire image
│  └─ Output: Enhanced face crop
│
├─ Option B: DeepFaceLab Detail Enhancement
│  ├─ Match input crop size (don't auto-resize)
│  ├─ Preserve face landmarks alignment
│  ├─ Refine texture (NOT identity)
│  └─ Output: Texture-enhanced crop
│
└─ Option C: Diffusion Refinement (if enabled)
   ├─ Denoise: 0.15-0.25 (NOT 0.4+)
   ├─ Prompt: Neutral (no identity rewrite)
   ├─ Mask: Face region only
   ├─ ControlNet: Use to preserve structure
   └─ Output: Subtly refined crop
```

### Phase 4: Verify Identity Preserved
```
Enhanced Face Crop
    ↓
Extract Embedding
    ↓
Compare with Locked Embedding
    ↓
Similarity > 0.90?
├─ YES → Continue
└─ NO → Fallback to basic enhancement
```

### Phase 5: Blend Enhanced Face Back
```
Enhanced Face Crop + Original Mask
    ↓
Create Gaussian Feathered Mask (15-25px kernel)
    ↓
Seamless Blend (Poisson Blending)
    ├─ Input: Full swapped image, enhanced crop
    ├─ Method: cv2.seamlessClone
    ├─ Mask: Feathered face mask
    └─ Output: Blended image
    ↓
Apply Color Correction
    ├─ LAB color transfer
    ├─ Match lighting (source → target)
    └─ Output: Color-corrected image
```

### Phase 6: Global Refinement (SAFE)
```
Color-Corrected Image
    ↓
├─ Soft Sharpening (if Studio/Cinematic)
│  └─ Unsharp mask: 1.0 strength, small kernel
│
└─ Noise Matching (if Cinematic)
    └─ Match target image grain
```

---

## 🔥 Mode Specifications

### Fast Mode
```
1. Detect faces
2. Swap (InsightFace)
3. Output
Time: ~2s
Quality: Good (baseline)
```

### Studio Mode
```
1. Detect faces
2. Swap (InsightFace) → LOCK embedding
3. Extract swapped face crop (high-res, no downscale)
4. CodeFormer refine (fidelity=0.7, face crop only)
5. Verify embedding similarity (must be >0.90)
6. Seamless blend back (Poisson + feathered mask)
7. Color correction (LAB transfer)
8. Light sharpening (Unsharp mask 1.0)
9. Output
Time: ~30s
Quality: Better (enhanced texture, cleaner edges)
```

### Cinematic Mode
```
1. Detect faces
2. Swap (InsightFace) → LOCK embedding
3. Extract swapped face crop (high-res, no downscale)
4. CodeFormer refine (fidelity=0.8, face crop only)
5. Verify embedding similarity (must be >0.90)
6. Optional: Subtle diffusion refinement (denoise=0.20, masked)
7. Seamless blend back (Poisson + feathered mask)
8. Advanced color correction (LAB + gamma adaptation)
9. Tone mapping (cinematic grade)
10. Global soft sharpening
11. Output
Time: ~60s
Quality: Best (cinematic, realistic, no distortion)
```

---

## 🛡️ Identity Preservation Safeguards

### Before Enhancement:
```python
# 1. Extract embedding from swapped face immediately after swap
swapped_embedding = insightface_model.get_embedding(swapped_image, swapped_face)
swapped_bbox = face.bbox
swapped_landmarks = face.kps
```

### During Enhancement:
```python
# 2. Only process face crop, NOT full image
face_crop = swapped_image[y1:y2, x1:x2]  # high-res, no downscale
enhanced_crop = enhance_face_crop_only(face_crop, model)
```

### After Enhancement:
```python
# 3. Verify identity unchanged
enhanced_embedding = insightface_model.get_embedding(enhanced_crop, ...)
similarity = cosine_similarity(swapped_embedding, enhanced_embedding)

if similarity < 0.90:
    print("⚠️  Identity drift detected! Fallback to basic enhancement")
    use_fallback_enhancement()
else:
    print("✓ Identity preserved")
```

---

## ❌ What NOT to Do

1. **DO NOT** apply GFPGAN/CodeFormer to full image
   - ✓ Apply only to face crop
   
2. **DO NOT** use high denoise values in diffusion
   - ✓ Use denoise 0.15-0.30

3. **DO NOT** downscale image to 256x256
   - ✓ Keep original resolution

4. **DO NOT** skip identity verification
   - ✓ Check embedding similarity after enhancement

5. **DO NOT** apply face restoration tools on background
   - ✓ Apply mask-only processing

6. **DO NOT** regenerate face instead of enhance
   - ✓ Refine details, not recreate

---

## 📊 Processing Order Diagram

```
┌─────────────────────────────────────────────────────────┐
│ INPUT: Source + Target Images                           │
└────────────────┬────────────────────────────────────────┘
                 ↓
        ┌────────────────────┐
        │ Detect Faces       │
        │ (InsightFace)      │
        └────────┬───────────┘
                 ↓
        ┌────────────────────┐
        │ Swap Face          │
        │ (InsightFace)      │
        └────────┬───────────┘
                 ↓
   ╔════════════════════════════╗
   ║ LOCK Identity              ║
   ║ - Save embedding           ║
   ║ - Save bbox                ║
   ║ - Save landmarks           ║
   ╚════════┬═══════════════════╝
            ↓
   ╔════════════════════════════╗
   ║ Extract Face Crop          ║
   ║ - High resolution          ║
   ║ - NO downscale             ║
   ╚════════┬═══════════════════╝
            ↓
   ┌────────────────────────────┐
   │ Choose Enhancement Method  │
   │ ┌──────────┐               │
   │ │ Fast     │ → Skip        │
   │ ├──────────┤               │
   │ │ Studio   │ → CodeFormer  │
   │ │          │    (0.7)      │
   │ ├──────────┤               │
   │ │ Cinematic│ → CodeFormer  │
   │ │          │    (0.8) +    │
   │ │          │    Diffusion  │
   │ └──────────┘               │
   └────────┬───────────────────┘
            ↓
   ╔════════════════════════════╗
   ║ Enhance Face Crop Safely   ║
   ║ - Mask-only processing     ║
   ║ - Preserve landmarks       ║
   ╚════════┬═══════════════════╝
            ↓
   ╔════════════════════════════╗
   ║ Verify Identity Preserved  ║
   ║ - Compare embeddings       ║
   ║ - Similarity > 0.90?       ║
   ║ - Fallback if failed       ║
   ╚════════┬═══════════════════╝
            ↓
   ┌────────────────────────────┐
   │ Seamless Blend             │
   │ - Poisson blending         │
   │ - Feathered mask           │
   └────────┬───────────────────┘
            ↓
   ┌────────────────────────────┐
   │ Color Correction           │
   │ - LAB transfer             │
   │ - Gamma adaptation         │
   └────────┬───────────────────┘
            ↓
   ┌────────────────────────────┐
   │ Global Refinement          │
   │ - Soft sharpening          │
   │ - Noise matching           │
   └────────┬───────────────────┘
            ↓
        ┌────────────────────┐
        │ OUTPUT: Final      │
        │ Swapped Image      │
        └────────────────────┘
```

---

## 🧪 Expected Results

### Fast Mode:
- ✓ Clean face swap
- ✓ Identity preserved
- ✓ ~2s processing
- Result: **Good** (baseline)

### Studio Mode:
- ✓ Swapped identity locked
- ✓ Texture refinement applied
- ✓ Seamless blending
- ✓ Color correction
- ✓ No identity drift
- Result: **Better than Fast** (enhanced details)

### Cinematic Mode:
- ✓ All Studio features
- ✓ Subtle diffusion refinement
- ✓ Advanced color grading
- ✓ Cinematic tone mapping
- ✓ Professional lighting
- Result: **Best quality** (movie-grade)

---

## 🔧 Implementation Roadmap

### Step 1: Create `identity_lock.py`
- Extract and store embeddings
- Verify similarity after enhancement
- Fallback mechanism

### Step 2: Create `safe_enhancer.py`
- Face crop extraction (no downscale)
- Identity-aware enhancement
- Embedding verification

### Step 3: Create `dfl_refiner.py`
- DeepFaceLab texture refinement
- Proper model loading
- Correct alignment handling

### Step 4: Create `diffusion_refiner_safe.py`
- Proper denoise values (0.15-0.30)
- Neutral prompts
- Mask-only processing

### Step 5: Update `image_pipeline.py`
- Correct order of operations
- Use new modules
- Add identity checks

### Step 6: Testing & Validation
- Test Fast Mode (should remain good)
- Test Studio Mode (should improve)
- Test Cinematic Mode (should be best)
- Verify no identity loss
- Check for artifacts

---

## 🧠 Technical Notes

### Why CodeFormer is Better Than GFPGAN:
- CodeFormer has fidelity weight (0.0-1.0) to preserve identity
- GFPGAN doesn't offer explicit identity lock
- CodeFormer paper: "Towards Robust Blind Face Restoration with Codebook Prior"

### Denoise Values Explained:
- **0.15-0.25**: Subtle refinement, structure preserved
- **0.30-0.40**: Noticeable enhancement, some risk
- **0.50+**: Likely to regenerate face, identity loss

### Seamless Blending:
- Poisson blending preserves gradients at boundaries
- Feathered mask avoids hard edge artifacts
- Gaussian kernel size: 15-25px (depends on image size)

### Color Transfer:
- LAB space is perceptually uniform (better than RGB)
- Avoids over-saturation or color banding
- Gamma adaptation handles backlight scenarios

