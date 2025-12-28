# GPU Acceleration Implementation Plan

**Branch**: `GPU`
**Base**: `without-obs` (stable, working version)
**Objective**: Achieve 10-20x speedup through GPU acceleration

---

## Executive Summary

This document outlines the phased implementation of GPU acceleration for BG-Soft rendering pipeline. Based on simulation testing, we target:

- **Phase 1**: 5-7x speedup (TensorRT ONNX)
- **Phase 2**: 8-10x speedup (CUVID GPU decode)
- **Phase 3**: 10-15x speedup (CuPy GPU compositing)
- **Phase 4**: 12-20x speedup (NVENC tuning + optimization)

---

## Phase 1: TensorRT ONNX Runtime (Expected: 5-7x)

### Objective
Replace CPU-only ONNX inference with GPU-accelerated TensorRT provider.

### Changes Required
1. **Dependency Update**
   ```bash
   pip uninstall onnxruntime
   pip install onnxruntime-gpu
   ```

2. **Code Changes** (local_renderer.py)
   ```python
   # Change from:
   providers = ["CPUExecutionProvider"]

   # To:
   providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
   ```

3. **Requirements**
   - CUDA Toolkit 12.x
   - cuDNN 9.x
   - NVIDIA GPU with compute capability >= 5.0

4. **Testing**
   - Process 2 test videos (300 frames each)
   - Benchmark vs. current CPU version
   - Verify output quality matches

### Success Criteria
- ✅ ONNX inference runs on GPU
- ✅ Speedup >= 5x measured
- ✅ No quality degradation
- ✅ CPU fallback works when GPU unavailable

---

## Phase 2: Hardware Video Decoding (Expected: 8-10x)

### Objective
Enable CUVID hardware decoding and keep frames on GPU to avoid PCIe transfers.

### Changes Required
1. **FFmpeg Decode Command**
   ```python
   # Add to iter_frames():
   cmd_decode = [
       "ffmpeg",
       "-hwaccel", "cuda",
       "-hwaccel_output_format", "cuda",
       "-i", str(input_video),
       "-vf", "scale_cuda=w=width:h=height",
       # ... rest of command
   ]
   ```

2. **Key Optimization**: Keep decoded frames on GPU
   - Avoid CPU transfers after decode
   - Use `scale_cuda` instead of software scale
   - Result: frames stay in GPU memory throughout

3. **Requirements**
   - FFmpeg compiled with CUVID support
   - Check: `ffmpeg -hwaccels | grep cuda`

4. **Testing**
   - Verify FFmpeg has CUVID support
   - Test frame transfer efficiency
   - Measure decode speedup

### Success Criteria
- ✅ CUVID decoding works
- ✅ Frames kept on GPU (verify memory layout)
- ✅ Speedup >= 8x total (Phase 1 + 2)
- ✅ CPU fallback if CUVID unavailable

---

## Phase 3: GPU Image Processing with CuPy (Expected: 10-15x)

### Objective
Implement GPU-native image compositing using CuPy instead of NumPy.

### Changes Required
1. **CuPy Installation**
   ```bash
   pip install cupy-cuda12x  # CUDA 12.x
   ```

2. **Code Changes** (local_renderer.py)
   - Rewrite `apply_mask_filters()` with CuPy
   - Rewrite `composite()` with CuPy
   - Keep data in GPU memory (cupy.ndarray)

3. **Memory Management**
   ```python
   # Pseudo-code:
   mask_gpu = cupy.asarray(mask)  # CPU to GPU
   mask_filtered = cupy_blur(mask_gpu)
   frame_gpu = cupy.asarray(frame)
   output_gpu = cupy_composite(frame_gpu, mask_filtered)
   output = cupy.asnumpy(output_gpu)  # GPU to CPU (only output)
   ```

4. **Optimization Strategy**
   - Minimize transfers: only send input, get output
   - Batch processing for efficiency
   - Memory chunking if GPU exhausted

5. **Testing**
   - Profile memory usage
   - Test on 2 videos
   - Verify image quality (compare pixel values)
   - Handle out-of-memory errors gracefully

### Success Criteria
- ✅ CuPy operations work correctly
- ✅ Speedup >= 10x total
- ✅ Output quality matches CPU version
- ✅ Memory errors fall back to CPU

---

## Phase 4: NVENC Tuning & Optimization (Expected: 12-20x)

### Objective
Fine-tune NVENC parameters and eliminate remaining bottlenecks.

### Changes Required
1. **Aggressive NVENC Settings**
   ```python
   # Modify encode_video():
   # From: preset p4 (quality-focused)
   # To:   preset p2 or p1 (speed-focused)

   cmd_encode = [
       # ...
       "-preset", "p2",      # Faster encoding
       "-rc", "vbr_hq",      # Variable bitrate, high quality
       "-cq", "18",          # Adjust quality/speed
       "-look-ahead", "20",  # Frame lookahead for better rate control
       # ...
   ]
   ```

2. **GPU→File Direct Transfer**
   - Minimize buffer copies
   - Use zero-copy encoding if available
   - Monitor GPU memory during encoding

3. **Multi-Instance NVENC** (Optional)
   - Process multiple videos in parallel
   - Each uses separate NVENC engine
   - Requires careful GPU memory management

4. **Testing**
   - Benchmark encoding speed
   - Verify quality (VMAF scores)
   - Test on longer videos (500+ frames)

### Success Criteria
- ✅ Encoding speedup measured
- ✅ Total speedup >= 12x
- ✅ No quality loss (acceptable trade-off possible)
- ✅ Stable for multi-file processing

---

## Implementation Checklist

### Phase 1
- [ ] Update onnxruntime to GPU version
- [ ] Change ONNX provider to TensorRT
- [ ] Install CUDA 12.x + cuDNN 9.x
- [ ] Test 2 videos, benchmark
- [ ] Verify CPU fallback works
- [ ] Commit: "Phase 1: TensorRT ONNX GPU acceleration"

### Phase 2
- [ ] Check FFmpeg CUVID support
- [ ] Add `-hwaccel cuda` to decode
- [ ] Add `scale_cuda` filter
- [ ] Test frame memory layout (on GPU)
- [ ] Benchmark decode speedup
- [ ] Test CPU fallback
- [ ] Commit: "Phase 2: CUVID hardware decode + GPU memory pipeline"

### Phase 3
- [ ] Install CuPy
- [ ] Rewrite mask_filters with CuPy
- [ ] Rewrite composite with CuPy
- [ ] Profile memory usage
- [ ] Handle memory errors
- [ ] Test 2 videos, verify quality
- [ ] Commit: "Phase 3: GPU compositing with CuPy"

### Phase 4
- [ ] Adjust NVENC preset and parameters
- [ ] Enable lookahead
- [ ] Test aggressive settings
- [ ] Benchmark total speedup
- [ ] Test multi-file processing
- [ ] Optimize remaining bottlenecks
- [ ] Commit: "Phase 4: NVENC tuning and optimization"

### Final
- [ ] End-to-end testing (3+ videos)
- [ ] Benchmark final speedup
- [ ] Document achieved speedup
- [ ] Compare quality to CPU version
- [ ] Test fallback scenarios
- [ ] Create PR to main branch

---

## Testing Protocol

### Per-Phase Testing
1. **Functional Test**
   - Process 1 test video (300 frames)
   - Verify output file created
   - Check file size reasonable

2. **Performance Test**
   - Measure time per phase
   - Calculate speedup
   - Log results

3. **Quality Test**
   - Compare output frames to CPU version
   - Check for artifacts
   - Verify metadata preserved

4. **Stability Test**
   - Process 2+ videos back-to-back
   - Monitor memory usage
   - Check for crashes/hangs

### Fallback Testing
- Disable GPU feature
- Verify automatic fallback
- Test mixed GPU/CPU pipeline

---

## Known Risks & Mitigations

| Risk | Probability | Mitigation |
|------|-------------|-----------|
| CUVID not available | Medium | CPU fallback decode |
| GPU memory exhaustion | Low | Chunked processing |
| ONNX GPU errors | Low | CPU provider fallback |
| Output quality degradation | Low | Careful parameter tuning |
| Performance regression | Very Low | Benchmark each phase |

---

## Success Metrics

- **Primary**: Total speedup >= 10x
- **Secondary**: Total speedup = 15-20x (ambitious target)
- **Quality**: No visible quality loss vs. CPU version
- **Stability**: Processes 5+ videos without crash
- **Robustness**: Graceful fallback to CPU for all operations

---

## Timeline Estimate

- Phase 1: 1-2 hours
- Phase 2: 2-3 hours
- Phase 3: 3-4 hours
- Phase 4: 2-3 hours
- Testing & Optimization: 3-4 hours
- **Total**: 11-17 hours

---

## Resources

### Research
- [ONNX Runtime CUDA Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [FFmpeg NVIDIA GPU Acceleration](https://docs.nvidia.com/video-technologies/video-codec-sdk/12.0/ffmpeg-with-nvidia-gpu/index.html)
- [CuPy Documentation](https://cupy.dev/)
- [NVIDIA CUDA Python](https://developer.nvidia.com/how-to-cuda-python)

### Simulation Results
- GPU Simulation: `/tmp/gpu_simulation.py`
- Optimization Analysis: `/tmp/gpu_simulation_optimized.py`
- Final Strategy: `/tmp/gpu_final_strategy.py`

---

## Notes

1. **Fallback Strategy**: Every GPU operation has a CPU equivalent. If GPU fails, system automatically falls back to CPU version.

2. **Memory Management**: GPU memory is precious. Implement careful buffer management and cleanup.

3. **Quality vs. Speed**: Some optimizations may trade quality for speed. Monitor output quality carefully.

4. **Testing on Target Hardware**: Ensure testing matches deployment GPU (different GPUs may have different characteristics).

---

## Next Steps

1. Start Phase 1 implementation
2. Test thoroughly on 2 videos
3. Commit and document results
4. Move to Phase 2
5. Continue iteratively

---

**Created**: 2025-12-19
**Status**: Ready for Phase 1 implementation
**Branch**: GPU
**Base**: without-obs (commit 8b4d1b0)
