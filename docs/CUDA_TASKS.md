# CUDA Implementation Tasks

## Status: Pending Installation

**Installation Required:**
- [ ] Install CUDA Toolkit 12.x (`sudo apt-get install nvidia-cuda-toolkit`)
- [ ] Install OptiX 8.0 (optional, for advanced ray tracing)
- [ ] Configure environment variables

## Implementation Checklist

### Phase 1: Foundation
- [ ] Create `renderer/cuda/` directory structure
- [ ] Implement `cuda_runtime.h` - CUDA initialization/error handling
- [ ] Implement `cuda_context.h` - Context management
- [ ] Implement `cuda_memory.h` - Memory allocation (pinned, managed)
- [ ] Implement `cuda_stream.h` - Stream management
- [ ] Add CUDA build support to CMakeLists.txt
- [ ] Add fallback CPU renderer if CUDA unavailable

### Phase 2: Ray Tracing Core
- [ ] Create `renderer/cuda/rt/` directory
- [ ] Implement `bvh_builder.h` - SAH BVH construction
- [ ] Implement `bvh_node.h` - BVH node structures
- [ ] Implement `ray_intersect.h` - RT Core intersection
- [ ] Implement `ray_generator.h` - Ray generation kernels
- [ ] Implement `shading_kernel.h` - GPU material evaluation
- [ ] Implement `path_tracer.h` - Path tracing kernel
- [ ] Create `ray_tracer.h` main interface class

### Phase 3: Compute Kernels
- [ ] Create `renderer/cuda/compute/` directory
- [ ] Implement `particle_kernel.h` - GPU particle simulation
- [ ] Implement `physics_kernel.h` - GPU physics
- [ ] Implement `blur_kernel.h` - Gaussian blur
- [ ] Implement `denoise_kernel.h` - AI denoising
- [ ] Implement `reduction_kernel.h` - Parallel reduction
- [ ] Implement `sort_kernel.h` - Radix sort

### Phase 4: Integration
- [ ] Implement `cuda_opengl_interop.h` - OpenGL-CUDA sharing
- [ ] Implement `hybrid_renderer.h` - Select GPU path at runtime
- [ ] Implement `rt_hybridizer.h` - Raster + ray trace combination
- [ ] Update `RenderDevice` to use CUDA when available
- [ ] Add performance profiling hooks

### Phase 5: Optimization
- [ ] Multi-stream architecture (raster, compute, rt, copy)
- [ ] BVH incremental updates for dynamic geometry
- [ ] Temporal anti-aliasing integration
- [ ] DLSS/OMPTAI integration for denoising
- [ ] Benchmark and optimize critical paths

## File Structure to Create

```
renderer/cuda/
├── CMakeLists.txt
├── cuda_runtime.h
├── cuda_runtime.cpp
├── cuda_context.h
├── cuda_memory.h
├── cuda_stream.h
├── cuda_error.h
│
├── rt/
│   ├── CMakeLists.txt
│   ├── ray_tracer.h
│   ├── ray_tracer.cpp
│   ├── bvh_builder.h
│   ├── bvh_builder.cpp
│   ├── bvh_node.h
│   ├── ray_generator.h
│   ├── ray_intersect.h
│   ├── shading_kernel.h
│   ├── path_tracer.h
│   └── kernels/
│       ├── trace.cu
│       ├── shade.cu
│       └── denoise.cu
│
├── compute/
│   ├── CMakeLists.txt
│   ├── compute_scheduler.h
│   ├── particle_kernel.h
│   ├── physics_kernel.h
│   ├── blur_kernel.h
│   └── kernels/
│       ├── particles.cu
│       ├── physics.cu
│       └── blur.cu
│
└── interop/
    ├── cuda_opengl_interop.h
    └── cuda_opengl_interop.cpp

renderer/
├── hybrid_renderer.h
└── hybrid_renderer.cpp
```

## Dependencies

### Internal
- `core/math/Vector3.h`
- `core/math/Matrix4.h`
- `core/math/Quaternion.h`
- `core/math/Transform.h`
- `renderer/RenderDevice.h`
- `scene/resources/Mesh.h`
- `scene/resources/Material.h`

### External (System)
- CUDA Toolkit 12.x
- NVIDIA Driver 535+
- OpenGL (already integrated)

## Testing Plan

1. **Unit Tests**
   - Memory allocation/transfer
   - BVH construction accuracy
   - Ray-triangle intersection

2. **Integration Tests**
   - OpenGL-CUDA interop
   - Hybrid rendering pipeline

3. **Performance Tests**
   - Particle count scaling
   - Ray tracing FPS vs quality
   - Memory bandwidth utilization

## Estimated Development Time

| Phase | Effort |
|-------|--------|
| Foundation | 1 week |
| Ray Tracing Core | 2 weeks |
| Compute Kernels | 1 week |
| Integration | 1 week |
| Optimization | 1 week |
| **Total** | **~6 weeks** |
