# System Requirements and Installation Guide

## Hardware

| Component | Specification | Status |
|-----------|---------------|--------|
| GPU | NVIDIA GeForce RTX 4060 Laptop | ✅ Detected |
| VRAM | 8 GB | ✅ Available |
| Compute Capability | 8.9 (Ada) | ✅ CUDA 12.x supported |
| RT Cores | Gen 3 | ✅ Hardware Ray Tracing |
| Tensor Cores | Gen 4 | ✅ AI Denoising |

## Required Software

### 1. CUDA Toolkit 12.x

**Purpose**: CUDA compiler (nvcc), runtime libraries, development headers

**Status**: ✅ INSTALLED (v12.0)

```bash
# Verification
nvcc --version

# Output:
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2023 NVIDIA Corporation
# Built on Fri_Jan__6_16:45:21_PST_2023
# Cuda compilation tools, release 12.0, V12.0.140
# Build cuda_12.0.r12.0/compiler.32267302_0
```

**Environment Variables** (added to ~/.bashrc):
```bash
export CUDA_HOME=/usr
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

**Compilation Test**: ✅ Passed
- CUDA Devices: 1
- Device: NVIDIA GeForce RTX 4060 Laptop GPU
- Compute Capability: 8.9
- VRAM: 8187 MB

---

### 2. NVIDIA Driver

**Status**: ✅ Already installed via WSL

```bash
nvidia-smi
# Shows driver version, CUDA version, GPU info
```

---

### 3. CMake 3.18+

**Status**: ✅ Installed

```bash
cmake --version
# cmake version 3.28.0
```

---

### 4. Python 3.8+

**Status**: ✅ Installed

```bash
python3 --version
# Python 3.12.x
```

---

### 5. Git

**Status**: ✅ Installed

```bash
git --version
# git version 2.x
```

---

## Optional Software

### OptiX 8.0 (Ray Tracing SDK)

**Purpose**: NVIDIA's ray tracing API, optimized for RT Cores

**Installation**:

```bash
# Download from https://developer.nvidia.com/optix
# Select: OptiX 8.0.x SDK

wget OptiX-8.0.0-sdk-linux64.sh
chmod +x OptiX-8.0.0-sdk-linux64.sh
./OptiX-8.0.0-sdk-linux64.sh

# Add to PATH
export OPTIX_INSTALL_DIR=/opt/OptiX_sdk_8.0.0
export LD_LIBRARY_PATH=$OPTIX_INSTALL_DIR/lib64:$LD_LIBRARY_PATH
```

**Manual Installation Required**: Yes

---

### Nsight Tools (Profiling)

**Purpose**: CUDA profiling and debugging

```bash
# Install Nsight Compute
sudo apt-get install -y nvidia-nsight

# Install Nsight Systems (timeline profiling)
sudo apt-get install -y nvidia-nsight-systems
```

**Manual Installation Required**: Yes (requires sudo)

---

## WSL2 Specific Setup

### Enable WSLg (Display)

```bash
# WSLg should be enabled by default in WSL2
# Verify with:
echo $WAYLAND_DISPLAY
# Should show something like: wayland-0
```

### GPU Passthrough Verification

```bash
# Verify GPU is accessible in WSL
nvidia-smi

# Check CUDA devices
ls /dev/nvidia*
# Should show: /dev/nvidia0, /dev/nvidiactl, /dev/nvidia-modeset
```

---

## Directory Structure After Installation

```
/usr/local/cuda/
├── bin/
│   ├── nvcc           # CUDA compiler
│   ├── cuobjdump      # Object dump utility
│   └── nvdisasm       # Disassembler
├── include/
│   ├── cuda.h
│   ├── cuda_runtime.h
│   └── curand.h
├── lib64/
│   ├── libcudart.so
│   ├── libcublas.so
│   └── libcurand.so
└── samples/           # Example programs
```

---

## Environment Variables

Add to `~/.bashrc` or `~/.zshrc`:

```bash
# CUDA
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# OptiX (if installed)
export OPTIX_HOME=/opt/OptiX_sdk_8.0.0
export LD_LIBRARY_PATH=$OPTIX_HOME/lib64:$LD_LIBRARY_PATH

# Nsight
export NSIGHT_COMPUTE_ROOT=/usr/local/NsightCompute
export PATH=$NSIGHT_COMPUTE_ROOT/bin:$PATH
```

Then run:
```bash
source ~/.bashrc
```

---

## Verification Commands

After installation, run these to verify:

```bash
# 1. Check CUDA compiler
nvcc --version

# 2. Check driver
nvidia-smi

# 3. Compile a simple CUDA program
cat > test.cu << 'EOF'
#include <stdio.h>
__global__ void hello() {
    printf("Hello from GPU!\n");
}
int main() {
    hello<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}
EOF
nvcc test.cu -o test
./test

# 4. Check device properties
cat > device_query.cu << 'EOF'
#include <stdio.h>
int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("CUDA Devices: %d\n", deviceCount);
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d: %s (CC %d.%d)\n", i, prop.name, prop.major, prop.minor);
    }
    return 0;
}
EOF
nvcc device_query.cu -o device_query
./device_query
```

---

## Troubleshooting

### "cuda.h not found"
```bash
# Install CUDA development package
sudo apt-get install -y cuda-dev-12-0
```

### "nvcc: command not found"
```bash
# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
```

### "No CUDA-capable device"
- Verify GPU is detected: `nvidia-smi`
- Check WSL2 GPU passthrough is enabled in Windows

### Permission denied on /dev/nvidia*
```bash
# Add user to video group
sudo usermod -aG video $USER
# Log out and back in
```
