# Summary: DeepSeek-OCR on AMD Radeon 780M (ROCm)

---

## 1. Hardware/OS Configuration

| Component | Specification |
|-----------|--------------|
| **Laptop** | Lenovo ThinkPad P16s Gen 2 (AMD) |
| **CPU** | AMD Ryzen 7 PRO 7840U (8C/16T, up to 5.1GHz) |
| **GPU** | AMD Radeon 780M (Integrated/Phoenix) |
| **GPU Architecture** | RDNA3 (gfx1103) |
| **RAM** | 16-64GB LPDDR5X-7500 (soldered) |
| **OS** | openSUSE Leap 15.6 |
| **Kernel** | 6.4.0-150600.23.84-default |

**Critical Note:** The Radeon 780M is an **integrated GPU (iGPU)** with gfx1103 architecture. AMD does not officially support gfx1103 in ROCm, but it can be made to work with overrides.

---

## 2. Setup Completed

### A. ROCm Installation
- **ROCm Version:** 6.2.2
- **Installation method:** SLES 15.6 packages on openSUSE Leap 15.6
- **Repositories:** 
  - `https://repo.radeon.com/amdgpu/6.2.2/sle/15.6/main/x86_64/`
  - `https://repo.radeon.com/rocm/zyp/6.2.2/main`
- **Packages installed:** `rocm-core`, `rocm-smi-lib`, `hip-runtime-amd`, `rocm-opencl-runtime`, `rocminfo`

### B. PyTorch ROCm Environment
- **PyTorch:** 2.5.1+rocm6.2
- **TorchVision:** 0.20.1+rocm6.2
- **Python:** 3.12 (venv at `~/Documents/DeepSeek-OCR/.venv`)
- **Transformers:** 4.46.3
- **Accelerate:** >=0.26.0

### C. Critical Environment Variables (in `.venv/bin/activate`)
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0 # Forces gfx1100 proxy for gfx1103
export ROCM_PATH=/opt/rocm-6.2.2
export HF_HOME="/home/jules/.cache/huggingface"
export HF_HUB_OFFLINE=0
export TRANSFORMERS_CACHE="/home/jules/.cache/huggingface"
export HF_HUB_CACHE="/home/jules/.cache/huggingface/hub"
```

### D. HuggingFace Cache
- **Location:** `/home/jules/.cache/huggingface/hub/`
- **Model:** `deepseek-ai/DeepSeek-OCR` (6.67GB, downloaded successfully)
- **Cache directories:** Both `models--deepseek-ai--deepseek-ocr` (lowercase) and `models--deepseek-ai--DeepSeek-OCR` (camelCase) exist

### E. Model Modifications Made
- Changed `_attn_implementation='flash_attention_2'` to `'eager'` (Flash Attention is CUDA-only)
- Commented out `os.environ["CUDA_VISIBLE_DEVICES"] = '0'` (NVIDIA-specific)
- Script location: `~/Documents/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-hf/run_dpsk_ocr.py`

---

## 3. Current Problem (Stuck Here)

### Symptoms
- **rocminfo** no longer shows GPU (Agent 2 / gfx1103 missing)
- **PyTorch** reports `torch.cuda.is_available() = False`
- **Error:** `RuntimeError: No HIP GPUs are available`
- Environment variables (`HSA_OVERRIDE_GFX_VERSION=11.0.0`) are set but not working

### What Changed
- Previously working: rocminfo showed gfx1103, PyTorch could run matmul on GPU
- Now broken: After system restart or shell session change, GPU disappeared from ROCm view
- Same environment variables set in both cases

### Diagnostic Commands to Run
```bash
# Check if amdgpu kernel module is loaded
lsmod | grep amdgpu

# Check if GPU visible in PCI
lspci | grep -i vga

# Check ROCm GPU detection
/opt/rocm-6.2.2/bin/rocminfo | grep -E "(Agent|gfx110)"

# Check kernel messages for driver errors
dmesg | grep -i amdgpu | tail -20

# Verify environment variables are set
echo $HSA_OVERRIDE_GFX_VERSION
echo $ROCM_PATH

# Check PyTorch ROCm status
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### Key Files
- Modified activate script: `/home/jules/Documents/DeepSeek-OCR/.venv/bin/activate`
- Model script: `/home/jules/Documents/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-hf/run_dpsk_ocr.py`
- HuggingFace cache: `/home/jules/.cache/huggingface/hub/`

### Working Configuration (Historical)
At one point, this worked:
- rocminfo showed Agent 2 with gfx1103
- PyTorch 2.5.1+rocm6.2 with `HSA_OVERRIDE_GFX_VERSION=11.0.0` could use GPU
- Successfully ran: `torch.matmul(torch.randn(100,100).cuda(), torch.randn(100,100).cuda())`
- Model loaded on `cuda:0` with `device_map='auto'`

### Suspected Causes
1. Kernel driver (amdgpu) not loaded or crashed
2. ROCm runtime initialization order issue
3. Environment variables not propagating to Python process
4. GPU power management/suspension
5. Multiple Python environments causing conflicts

---

## For Your IT Colleague

**The core issue:** We have proven that gfx1103 (Radeon 780M) works with ROCm 6.2.2 using `HSA_OVERRIDE_GFX_VERSION=11.0.0` proxy. The setup was functional (model loaded, matmul worked), but after restart/session change, ROCm no longer sees the GPU despite identical environment configuration.

**Question to investigate:** Why does rocminfo show no GPU agents when it previously showed gfx1103? Is this a driver issue, environment propagation issue, or ROCm runtime state corruption?
