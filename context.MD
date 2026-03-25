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

### C. Critical Environment Variables (in `~/.bashrc`)
```bash
# huggingface
export HF_HOME="/home/jules/Documents/huggingface_hub/downloads/"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_ENABLE_HF_TRANSFER=1     # Optional: faster downloads (requires hf_transfer package)
export HF_HUB_OFFLINE=0                # Optional: set to 1 only when you want strict offline mode

#ROCM param
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export ROCM_PATH=/opt/rocm-6.2.2
```

### D. HuggingFace Cache
- **Location:** `/home/jules/Documents/huggingface_hub/downloads//hub/`
- **Model:** `deepseek-ai/DeepSeek-OCR` (6.67GB, downloaded successfully)
- **Cache directories:**  `models--deepseek-ai--deepseek-ocr` 

### E. Model Modifications Made
- Changed `_attn_implementation='flash_attention_2'` to `'eager'` (Flash Attention is CUDA-only)
- Commented out `os.environ["CUDA_VISIBLE_DEVICES"] = '0'` (NVIDIA-specific)
- Script location: `~/Documents/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-hf/run_dpsk_ocr.py`


