from transformers import AutoModel, AutoTokenizer
import torch
import os


# Auto-detect GPU backend
def get_device_and_attn():
    """Detect available GPU and choose the best attention implementation."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Check if this is an AMD GPU (ROCm) or NVIDIA (CUDA)
        gpu_name = torch.cuda.get_device_name(0).lower()
        is_amd = "amd" in gpu_name or "radeon" in gpu_name or hasattr(torch.version, "hip")
        if is_amd:
            # Flash Attention 2 is CUDA-only; use eager on AMD/ROCm
            attn_impl = "eager"
            print(f"Detected AMD GPU: {torch.cuda.get_device_name(0)} (using eager attention)")
        else:
            attn_impl = "flash_attention_2"
            print(f"Detected NVIDIA GPU: {torch.cuda.get_device_name(0)} (using flash attention)")
    else:
        device = torch.device("cpu")
        attn_impl = "eager"
        print("No GPU detected, using CPU (this will be slow)")
    return device, attn_impl


device, attn_impl = get_device_and_attn()

model_name = "deepseek-ai/DeepSeek-OCR"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation=attn_impl,
    trust_remote_code=True,
    use_safetensors=True,
)
model = model.eval().to(device)
if device.type == "cuda":
    model = model.to(torch.bfloat16)


# prompt = "<image>\nFree OCR. "
prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = "your_image.jpg"
output_path = "your/output/dir"


# infer(self, tokenizer, prompt='', image_file='', output_path = ' ', base_size = 1024, image_size = 640, crop_mode = True, test_compress = False, save_results = False):

# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False

# Gundam: base_size = 1024, image_size = 640, crop_mode = True

res = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=image_file,
    output_path=output_path,
    base_size=1024,
    image_size=640,
    crop_mode=True,
    save_results=True,
    test_compress=True,
)
