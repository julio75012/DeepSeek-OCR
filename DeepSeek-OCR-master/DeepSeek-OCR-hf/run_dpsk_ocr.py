from transformers import AutoModel, AutoTokenizer
import torch
import os


#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#disable for amd

model_name = 'deepseek-ai/deepseek-ocr'


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name, 
    _attn_implementation='eager',  # Changed from 'flash_attention_2'
    trust_remote_code=True, 
    use_safetensors=True
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.eval().to(device)
if torch.cuda.is_available():
    model = model.to(torch.bfloat16)



# prompt = "<image>\nFree OCR. "
prompt = "<image>\n<|grounding|>describe the picture "
image_file = '/home/jules/Pictures/photo_2024-03-19_18-43-16.jpg'
output_path = './'



# infer(self, tokenizer, prompt='', image_file='', output_path = ' ', base_size = 1024, image_size = 640, crop_mode = True, test_compress = False, save_results = False):

# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False

# Gundam: base_size = 1024, image_size = 640, crop_mode = True

res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 1024, image_size = 640, crop_mode=True, save_results = True, test_compress = True)
