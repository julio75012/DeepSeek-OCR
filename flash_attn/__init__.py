"""
Flash Attention Monkey Patch for ROCm/AMD GPUs
Uses PyTorch's optimized scaled_dot_product_attention
"""
import torch
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore", message="Using ROCm-compatible flash-attn")
__version__ = "2.5.0"

def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False):
    """ROCm-compatible flash attention using native PyTorch"""
    q, k, v = qkv.unbind(dim=2)
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, 
                                         is_causal=causal, scale=softmax_scale)

def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False):
    """Standard API compatibility"""
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p,
                                         is_causal=causal, scale=softmax_scale)

flash_attn_varlen_func = flash_attn_func
flash_attn_with_kvcache = None
