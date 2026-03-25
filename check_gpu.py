#!/usr/bin/env python3
import torch
import sys

print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device count:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f'  Device {i}:', torch.cuda.get_device_name(i))
    if hasattr(torch.version, 'hip'):
        print('ROCm version:', torch.version.hip)
else:
    print('CUDA not available. Possible reasons:')
    print('1. ROCm not installed correctly')
    print('2. User not in video group')
    print('3. HSA_OVERRIDE_GFX_VERSION not set')
    print('4. GPU not supported')
    sys.exit(1)