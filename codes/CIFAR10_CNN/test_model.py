import sys
import subprocess

# 检查是否有 NVIDIA GPU
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
if result.returncode == 0:
    print("✅ NVIDIA GPU 已检测到")
    print(result.stdout.split('\n')[2])  # 打印 GPU 名称行
else:
    print("❌ 未检测到 NVIDIA GPU")

# PyTorch 检查
try:
    import torch
    print(f"PyTorch CUDA: {torch.cuda.is_available()}")
except ImportError:
    print("PyTorch 未安装")