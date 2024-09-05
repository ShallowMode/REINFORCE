import torch

# PyTorch 설치가 잘 되었는지 확인
print("PyTorch version:", torch.__version__)

# GPU 사용 가능 여부 확인
print("CUDA is available:", torch.cuda.is_available())

# 간단한 텐서 연산
x = torch.rand(5, 3)
print("Random tensor:\n", x)