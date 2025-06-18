import torch
print(torch.__version__)                     # → 2.3.0
print(torch.cuda.is_available())  # → True
print(torch.backends.cuda.sdp_kernel())
