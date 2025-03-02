from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

cuda_src = Path("vectAddTorch.cu").read_text()
cpp_src = "torch::Tensor vect_add(torch::Tensor x, torch::Tensor y);"

# load the low-level CUDA and C++ code
module = load_inline(
    name="vectAdd",
    cuda_sources=[cuda_src],
    cpp_sources=[cpp_src],
    functions=["vect_add"],
    with_cuda=True,
    # nvcc compiler flag for optimization level
    extra_cuda_cflags=["-O2"],
)

# input tensors for vector addition
x = torch.tensor([1, 2, 3, 4], dtype=torch.int, device="cuda")
y = torch.tensor([1, 2, 6, 4], dtype=torch.int, device="cuda")

# vector addition function
res = module.vect_add(x, y)
print(res)

## Output: tensor([2, 4, 9, 8], device='cuda:0', dtype=torch.int32)