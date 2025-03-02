import triton
import triton.language as tl

import os
import torch

os.environ["TRITON_INTERPRET"] = "1"
DEVICE = "cuda:0"
BLOCK_SIZE = 2


@triton.jit
def vect_add_kernel(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: tl.constexpr, num_elements):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offset = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offset < num_elements

    print(f"PID: {pid}\nBlock start: {block_start}\nOffset: {offset}\nMask: {mask}\n")

    x = tl.load(x_ptr + offset, mask=mask)
    y = tl.load(y_ptr + offset, mask=mask)

    output = x + y
    tl.store(out_ptr + offset, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    assert x.device == y.device == torch.device(DEVICE)

    output = torch.empty_like(x)
    vect_add_kernel[(3,)](x, y, output, BLOCK_SIZE, num_elements=x.numel())
    return output


if __name__ == "__main__":

    x = torch.tensor([1, 2, 3, 4], device=DEVICE)
    y = torch.tensor([5, 6, 7, 8], device=DEVICE)

    res = add(x, y)
    print(res)