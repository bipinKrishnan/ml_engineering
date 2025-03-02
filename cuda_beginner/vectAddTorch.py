import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r'''
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>


__global__
void mean_filter_kernel(unsigned char* output, unsigned char* input, int width, int height, int radius) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = threadIdx.z;

    int baseOffset = channel * height * width;
    if (col < width && row < height) {

        int pixVal = 0;
        int pixels = 0;

        for (int blurRow=-radius; blurRow <= radius; blurRow += 1) {
            for (int blurCol=-radius; blurCol <= radius; blurCol += 1) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                if (curRow >= 0 && curRow < height && curCol >=0 && curCol < width) {
                    pixVal += input[baseOffset + curRow * width + curCol];
                    pixels += 1;
                }
            }
        }

        output[baseOffset + row * width + col] = (unsigned char)(pixVal / pixels);
    }
}


// helper function for ceiling unsigned integer division
inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}


torch::Tensor mean_filter(torch::Tensor image, int radius) {
    assert(image.device().type() == torch::kCUDA);
    assert(image.dtype() == torch::kByte);
    assert(radius > 0);

    const auto channels = image.size(0);
    const auto height = image.size(1);
    const auto width = image.size(2);

    auto result = torch::empty_like(image);

    dim3 threads_per_block(16, 16, channels);
    dim3 number_of_blocks(
        cdiv(width, threads_per_block.x),
        cdiv(height, threads_per_block.y)
    );

    mean_filter_kernel<<<number_of_blocks, threads_per_block, 0, torch::cuda::getCurrentCUDAStream()>>>(
        result.data_ptr<unsigned char>(),
        image.data_ptr<unsigned char>(),
        width,
        height,
        radius
    );

    // check CUDA error status (calls cudaGetLastError())
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}'''

cpp_src = "torch::Tensor mean_filter(torch::Tensor image, int radius);"

cuda_src = """
__global__ void vectAddKernel(int *a, int *b, int *c, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < len) {
        c[idx] = a[idx] + b[idx];
    }
}

torch::Tensor vect_add(torch::Tensor x, torch::Tensor y) {
    int size = x.numel();
    int n_threads = 3;
    int n_blocks = ceil(((float) n_threads + size - 1) / size);
    torch::Tensor out = torch::empty_like(x);

    vectAddKernel <<<n_blocks, n_threads>>> (x.data_ptr<int>(), y.data_ptr<int>(), out.data_ptr<int>(), size);

    return out;
}
"""

cpp_src = "torch::Tensor vect_add(torch::Tensor x, torch::Tensor y);"

module = load_inline(
        name="vect_add",
        cpp_sources=cpp_src,
        cuda_sources=cuda_src,
        functions=["vect_add"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
    )


x = torch.tensor([1, 2, 3, 4], dtype=torch.int, device="cuda")
y = torch.tensor([1, 2, 6, 4], dtype=torch.int, device="cuda")

res = module.vect_add(x, y)
print(res)