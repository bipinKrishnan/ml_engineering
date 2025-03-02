// GPU kernel
__global__ void vectAddKernel(int *a, int *b, int *c, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < len) {
        c[idx] = a[idx] + b[idx];
    }
}

// define the C++ function declared in `cpp_src`
torch::Tensor vect_add(torch::Tensor x, torch::Tensor y) {
    // get the number of elements in the tensor
    int size = x.numel();

    // no. of threads & blocks for launching the kernel 
    int n_threads = 5;
    int n_blocks = 1;

    // create an empty tensor to store the results of
    // vector addition
    torch::Tensor out = torch::empty_like(x);

    // launch the vector addition kernel
    // pass the pointer to x, y and out along with the size
    vectAddKernel <<<n_blocks, n_threads>>> (x.data_ptr<int>(), y.data_ptr<int>(), out.data_ptr<int>(), size);

    // return the result
    return out;
}
