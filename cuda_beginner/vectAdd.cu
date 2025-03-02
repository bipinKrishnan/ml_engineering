#include <stdio.h>
#include <cuda.h>


#define ARRAY_LEN 5

__global__ void vectAdd(int *d_inpA, int *d_inpB, int *d_outC, int arrLen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < arrLen){
        d_outC[idx] = d_inpA[idx] + d_inpB[idx];
    }
}

int main() {
    
    // define the no of blocks and threads 
    // required for the kernel launch
    int nThreads = 5;
    int nBlocks = 1;

    // define the input and output vectors in host
    int h_vectA[ARRAY_LEN] = {1, 2, 3, 4, 5};
    int h_vectB[ARRAY_LEN] = {4, 5, 3, 3, 5};
    int h_vectC[ARRAY_LEN];
    

    /* we pass `arrSize` during cudaMemcpy,  so we could just
    define a pointer as `int *d_vectA` and `cudaMalloc` will create
    space in memory according to `arrSize` passed
    */
    // declare the pointers for input and output vectors in device
    int *d_vectA;
    int *d_vectB;
    int *d_vectC;

    // we need the array size for allocating memory in GPU
    int arrSize = sizeof(int) * ARRAY_LEN;

    // allocate the memory in GPU for inputs and outputs
    cudaMalloc((void**) &d_vectA, arrSize);
    cudaMalloc((void**) &d_vectB, arrSize);
    cudaMalloc((void**) &d_vectC, arrSize);

    // copy the data from host to device
    cudaMemcpy(d_vectA, h_vectA, arrSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vectB, h_vectB, arrSize, cudaMemcpyHostToDevice);

    // launch the kernel
    vectAdd <<<nBlocks, nThreads>>>(d_vectA, d_vectB, d_vectC, ARRAY_LEN);

    // copy the output data from device to host
    cudaMemcpy(h_vectC, d_vectC, arrSize, cudaMemcpyDeviceToHost);

    // free the memory allocated in device
    cudaFree(d_vectA);
    cudaFree(d_vectB);
    cudaFree(d_vectC);

    // print the vector addition result
    for(int i=0; i<ARRAY_LEN; i++) {
        printf("%d\n", h_vectC[i]);
    }

    return 0;

}
