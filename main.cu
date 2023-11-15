#include <cuda_runtime.h>
#include <iostream>

template<class T>
__device__ T plus_scan(T *x)
{
	unsigned int i = threadIdx.x;
	unsigned int n = blockDim.x;
	for(unsigned int offset=1; offset<n; offset *= 2)
	{
		T t;
		if(i>=offset) t = x[i-offset];
		__syncthreads();
		if(i>=offset) x[i] = t + x[i];
		__syncthreads();
	}
	return x[i];
}

__device__ void partition_by_bit(unsigned int *values, unsigned int bit)
{
	unsigned int i = threadIdx.x;
	unsigned int size = blockDim.x;
	unsigned int x_i = values[i];
	unsigned int p_i = (x_i >> bit) & 1;
	values[i] = p_i;
	__syncthreads();
	// Compute number of T bits up to and including p_i.
	// Record the total number of F bits as well.
	unsigned int T_before = plus_scan(values);
	unsigned int T_total = values[size-1];
	unsigned int F_total = size - T_total;
	__syncthreads();
	// Write every x_i to its proper place
	if( p_i )
	values[T_before-1 + F_total] = x_i;
	else
	values[i - T_before] = x_i;
}

__global__ void radix_sort(unsigned int *values)
{
	for(int bit=0; bit<32; ++bit)
	{
	partition_by_bit(values, bit);
	__syncthreads();
	}
}

int main() {
    // Number of elements to sort
    int arraySize;
	printf("Provide the size of the array: " );
    scanf("%d",&arraySize);
    unsigned int *h_values = new unsigned int[arraySize]; // host array
    unsigned int *d_values; // device array

    // Initialize host array with data
    for(int i = 0; i < arraySize; ++i) {
		printf("Provide the values for array's position %d: \n",i);
		scanf("%u",&h_values[i]);
    }

    // Allocate memory on the device
    cudaMalloc((void**)&d_values, arraySize * sizeof(unsigned int));

    // Copy data from host to device
    cudaMemcpy(d_values, h_values, arraySize * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Launch the kernel
	int threadsPerBlock = 256;
    int nblocks = (arraySize + threadsPerBlock - 1) / threadsPerBlock; // This ensures rounding up if n is not a multiple of threadsPerBlock
    radix_sort<<<nblocks, arraySize>>>(d_values);

    // Copy sorted array back to host
    cudaMemcpy(h_values, d_values, arraySize * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Display sorted data
	printf("\nSorted Array: ");
    for (int i = 0; i < arraySize; ++i) {
        printf("%d ", h_values[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_values);

    // Free host memory
    delete[] h_values;

    return 0;
}