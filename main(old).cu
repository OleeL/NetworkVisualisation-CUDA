//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <stdio.h>
//#include <cstdlib>
//
//#define gpuErrchk(ans) gpuAssert((ans), __FILE__, __LINE__);
//inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
//{
//	if (code == cudaSuccess) return;
//	fprintf(stderr, "Error: %s %s Ln: %d\n", cudaGetErrorString(code), file, line);
//	if (abort) exit(code);
//}
//
//__global__ // Answer gets put into y
//void add(int n, float* x, float* y)
//{
//	auto index = threadIdx.x;
//	auto stride = blockDim.x;
//	for (auto i = index; i < n; i += stride) {
//		y[i] = x[i] + y[i];
//		printf("- %d\n", i);
//	}
//}
//
//int main(void)
//{
//	const auto N = 1 << 20;
//	float* x, float* y;
//
//	gpuErrchk(cudaMallocManaged(&x, N * sizeof(float)));
//	gpuErrchk(cudaMallocManaged(&y, N * sizeof(float)));
//
//	// initialize x and y arrays of the host
//	for (auto i = 0; i < N; i++)
//	{
//		x[i] = 1.0f;
//		y[i] = 2.0f;
//	}
//
//	// Add kernal on 1M elements on the CPU
//	printf("%d\n", N);
//	add << <1, 256 >> > (N, x, y);
//	gpuErrchk(cudaPeekAtLastError());
//
//	// Wait for GPU to finish before accessing on host
//	cudaDeviceSynchronize();
//
//	// Freeing memory
//	cudaFree(x);
//	cudaFree(y);
//
//	return 0;
//}