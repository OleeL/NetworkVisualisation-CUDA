#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include "helper_math.cuh"

#include <fstream>
#include <chrono>
#include <algorithm>
#include <stdlib.h>
#include <math.h>
#include <cassert>
#include <iostream>

#include "forceDirectedPlacement.cuh"
#include "startup.cuh"
#include "graph.cuh"

#pragma once
#ifdef __INTELLISENSE__
void __syncthreads() {};
#endif

// Good practice to put on kernels
inline cudaError_t gpuErrchk(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}

struct ConstantDeviceParams {
	int numberOfNodes;
	int scale;  // This is known as k in reingold's
	float spread;
	int iterations;
	int windowWidth;
	int windowHeight;
};

__constant__
ConstantDeviceParams c_parameters;

// std::max doesn't handle floats
template <typename T>
inline __host__ __device__ __forceinline__
T maxD(T a, T b) {
	return (a > b) ? a : b;
}

// std::min doesn't handle floats
template <class T>
inline __host__ __device__ __forceinline__
T minD(T a, T b) {
	return (a < b) ? a : b;
}

// f_a(d) = d^2 / k
inline __device__ __forceinline__
float attractiveForce(float dist, int numberOfNodes, float spread) {
	return dist * dist / spread / numberOfNodes;
}

// f_t = -k^2 / d
inline __device__ __forceinline__
float repulsiveForce(float dist, int numberOfNodes, float spread) {
	return spread / dist / numberOfNodes / 100;
}

inline __device__ __forceinline__ int no_bank_conflict_index(int thread_id, int logical_index)
{
	return logical_index * 64 + thread_id;
}

inline __device__ __forceinline__ int getSharedMemoryIndex(
	unsigned int warpId,
	unsigned int laneId,
	unsigned int step,
	unsigned int start,
	unsigned int sharedMemCapacity,
	unsigned int threadsInBlock)
{
	return warpId * threadsInBlock + ((step * warpSize + start * warpSize) % sharedMemCapacity) + laneId;
}

/// <summary>
/// force directed placement algorithm
/// </summary>
/// <param name="nodes">Nodes you want to run the alg on</param>
/// <param name="displacement">the displacement of nodes</param>
/// <param name="adjacency matrix">stores the edges</param>
__global__
void forceUpdate(float2* g_nodes, float2* g_displacement, int* g_adjMatrix)
{
	// Basic necessities
	auto nNodes = c_parameters.numberOfNodes;
	auto threadsInBlock = blockDim.x;
	auto tid = threadIdx.x;
	auto bid = blockIdx.x;
	auto id = bid * threadsInBlock + tid;
	if (id >= nNodes) return;

	// Shared Memory Stuff
	const auto banks = 32;
	const auto sharedMemCapacity = 0xc000;
	const auto sharedMemCapacityInt = sharedMemCapacity / sizeof(int);
	const auto sharedMemCols = sharedMemCapacityInt / banks;
	auto activeThreads = fminf(threadsInBlock, nNodes);
	int colsPerThread = sharedMemCols / (activeThreads / 32) / 32;
	__shared__ int s_adjMatrix[sharedMemCols][32]; // Extra 1 for the padding to prevent bank conflicts
	auto laneId = tid % warpSize;
	auto warpId = tid / warpSize;
	auto warpWidth = static_cast<int>(ceilf(nNodes / static_cast<float>(warpSize)));

	// Math based
	auto scale = c_parameters.scale;
	auto windowWidth = c_parameters.windowWidth;
	auto windowHeight = c_parameters.windowHeight;
	auto spread = c_parameters.spread;
	auto sps = spread * spread;
	auto d = make_float2(0.0f,0.0f);
	float dist, force;
	float2 node1, node2;

	int i, ic = 0, clamp;

	node1 = g_nodes[id];
	// Repulsive force all nodes repel each other
	for (i = 0; i < nNodes; ++i) {
		node2 = g_nodes[i];
		node2 = node1 - node2;
		dist = fmaxf(length(node2), 0.001f);
		d += node2 / dist * repulsiveForce(dist, nNodes, sps);
	}

	__syncthreads();
	do 
	{
		clamp = fminf(nNodes - ic, colsPerThread);

		for (i = 0; i < clamp; ++i) {
			s_adjMatrix[colsPerThread*warpId+i][laneId] = g_adjMatrix[(ic + i) * nNodes + id];
		}

		__syncthreads();

		for (i = 0; i < clamp; ++i) {
			auto checkingNode = ic + i;
			if ((s_adjMatrix[colsPerThread*warpId+i][laneId]) != 1 || checkingNode == id) continue;
			node2 = g_nodes[checkingNode];
			node2 = node1 - node2;
			dist = fmaxf(length(node2), 0.001f);
			force = attractiveForce(dist, nNodes, spread);
			d -= node2 / dist * force;
		}
		ic += colsPerThread;
	} 	while (ic < nNodes);

	__syncthreads();

	node1.x += (dist > scale) ? d.x / dist * scale : d.x,
		node1.y += (dist > scale) ? d.y / dist * scale : d.y;
	node1.x = fminf(windowWidth * 0.5f, fmaxf(-windowWidth * 0.5f, node1.x)),
		node1.y = fminf(windowHeight * 0.5f, fmaxf(-windowHeight * 0.5f, node1.y));

	g_nodes[id] = node1;

}

/// <summary>
/// displacement update
/// </summary>
/// <param name="fdp">force directed placement context</param>
/// <returns></returns>
//__global__
//void displaceUpdate(float2* g_nodes, float2* displacement)
//{
//
//	auto nNodes = c_parameters.numberOfNodes;
//	auto id = blockIdx.x * blockDim.x + threadIdx.x;
//	if (id >= nNodes) return;
//
//	auto node1 = g_nodes[id];
//	auto d = displacement[id];
//}

/// <summary>
/// Prints all node data
/// </summary>
/// <param name="fdp">force directed placement args</param>
/// <param name="args">user args passed in</param>
void printData(ParamLaunch* args, ConstantDeviceParams dv, const float SPREADOFFSET)
{
	std::cout << "===================" << std::endl;
	std::cout << "Nodes:\t" << dv.numberOfNodes << std::endl;
	if (args->fileName == NULL)
		std::cout << "Seed:\t" << args->seed << std::endl;
	else
		std::cout << "File:\t" << args->fileName << std::endl;
	std::cout << "Itera:\t" << args->iterations << std::endl;
#if defined(DEBUG) || defined(_DEBUG)
	std::cout << "Offset: " << SPREADOFFSET << std::endl;
	std::cout << "Total spread:\t" << dv.spread << std::endl;
#endif
	std::cout << "Size:\t" << dv.windowWidth << "x" << dv.windowHeight << std::endl;
	std::cout << "===================" << std::endl;
}

/// <summary>
/// progress bar during execution
/// </summary>
/// <param name="i"></param>
/// <param name="iterations"></param>
/// <param name="progress"></param>
/// <param name="lastCaught"></param>
inline void printProgressReport(int i, int iterations, int progress, int lastCaught)
{
	progress = int(static_cast<float>(i) / static_cast<float>(iterations) * 100);
	if (progress != lastCaught && progress % 10 == 0) {
		std::cout << "Progress: " << progress << "%" << std::endl;
		lastCaught = progress;
	}
}

template <class T>
void printTimeTaken(T time)
{
	std::cout << "Time taken: " << time;
	time /= 1000;
	std::cout << "ms / " << time;
	time /= 60;
	std::cout << "s / " << time << " minutes" << std::endl;
}

/// <summary>
/// Force directed placement algorithm
/// </summary>
/// <param name="args">parameter launch args</param>
/// <param name="graph">graph args</param>
void forceDirectedPlacement(ParamLaunch* args, Graph* graph)
{
	// Putting memory to GPU constant memory
	constexpr auto BLOCK_SIZE = 384;

	//auto SPREADOFFSET = maxD((1.0f - (MIN_NUM * (args->iterations / args->numNodes))), float(0.25));
	auto SPREADOFFSET = 0.2f;
	ConstantDeviceParams data;
	data.numberOfNodes = graph->numberOfNodes;
	data.iterations = args->iterations;
	data.scale = args->windowSize.x + args->windowSize.y;
	data.spread = SPREADOFFSET * sqrtf(static_cast<float>(args->windowSize.x) * args->windowSize.y / graph->numberOfNodes);
	data.windowWidth = 600;
	data.windowHeight = 600;

	printData(args, data, SPREADOFFSET);

	cudaMemcpyToSymbol(c_parameters, &data, sizeof(ConstantDeviceParams));
	gpuErrchk(cudaPeekAtLastError());

	float2* d_nodes;
	float2* d_displacement;
	int* d_adjacencyMatrix;

	// Memory allocation. Takes the point of the pointer into cudaMalloc
	// Allocating the number of nodes for the number of floats * 2.
	cudaMalloc(&d_nodes, sizeof(float2) * graph->numberOfNodes);
	cudaMalloc(&d_displacement, sizeof(float2) * graph->numberOfNodes);
	cudaMalloc(&d_adjacencyMatrix, sizeof(int) * graph->numberOfNodes * graph->numberOfNodes);
	gpuErrchk(cudaPeekAtLastError());

	cudaMemcpy(d_nodes, graph->nodes, sizeof(float2) * graph->numberOfNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_displacement, graph->displacement, sizeof(float2) * graph->numberOfNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_adjacencyMatrix, graph->adjacencyMatrix, sizeof(int) * graph->numberOfNodes * graph->numberOfNodes, cudaMemcpyHostToDevice);
	gpuErrchk(cudaPeekAtLastError());

	auto block_size = std::min(BLOCK_SIZE, int(graph->numberOfNodes));
	block_size += ((32 - block_size % 32) % 32);
	auto blockDim = dim3(block_size);
	auto gridDim = dim3(minD(32, (block_size - 1 + int(graph->numberOfNodes)) / block_size));

	using namespace std::chrono;
	auto start = steady_clock::now();
	auto lastCaught = 0;
	auto progress = 0;

	for (auto i = 0; i < data.iterations; ++i)
	{
		printProgressReport(i, data.iterations, progress, lastCaught);
		forceUpdate<<<gridDim, blockDim>>>(d_nodes, d_displacement, d_adjacencyMatrix);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	printTimeTaken(double(duration_cast<milliseconds>(steady_clock::now() - start).count()));

	cudaMemcpy(graph->nodes, d_nodes, sizeof(float2) * graph->numberOfNodes, cudaMemcpyDeviceToHost);
	cudaFree(d_nodes);
	cudaFree(d_displacement);
	cudaFree(d_adjacencyMatrix);
	//cudaDeviceReset();

}