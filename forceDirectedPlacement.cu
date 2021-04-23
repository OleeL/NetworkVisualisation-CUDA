#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include <fstream>
#include <chrono>
#include <algorithm>
#include <stdlib.h>
#include <math.h>
#include <cassert>
#include <iostream>

#include "forceDirectedPlacement.cuh"
#include "vector2.cuh"
#include "startup.cuh"
#include "graph.cuh"

#pragma once
#ifdef __INTELLISENSE__
void __syncthreads() {};
#endif

constexpr float MIN_NUM = 1.0f / (1 << 12);

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
	int warpSize;
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
inline __host__ __device__ __inline__
T maxD(T a, T b) {
	return (a > b) ? a : b;
}

// std::min doesn't handle floats
template <class T>
inline __host__ __device__ __inline__
T minD(T a, T b) {
	return (a < b) ? a : b;
}

// f_a(d) = d^2 / k
inline __device__ __inline__
float attractiveForce(float& dist, int& numberOfNodes, float& spread) {
	return dist * dist / spread / numberOfNodes;
}

// f_t = -k^2 / d
inline __device__ __inline__
float repulsiveForce(float& dist, int& numberOfNodes, float& spread) {
	return spread * spread / dist / numberOfNodes / 100;
}

/// <summary>
/// force directed placement algorithm
/// </summary>
/// <param name="nodes">Nodes you want to run the alg on</param>
/// <param name="displacement">the displacement of nodes</param>
/// <param name="adjacency matrix">stores the edges</param>
__global__
void forceUpdate(Vector2f* g_nodes, Vector2f* g_displacement, bool* g_adjMatrix)
{
	auto nNodes = c_parameters.numberOfNodes;
	auto id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= nNodes) return;
	const auto warpSize = c_parameters.warpSize;
	const auto maxElements = 48;

	extern __shared__ bool s_adjMatrix[];

	int i, ic, clamp, start = 0, index;

	auto tid = threadIdx.x;
	auto laneId = threadIdx.x % 32;
	auto spread = c_parameters.spread;
	auto inc = blockDim.x * gridDim.x;
	auto d = Vector2f();
	float dist, force;
	Vector2f node1, node2;
	start = 0;
	__syncthreads();
	node1 = g_nodes[id];

	// Repulsive force all nodes repel each other
	for (ic = 0; ic < nNodes; ++ic) {
		node2 = g_nodes[ic];
		node2 = node1 - node2;

		// using max to prevent dividing by 0
		dist = maxD(sqrtf(node2.x * node2.x + node2.y * node2.y), MIN_NUM);
		if (dist == MIN_NUM) continue;

		force = repulsiveForce(dist, nNodes, spread);
		d += node2 / dist * force;
	}

	while (start < nNodes)
	{
		clamp = minD(nNodes - start, maxElements);
		__syncthreads();
		for (i = 0; i < clamp; ++i) {
			s_adjMatrix[tid * maxElements + i] = g_adjMatrix[id + (nNodes * (i + start))];
		}

		// Attractive Force
		// Nodes that are connected pull one another
		//__syncthreads();
		for (i = 0; i < clamp; ++i) {
			if ((s_adjMatrix[tid * maxElements + i]) != 1 || start + i == id) continue;
			//printf("Matrix %d %d %d\n", id, ic, s_adjMatrix[tid * maxElements + ic]);
			node2 = g_nodes[start + i];
			node2 = node1 - node2;
			dist = maxD(sqrtf(node2.x * node2.x + node2.y * node2.y), MIN_NUM);
			if (dist == MIN_NUM) continue;
			force = attractiveForce(dist, nNodes, spread);
			d -= node2 / dist * force;
		}
		__syncthreads();

		g_displacement[id] = d;
		start += maxElements;
	}
}

/// <summary>
/// displacement update
/// </summary>
/// <param name="fdp">force directed placement context</param>
/// <returns></returns>
__global__
void displaceUpdate(Vector2f* g_nodes, Vector2f* displacement)
{
	auto id = blockIdx.x * blockDim.x + threadIdx.x;
	auto inc = blockDim.x * gridDim.x;
	auto scale = c_parameters.scale;
	auto windowWidth = c_parameters.windowWidth;
	auto windowHeight = c_parameters.windowHeight;
	auto nNodes = c_parameters.windowHeight;
	__syncthreads();
	auto node = g_nodes[id];
	auto displace = displacement[id];

	auto dist = sqrtf(displace.x * displace.x + displace.y * displace.y);
	node.x += (dist > scale) ? displace.x / dist * scale : displace.x,
		node.y += (dist > scale) ? displace.y / dist * scale : displace.y;
	node.x = minD(windowWidth * 0.5f, maxD(-windowWidth * 0.5f, node.x)),
		node.y = minD(windowHeight * 0.5f, maxD(-windowHeight * 0.5f, node.y));

	__syncthreads();
	g_nodes[id] = node;


}

/// <summary>
/// Prints all node data
/// </summary>
/// <param name="fdp">force directed placement args</param>
/// <param name="args">user args passed in</param>
void printData(ParamLaunch& args, ConstantDeviceParams& dv, const float& SPREADOFFSET)
{
	std::cout << "===================" << std::endl;
	std::cout << "Nodes:\t" << dv.numberOfNodes << std::endl;
	if (args.fileName == nullptr)
		std::cout << "Seed:\t" << args.seed << std::endl;
	else
		std::cout << "File:\t" << args.fileName << std::endl;
	std::cout << "Itera:\t" << args.iterations << std::endl;
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
inline void printProgressReport(int& i, int& iterations, int& progress, int& lastCaught)
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
void forceDirectedPlacement(ParamLaunch& args, Graph& graph)
{
	// Putting memory to GPU constant memory
	constexpr auto BLOCK_SIZE = 1024;
	const size_t warpSize = 49152;

	//auto SPREADOFFSET = maxD((1.0f - (MIN_NUM * (args.iterations / args.numNodes))), float(0.25));
	auto SPREADOFFSET = 0.2f;
	ConstantDeviceParams data;
	data.numberOfNodes = graph.numberOfNodes;
	data.iterations = args.iterations;
	data.scale = args.windowSize.x + args.windowSize.y;
	data.spread = SPREADOFFSET * sqrtf(static_cast<float>(args.windowSize.x) * args.windowSize.y / graph.numberOfNodes);
	data.windowWidth = args.windowSize.x;
	data.windowHeight = args.windowSize.y;
	data.warpSize = 32;

	printData(args, data, SPREADOFFSET);
	cudaMemcpyToSymbol(c_parameters, &data, sizeof(ConstantDeviceParams));
	gpuErrchk(cudaPeekAtLastError());

	Vector2f* d_nodes;
	Vector2f* d_displacement;
	bool* d_adjacencyMatrix;

	// Memory allocation. Takes the point of the pointer into cudaMalloc
	// Allocating the number of nodes for the number of floats * 2.
	cudaMalloc(&d_nodes, sizeof(Vector2f) * graph.numberOfNodes);
	cudaMalloc(&d_displacement, sizeof(Vector2f) * graph.numberOfNodes);
	cudaMalloc(&d_adjacencyMatrix, sizeof(bool) * graph.numberOfNodes * graph.numberOfNodes);
	gpuErrchk(cudaPeekAtLastError());

	cudaMemcpy(d_nodes, graph.nodes, sizeof(Vector2f) * graph.numberOfNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_displacement, graph.displacement, sizeof(Vector2f) * graph.numberOfNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_adjacencyMatrix, graph.adjacencyMatrix, sizeof(bool) * graph.numberOfNodes * graph.numberOfNodes, cudaMemcpyHostToDevice);
	gpuErrchk(cudaPeekAtLastError());

	auto block_size = std::min(BLOCK_SIZE, int(graph.numberOfNodes));
	block_size += (32 - block_size % 32) % 32;
	auto blockDim = dim3(block_size);
	auto gridDim = dim3(minD(32, (block_size - 1 + int(graph.numberOfNodes)) / block_size));

	using namespace std::chrono;
	auto start = steady_clock::now();
	auto lastCaught = 0;
	auto progress = 0;

	for (auto i = 0; i < data.iterations; ++i)
	{
		printProgressReport(i, data.iterations, progress, lastCaught);
		forceUpdate << <gridDim, blockDim, warpSize >> > (d_nodes, d_displacement, d_adjacencyMatrix);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		displaceUpdate << <gridDim, blockDim >> > (d_nodes, d_displacement);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	printTimeTaken(double(duration_cast<milliseconds>(steady_clock::now() - start).count()));

	cudaMemcpy(graph.nodes, d_nodes, sizeof(Vector2f) * graph.numberOfNodes, cudaMemcpyDeviceToHost);
	cudaFree(d_nodes);
	cudaFree(d_displacement);
	cudaFree(d_adjacencyMatrix);

}