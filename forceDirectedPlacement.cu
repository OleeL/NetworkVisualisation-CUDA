
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
	int numberOfEdges;
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
__host__ __device__ __inline__
T maxD(T a, T b) {
	return (a > b) ? a : b;
}

// std::min doesn't handle floats
template <class T>
__host__ __device__ __inline__
T minD(T a, T b) {
	return (a < b) ? a : b;
}

// f_a(d) = d^2 / k
__device__ __inline__
float attractiveForce(float& dist, int& numberOfNodes, float& spread) {
	return dist * dist / spread / numberOfNodes;
}

// f_t = -k^2 / d
__device__ __inline__
float repulsiveForce(float& dist, int& numberOfNodes, float& spread) {
	return spread * spread / dist / numberOfNodes / 100;
}

/// <summary>
/// force directed placement algorithm
/// </summary>
/// <param name="nodes">Nodes you want to run the alg on</param>
/// <param name="edges">length of spring</param>
/// <param name="connectionIndex">a list of connectionIndex</param>
__global__
void forceUpdate(Vector2f* nodes, Vector2f* displacement, Vector2i* edges, int* connectionIndex)
{
	auto id = blockIdx.x * blockDim.x + threadIdx.x;
	auto nNodes = c_parameters.numberOfNodes;
	if (id >= nNodes) return;

	extern __shared__ int s[];
	auto spread = c_parameters.spread;
	auto inc = blockDim.x * gridDim.x;
	auto d = Vector2f();
	float dist, force;
	Vector2f node1, node2;

	while (id < nNodes)
	{
		__syncthreads();
		node1 = nodes[id];

		// Repulsive force all nodes repel each other
		for (auto ic = 0; ic < nNodes; ++ic) {
			if (ic == id) continue;
			node2 = nodes[ic];
			node2 = node1 - node2;

			// using max to prevent dividing by 0
			dist = maxD(sqrtf(node2.x * node2.x + node2.y * node2.y), MIN_NUM);
			if (dist == MIN_NUM) continue;

			force = repulsiveForce(dist, nNodes, spread);
			d.x += node2.x / dist * force;
			d.y += node2.y / dist * force;
		}

		auto start = (id > 0) ? connectionIndex[id - 1] : 0;

		// Attractive Force
		// Nodes that are connected pull one another
		//__syncthreads();
		for (auto ic = start; ic < connectionIndex[id]; ++ic) {
			node2 = nodes[edges[ic].y];
			node2 = node1 - node2;
			dist = maxD(sqrtf(node2.x * node2.x + node2.y * node2.y), MIN_NUM);
			if (dist == MIN_NUM) continue;

			force = attractiveForce(dist, nNodes, spread);
			d.x -= node2.x / dist * force;
			d.y -= node2.y / dist * force;
		}
		__syncthreads();

		displacement[id] = d;
		id += inc;
	}
}

/// <summary>
/// displacement update
/// </summary>
/// <param name="fdp">force directed placement context</param>
/// <returns></returns>
__global__
void displaceUpdate(Vector2f* nodes, Vector2f* displacement)
{
	auto id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= c_parameters.numberOfNodes) return;
	extern __shared__ int s[];
	auto scale = c_parameters.scale;
	auto windowWidth = c_parameters.windowWidth;
	auto windowHeight = c_parameters.windowHeight;

	__syncthreads();
	auto node = nodes[id];
	auto displace = displacement[id];

	auto dist = sqrtf(displace.x * displace.x + displace.y * displace.y);
	node.x += (dist > scale) ? displace.x / dist * scale : displace.x,
	node.y += (dist > scale) ? displace.y / dist * scale : displace.y;
	node.x = minD(windowWidth * 0.5f, maxD(-windowWidth * 0.5f, node.x)),
	node.y = minD(windowHeight * 0.5f, maxD(-windowHeight * 0.5f, node.y));

	__syncthreads();
	nodes[id] = node;
}

/// <summary>
/// Prints all node data
/// </summary>
/// <param name="fdp">force directed placement args</param>
/// <param name="args">user args passed in</param>
void printData(ParamLaunch& args, ConstantDeviceParams& dv)
{
	std::cout << "===================" << std::endl;
	std::cout << "Nodes:\t" << dv.numberOfNodes << std::endl;
	if (args.fileName == nullptr)
		std::cout << "Seed:\t" << args.seed << std::endl;
	else
		std::cout << "File:\t" << args.fileName << std::endl;
	std::cout << "Itera:\t" << args.iterations << std::endl;
	std::cout << "Spread:\t" << dv.spread << std::endl;
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

/// <summary>
/// Force directed placement algorithm
/// </summary>
/// <param name="args">parameter launch args</param>
/// <param name="graph">graph args</param>
void forceDirectedPlacement(ParamLaunch& args, Graph& graph)
{
	// Putting memory to GPU constant memory
	constexpr auto BLOCK_SIZE = 1024;
	auto SPREADOFFSET = maxD((1.0f - (MIN_NUM * (args.iterations / args.numNodes))), float(0.25));
	ConstantDeviceParams data;
	data.numberOfNodes = graph.numberOfNodes;
	data.numberOfEdges = graph.numberOfEdges;
	data.iterations = args.iterations;
	data.scale = args.windowSize.x + args.windowSize.y;
	data.spread = SPREADOFFSET * sqrtf(static_cast<float>(args.windowSize.x) * args.windowSize.y / graph.numberOfNodes);
	data.windowWidth = args.windowSize.x;
	data.windowHeight = args.windowSize.y;

	std::cout << SPREADOFFSET << std::endl;

	printData(args, data);
	cudaMemcpyToSymbol(c_parameters, &data, sizeof(ConstantDeviceParams));
	gpuErrchk(cudaPeekAtLastError());

	Vector2f* d_nodes;
	Vector2f* d_displacement;
	Vector2i* d_edges;
	int* d_connectionIndex;

	// Memory allocation. Takes the point of the pointer into cudaMalloc
	// Allocating the number of nodes for the number of floats * 2.
	cudaMalloc(&d_connectionIndex, sizeof(int) * graph.numberOfNodes);
	cudaMalloc(&d_nodes, sizeof(Vector2f) * graph.numberOfNodes);
	cudaMalloc(&d_displacement, sizeof(Vector2f) * graph.numberOfNodes);
	cudaMalloc(&d_edges, sizeof(Vector2i) * graph.numberOfEdges * 2);
	gpuErrchk(cudaPeekAtLastError());

	cudaMemcpy(d_connectionIndex, graph.connectionIndex, sizeof(int) * graph.numberOfNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_nodes, graph.nodes, sizeof(Vector2f) * graph.numberOfNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_displacement, graph.displacement, sizeof(Vector2f) * graph.numberOfNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_edges, graph.edges, sizeof(Vector2i) * graph.numberOfEdges * 2, cudaMemcpyHostToDevice);
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
		forceUpdate<<<gridDim, blockDim>>>(d_nodes, d_displacement, d_edges, d_connectionIndex);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		displaceUpdate<<<gridDim, blockDim>>>(d_nodes, d_displacement);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	std::cout << "Time taken: " <<
		duration_cast<milliseconds>(steady_clock::now() - start).count()
		<< "ms" <<
		std::endl;

	cudaMemcpy(graph.nodes, d_nodes, sizeof(Vector2f) * graph.numberOfNodes, cudaMemcpyDeviceToHost);
	cudaFree(d_nodes);
	cudaFree(d_displacement);
	cudaFree(d_edges);
	cudaFree(d_connectionIndex);

	for (unsigned int i = 0; i < graph.numberOfNodes; ++i)
	{
		graph.nodes[i].x += (args.windowSize.x * 0.5f);
		graph.nodes[i].y += (args.windowSize.y * 0.5f);
	}

}