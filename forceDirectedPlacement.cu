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
#include "vector2.cuh"
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

constexpr int SPATIAL_GRID_DIM = 32;
constexpr int SPATIAL_GRID_CELLS = SPATIAL_GRID_DIM * SPATIAL_GRID_DIM;

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

inline __device__ __forceinline__ int getSpatialGridCell(float2 node)
{
	auto normalisedX = (node.x + c_parameters.windowWidth * 0.5f) / c_parameters.windowWidth;
	auto normalisedY = (node.y + c_parameters.windowHeight * 0.5f) / c_parameters.windowHeight;
	auto x = minD(SPATIAL_GRID_DIM - 1, maxD(0, static_cast<int>(normalisedX * SPATIAL_GRID_DIM)));
	auto y = minD(SPATIAL_GRID_DIM - 1, maxD(0, static_cast<int>(normalisedY * SPATIAL_GRID_DIM)));
	return y * SPATIAL_GRID_DIM + x;
}

__global__
void clearSpatialGrid(float* g_cellX, float* g_cellY, int* g_cellCount, int* g_cellHead)
{
	auto id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= SPATIAL_GRID_CELLS) return;

	g_cellX[id] = 0.0f;
	g_cellY[id] = 0.0f;
	g_cellCount[id] = 0;
	g_cellHead[id] = -1;
}

__global__
void buildSpatialGrid(float2* g_nodes, float* g_cellX, float* g_cellY, int* g_cellCount, int* g_cellHead, int* g_nextNode)
{
	auto id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= c_parameters.numberOfNodes) return;

	auto node = g_nodes[id];
	auto cell = getSpatialGridCell(node);
	g_nextNode[id] = atomicExch(&g_cellHead[cell], id);
	atomicAdd(&g_cellX[cell], node.x);
	atomicAdd(&g_cellY[cell], node.y);
	atomicAdd(&g_cellCount[cell], 1);
}

__global__
void clearDisplacement(float2* g_displacement)
{
	auto id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= c_parameters.numberOfNodes) return;
	g_displacement[id] = make_float2(0.0f, 0.0f);
}

__global__
void computeExactRepulsion(float2* g_nodes, float2* g_displacement)
{
	extern __shared__ float2 s_nodes[];

	auto nNodes = c_parameters.numberOfNodes;
	auto id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= nNodes) return;

	auto node1 = g_nodes[id];
	auto d = make_float2(0.0f, 0.0f);
	auto sps = c_parameters.spread * c_parameters.spread;

	for (auto tileStart = 0; tileStart < nNodes; tileStart += blockDim.x) {
		auto tileIndex = tileStart + threadIdx.x;
		if (tileIndex < nNodes) {
			s_nodes[threadIdx.x] = g_nodes[tileIndex];
		}
		__syncthreads();

		auto tileCount = minD(static_cast<int>(blockDim.x), nNodes - tileStart);
		for (auto i = 0; i < tileCount; ++i) {
			auto otherId = tileStart + i;
			if (otherId == id) continue;

			auto node2 = node1 - s_nodes[i];
			auto dist = fmaxf(length(node2), 0.001f);
			d += node2 / dist * repulsiveForce(dist, nNodes, sps);
		}
		__syncthreads();
	}

	g_displacement[id] += d;
}

__global__
void computeGridRepulsion(float2* g_nodes, float2* g_displacement, float* g_cellX, float* g_cellY, int* g_cellCount)
{
	auto nNodes = c_parameters.numberOfNodes;
	auto id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= nNodes) return;

	auto node1 = g_nodes[id];
	auto d = make_float2(0.0f, 0.0f);
	auto sps = c_parameters.spread * c_parameters.spread;

	for (auto i = 0; i < SPATIAL_GRID_CELLS; ++i) {
		auto cellCount = g_cellCount[i];
		if (cellCount == 0) continue;

		auto node2 = make_float2(g_cellX[i] / cellCount, g_cellY[i] / cellCount);
		node2 = node1 - node2;
		auto dist = fmaxf(length(node2), 0.001f);
		d += node2 / dist * repulsiveForce(dist, nNodes, sps) * cellCount;
	}

	g_displacement[id] += d;
}

__global__
void computeHybridRepulsion(float2* g_nodes, float2* g_displacement, float* g_cellX, float* g_cellY, int* g_cellCount, int* g_cellHead, int* g_nextNode)
{
	auto nNodes = c_parameters.numberOfNodes;
	auto id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= nNodes) return;

	auto node1 = g_nodes[id];
	auto d = make_float2(0.0f, 0.0f);
	auto sps = c_parameters.spread * c_parameters.spread;
	auto currentCell = getSpatialGridCell(node1);
	auto currentX = currentCell % SPATIAL_GRID_DIM;
	auto currentY = currentCell / SPATIAL_GRID_DIM;

	for (auto cell = 0; cell < SPATIAL_GRID_CELLS; ++cell) {
		auto cellCount = g_cellCount[cell];
		if (cellCount == 0) continue;

		auto cellX = cell % SPATIAL_GRID_DIM;
		auto cellY = cell / SPATIAL_GRID_DIM;
		auto nearby = abs(cellX - currentX) <= 1 && abs(cellY - currentY) <= 1;

		if (nearby) {
			for (auto otherId = g_cellHead[cell]; otherId != -1; otherId = g_nextNode[otherId]) {
				if (otherId == id) continue;
				auto node2 = node1 - g_nodes[otherId];
				auto dist = fmaxf(length(node2), 0.001f);
				d += node2 / dist * repulsiveForce(dist, nNodes, sps);
			}
		} else {
			auto node2 = make_float2(g_cellX[cell] / cellCount, g_cellY[cell] / cellCount);
			node2 = node1 - node2;
			auto dist = fmaxf(length(node2), 0.001f);
			d += node2 / dist * repulsiveForce(dist, nNodes, sps) * cellCount;
		}
	}

	g_displacement[id] += d;
}

__global__
void computeAttraction(float2* g_nodes, float2* g_displacement, int2* g_edges, int numberOfEdges)
{
	auto id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= numberOfEdges) return;

	auto edge = g_edges[id];
	auto source = edge.x;
	auto target = edge.y;
	auto nodeDelta = g_nodes[source] - g_nodes[target];
	auto dist = fmaxf(length(nodeDelta), 0.001f);
	auto force = attractiveForce(dist, c_parameters.numberOfNodes, c_parameters.spread);
	auto delta = nodeDelta / dist * force;

	atomicAdd(&g_displacement[source].x, -delta.x);
	atomicAdd(&g_displacement[source].y, -delta.y);
	atomicAdd(&g_displacement[target].x, delta.x);
	atomicAdd(&g_displacement[target].y, delta.y);
}

__global__
void updatePositions(float2* g_nodes, float2* g_displacement, float temperature)
{
	auto id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= c_parameters.numberOfNodes) return;

	auto node = g_nodes[id];
	auto d = g_displacement[id];
	auto displacementLength = length(d);
	if (displacementLength > 0.0f) {
		auto step = fminf(displacementLength, temperature);
		node += d / displacementLength * step;
	}

	node.x = fminf(c_parameters.windowWidth * 0.5f, fmaxf(-c_parameters.windowWidth * 0.5f, node.x));
	node.y = fminf(c_parameters.windowHeight * 0.5f, fmaxf(-c_parameters.windowHeight * 0.5f, node.y));
	g_nodes[id] = node;
}

/// <summary>
/// displacement update
/// </summary>
/// <param name="fdp">force directed placement context</param>
/// <returns></returns>
__global__
void displaceUpdate(float2* g_nodes, float2* displacement)
{

	auto nNodes = c_parameters.numberOfNodes;
	auto id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= nNodes) return;

	auto node1 = g_nodes[id];
	auto d = displacement[id];
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
	std::cout << "Mode:\t" << (args.repulsionMode == REPULSION_GRID ? "grid" : args.repulsionMode == REPULSION_HYBRID ? "hybrid" : "exact") << std::endl;
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

void printNodeCoordinates(Graph& graph)
{
	if (graph.numberOfNodes == 0) return;

	auto minX = graph.nodes[0].x;
	auto maxX = graph.nodes[0].x;
	auto minY = graph.nodes[0].y;
	auto maxY = graph.nodes[0].y;
	for (unsigned int i = 1; i < graph.numberOfNodes; ++i) {
		minX = std::min(minX, graph.nodes[i].x);
		maxX = std::max(maxX, graph.nodes[i].x);
		minY = std::min(minY, graph.nodes[i].y);
		maxY = std::max(maxY, graph.nodes[i].y);
	}

	std::cout << "===================" << std::endl;
	std::cout << "Node coordinate bounds:" << std::endl;
	std::cout << "X:\t" << minX << " to " << maxX << std::endl;
	std::cout << "Y:\t" << minY << " to " << maxY << std::endl;
	std::cout << "Node coordinates:" << std::endl;
	for (unsigned int i = 0; i < graph.numberOfNodes; ++i) {
		std::cout << i << ":\t" << graph.nodes[i].x << ", " << graph.nodes[i].y << std::endl;
	}
	std::cout << "===================" << std::endl;
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
	constexpr auto BLOCK_SIZE = 384;

	//auto SPREADOFFSET = maxD((1.0f - (MIN_NUM * (args.iterations / args.numNodes))), float(0.25));
	auto SPREADOFFSET = 0.2f;
	ConstantDeviceParams data;
	data.numberOfNodes = graph.numberOfNodes;
	data.iterations = args.iterations;
	data.scale = args.windowSize.x + args.windowSize.y;
	data.spread = SPREADOFFSET * sqrtf(static_cast<float>(args.windowSize.x) * args.windowSize.y / graph.numberOfNodes);
	data.windowWidth = args.windowSize.x;
	data.windowHeight = args.windowSize.y;

	printData(args, data, SPREADOFFSET);

	cudaMemcpyToSymbol(c_parameters, &data, sizeof(ConstantDeviceParams));
	gpuErrchk(cudaPeekAtLastError());

	float2* d_nodes;
	float2* d_displacement;
	int2* d_edges;
	float* d_cellX;
	float* d_cellY;
	int* d_cellCount;
	int* d_cellHead;
	int* d_nextNode;

	// Memory allocation. Takes the point of the pointer into cudaMalloc
	// Allocating the number of nodes for the number of floats * 2.
	cudaMalloc(&d_nodes, sizeof(float2) * graph.numberOfNodes);
	cudaMalloc(&d_displacement, sizeof(float2) * graph.numberOfNodes);
	cudaMalloc(&d_edges, sizeof(int2) * graph.numberOfEdges);
	cudaMalloc(&d_cellX, sizeof(float) * SPATIAL_GRID_CELLS);
	cudaMalloc(&d_cellY, sizeof(float) * SPATIAL_GRID_CELLS);
	cudaMalloc(&d_cellCount, sizeof(int) * SPATIAL_GRID_CELLS);
	cudaMalloc(&d_cellHead, sizeof(int) * SPATIAL_GRID_CELLS);
	cudaMalloc(&d_nextNode, sizeof(int) * graph.numberOfNodes);
	gpuErrchk(cudaPeekAtLastError());

	cudaMemcpy(d_nodes, graph.nodes, sizeof(float2) * graph.numberOfNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_edges, graph.distinctEdges, sizeof(int2) * graph.numberOfEdges, cudaMemcpyHostToDevice);
	gpuErrchk(cudaPeekAtLastError());

	auto block_size = std::min(BLOCK_SIZE, int(graph.numberOfNodes));
	block_size += ((32 - block_size % 32) % 32);
	auto blockDim = dim3(block_size);
	auto gridDim = dim3((int(graph.numberOfNodes) + block_size - 1) / block_size);
	auto gridClearBlockDim = dim3(256);
	auto gridClearDim = dim3((SPATIAL_GRID_CELLS + gridClearBlockDim.x - 1) / gridClearBlockDim.x);
	auto edgeBlockDim = dim3(256);
	auto edgeGridDim = dim3((int(graph.numberOfEdges) + edgeBlockDim.x - 1) / edgeBlockDim.x);
	auto exactRepulsionSharedMemory = sizeof(float2) * blockDim.x;

	using namespace std::chrono;
	auto start = steady_clock::now();
	auto lastCaught = 0;
	auto progress = 0;
	auto initialTemperature = static_cast<float>(std::min(args.windowSize.x, args.windowSize.y)) * 0.1f;

	for (auto i = 0; i < data.iterations; ++i)
	{
		printProgressReport(i, data.iterations, progress, lastCaught);
		auto coolingProgress = static_cast<float>(i) / static_cast<float>(data.iterations);
		auto temperature = fmaxf(initialTemperature * (1.0f - coolingProgress), 0.01f);
		clearDisplacement<<<gridDim, blockDim>>>(d_displacement);
		gpuErrchk(cudaPeekAtLastError());

		computeExactRepulsion<<<gridDim, blockDim, exactRepulsionSharedMemory>>>(d_nodes, d_displacement);
		gpuErrchk(cudaPeekAtLastError());

		computeAttraction<<<edgeGridDim, edgeBlockDim>>>(d_nodes, d_displacement, d_edges, graph.numberOfEdges);
		gpuErrchk(cudaPeekAtLastError());
		updatePositions<<<gridDim, blockDim>>>(d_nodes, d_displacement, temperature);
		gpuErrchk(cudaPeekAtLastError());
	}

	gpuErrchk(cudaDeviceSynchronize());
	printTimeTaken(double(duration_cast<milliseconds>(steady_clock::now() - start).count()));

	cudaMemcpy(graph.nodes, d_nodes, sizeof(float2) * graph.numberOfNodes, cudaMemcpyDeviceToHost);
	// printNodeCoordinates(graph);
	cudaFree(d_nodes);
	cudaFree(d_displacement);
	cudaFree(d_edges);
	cudaFree(d_cellX);
	cudaFree(d_cellY);
	cudaFree(d_cellCount);
	cudaFree(d_cellHead);
	cudaFree(d_nextNode);
	//cudaDeviceReset();

}
