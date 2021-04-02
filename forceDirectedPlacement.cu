#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <iostream>

#include "forceDirectedPlacement.cuh"
#include "vector2.cuh"
#include "startup.cuh"
#include "graph.cuh"
#include <cassert>

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
	int wWidth;
	int wHeight;
	int numberOfNodes;
	int numberOfEdges;
	float scale;  // This is known as k in reingold's
	float spread;
	int iterations;
};

__constant__
ConstantDeviceParams c_parameters;

// std::max doesn't handle floats
__device__ __forceinline__
float maxf(float a, float b) {
	return (a > b) ? a : b;
}

// std::min doesn't handle floats
__device__ __forceinline__
float minf(float a, float b) {
	return (a < b) ? a : b;
}

// f_a(d) = d^2 / k
__device__ __forceinline__
float attractiveForce(float dist, int numberOfNodes, float spread) {
	return dist * dist / spread / static_cast<float>(numberOfNodes);
}

// f_t = -k^2 / d
__device__ __forceinline__
float repulsiveForce(float dist, int numberOfNodes, float spread) {
	return spread * spread / dist / static_cast<float>(numberOfNodes) / 100.0f;
}

/// <summary>
/// force directed placement algorithm
/// </summary>
/// <param name="nodes">Nodes you want to run the alg on</param>
/// <param name="edges">length of spring</param>
/// <param name="connectionIndex">a list of connectionIndex</param>
__global__ __inline__
void forceUpdate(Node* nodes, Vector2i* edges, int* connectionIndex)
{
	auto id = blockIdx.x * blockDim.x + threadIdx.x;
	auto inc = blockDim.x * gridDim.x;
	auto nNodes = c_parameters.numberOfNodes;
	auto nEdges = c_parameters.numberOfEdges;

	if (id >= c_parameters.numberOfNodes) return;
	float d_x, d_y;
	float2 node1, node2 = make_float2(0,0);
	d_x = 0;
	d_y = 0;
	node1.x = nodes[id].x, node1.y = nodes[id].y;

	// Repulsive force all nodes repel each other
	for (auto ic = 0; ic < nNodes; ++ic) {
		node2.x = nodes[ic].x, node2.y = nodes[ic].y;
		auto cmpNode = nodes[ic];
		auto dx = node1.x - node2.x;
		auto dy = node1.y - node2.y;
		auto dist = maxf(sqrtf(dx * dx + dy * dy), 0.001f);
		auto force = repulsiveForce(dist, nNodes, c_parameters.spread);
		d_x += dx / dist * force;
		d_y += dy / dist * force;
	}

	int start = (id > 0) ? connectionIndex[id - 1] : 0;

	// Attractive Force
	// Nodes that are connected pull one another
	for (auto ic = start; ic < connectionIndex[id]; ++ic) {
		node2.x = nodes[edges[ic].y].x, node2.y = nodes[edges[ic].y].y;
		auto dx = node1.x - node2.x;
		auto dy = node1.y - node2.y;
		auto dist = maxf(sqrtf(dx * dx + dy * dy), 0.001f);
		auto af = attractiveForce(dist, int(nNodes), c_parameters.spread);
		d_x -= dx / dist * af;
		d_y -= dy / dist * af;
	}
	nodes[id].dx = d_x;
	nodes[id].dy = d_y;
	id += inc;
}

/// <summary>
/// displacement update
/// </summary>
/// <param name="fdp">force directed placement context</param>
/// <returns></returns>
__global__ __inline__
void displaceUpdate(Node* nodes)
{
	auto id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= c_parameters.numberOfNodes) return;
	auto scale = c_parameters.scale;
	auto wWidth = c_parameters.wWidth;
	auto wHeight = c_parameters.wHeight;
	auto& nodeRef = nodes[id];
	auto node = nodeRef;

	auto dist = sqrtf(nodeRef.dx * nodeRef.dx + nodeRef.dy * node.dy);
	node.x += (dist > scale) ? node.dx / dist * scale : node.dx;
	node.y += (dist > scale) ? node.dy / dist * scale : node.dy;
	node.x = minf(wWidth / 2.0f, maxf(-wWidth / 2.0f, node.x));
	node.y = minf(wHeight / 2.0f, maxf(-wHeight / 2.0f, node.y));
	nodeRef.x = node.x;
	nodeRef.y = node.y;
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
	std::cout << "Size:\t" << dv.wWidth << "x" << dv.wHeight << std::endl;
	std::cout << "===================" << std::endl;
}

inline void printProgressReport(int& i, int& iterations, int& progress, int& lastCaught)
{
	progress = static_cast<float>(i) / static_cast<float>(iterations) * 100;
	if (progress != lastCaught && progress % 10 == 0) {
		std::cout << "Progress: " << progress << "%" << std::endl;
		lastCaught = progress;
	}
}

/// <summary>
/// Force directed placement algorithm
/// </summary>
/// <param name="fdp"></param>
/// <param name="args"></param>
void forceDirectedPlacement(ParamLaunch& args, Graph& graph)
{
	// Putting memory to GPU constant memory
	const auto SPREADOFFSET = sqrt(args.numNodes / std::min(args.iterations, args.numNodes)); // Known as the C value
	const auto BLOCK_SIZE = 1024;
	ConstantDeviceParams data;
	data.numberOfNodes = graph.numberOfNodes;
	data.numberOfEdges = graph.numberOfEdges;
	data.iterations = args.iterations;
	data.scale = args.wWidth + args.wHeight;
	data.spread = SPREADOFFSET * sqrtf(static_cast<float>(args.wWidth) * static_cast<float>(args.wHeight) / graph.numberOfNodes); 
	data.wWidth = args.wWidth;
	data.wHeight = args.wHeight;

	printData(args, data);
	cudaMemcpyToSymbol(c_parameters, &data, sizeof(ConstantDeviceParams));
	gpuErrchk(cudaPeekAtLastError());

	Node* d_nodes;
	Vector2i* d_edges;
	int* d_connectionIndex;

	// Memory allocation. Takes the point of the pointer into cudaMalloc
	// Allocating the number of nodes for the number of floats * 2.
	cudaMalloc(&d_connectionIndex, sizeof(int) * graph.numberOfNodes);
	cudaMalloc(&d_nodes, sizeof(Node) * graph.numberOfNodes);
	cudaMalloc(&d_edges, sizeof(Vector2i) * graph.numberOfEdges * 2);
	gpuErrchk(cudaPeekAtLastError());

	cudaMemcpy(d_connectionIndex, graph.connectionIndex, sizeof(int) * graph.numberOfNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_nodes, graph.nodes, sizeof(Node) * graph.numberOfNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_edges, graph.edges, sizeof(Vector2i) * graph.numberOfEdges * 2, cudaMemcpyHostToDevice);
	gpuErrchk(cudaPeekAtLastError());

	auto block_size = min(BLOCK_SIZE, graph.numberOfNodes);
	block_size += (32 - block_size % 32) % 32;
	auto blockDim = dim3(block_size);
	auto gridDim = dim3(min(32, (block_size - 1 + graph.numberOfNodes) / block_size));

	using namespace std::chrono;
	auto start = steady_clock::now();
	auto lastCaught = 0;
	auto progress = 0;

	for (auto i = 0; i < data.iterations; ++i)
	{
		printProgressReport(i, data.iterations, progress, lastCaught);
		forceUpdate<<<gridDim, blockDim>>>(d_nodes, d_edges, d_connectionIndex);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		displaceUpdate<<<gridDim, blockDim>>>(d_nodes);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	std::cout << "Time taken: " <<
		duration_cast<milliseconds>(steady_clock::now() - start).count()
		<< "ms" <<
		std::endl;

	cudaMemcpy(graph.nodes, d_nodes, sizeof(Node) * graph.numberOfNodes, cudaMemcpyDeviceToHost);
	cudaMemcpy(graph.edges, d_edges, sizeof(Vector2i) * graph.numberOfEdges * 2, cudaMemcpyDeviceToHost);
	cudaFree(d_nodes);
	cudaFree(d_edges);
	cudaFree(d_connectionIndex);

	for (auto i = 0; i < graph.numberOfNodes; ++i)
	{
		graph.nodes[i].x += (args.wWidth / 2);
		graph.nodes[i].y += (args.wHeight / 2);
	}

}