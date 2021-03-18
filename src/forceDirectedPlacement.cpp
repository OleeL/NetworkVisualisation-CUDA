#include <iostream>
#include "forceDirectedPlacement.hpp"
#include "vector2.hpp"
#include "startup.hpp"
#include <chrono>

// std::max doesn't handle floats
inline float max(float a, float b) {
	return (a > b) ? a : b;
}

// std::min doesn't handle floats
inline float min(float a, float b) {
	return (a < b) ? a : b;
}

// f_a(d) = d^2 / k
inline float attractiveForce(float dist, int numberOfNodes, float spread) {
	return dist * dist / spread / ((float)numberOfNodes);
}

// f_t = -k^2 / d
inline float repulsiveForce(float dist, int numberOfNodes, float spread) {
	return spread * spread / dist / ((float) numberOfNodes) / 100.0f;
}


/// <summary>
/// force directed placement algorithm
/// </summary>
/// <param name="nodes">Nodes you want to run the alg on</param>
/// <param name="damping">rate at which spring goes back to normal</param>
/// <param name="springLength">length of spring</param>
/// <param name="maxIterations">number of iterations you want
/// the alg to run. Higher = More accuracy</param>
inline void update(FdpContext& fdp)
{
	auto numberOfNodes = fdp.nodes.size();
	auto& nodes = fdp.nodes;
	for (auto& node : nodes)
	{
		// Repulsive Force
		for (auto& cNode : nodes) {
			if (node->id == cNode->id) continue;
			Vector2 d = *node->position - *cNode->position;
			auto dist = max(sqrtf(d.x * d.x + d.y * d.y), 0.001f);
			auto rf = repulsiveForce(dist, int(numberOfNodes), fdp.spread);
			node->displacePosition->x += d.x / dist * rf;
			node->displacePosition->y += d.y / dist * rf;
		}

		// Attractive Force
		// Nodes that are connected pull one another
		for (auto& cNode : node->connectedNodes) {
			Vector2 d = *node->position - *cNode->position;
			auto dist = max(sqrtf(d.x * d.x + d.y * d.y), 0.001f);
			auto af = attractiveForce(dist, int(numberOfNodes), fdp.spread);
			node->displacePosition->x -= d.x / dist * af;
			node->displacePosition->y -= d.y / dist * af;
		}
	}
	// Update
	for (auto& node : nodes) {
		float dist = sqrtf(node->displacePosition->x * node->displacePosition->x + node->displacePosition->y * node->displacePosition->y);
		node->position->x += (dist > fdp.scale) ? node->displacePosition->x / dist * fdp.scale : node->displacePosition->x;
		node->position->y += (dist > fdp.scale) ? node->displacePosition->y / dist * fdp.scale : node->displacePosition->y;
		node->position->x = min(fdp.windowSize.x / 2.0f, max(-fdp.windowSize.x / 2.0f, node->position->x));
		node->position->y = min(fdp.windowSize.y / 2.0f, max(-fdp.windowSize.y / 2.0f, node->position->y));
		node->displacePosition->reset();
	}
};

/// <summary>
/// Prints all node data
/// </summary>
/// <param name="fdp">force directed placement args</param>
/// <param name="args">user args passed in</param>
void printData(FdpContext& fdp, ParamLaunch& args)
{
	std::cout << "===================" << std::endl;
	std::cout << "Nodes:\t" << args.numNodes << std::endl;
	if (args.fileName == nullptr)
		std::cout << "Seed:\t" << args.seed << std::endl;
	else
		std::cout << "File:\t" << args.fileName << std::endl;
	std::cout << "Itera:\t" << args.iterations << std::endl;
	std::cout << "Size:\t" << fdp.windowSize.x << "x" << fdp.windowSize.y << std::endl;
	std::cout << "===================" << std::endl;
}

void forceDirectedPlacement(FdpContext& fdp, ParamLaunch& args)
{
	printData(fdp, args);

	using namespace std::chrono;
	auto start = steady_clock::now();
	// Run alg 
	for (int i = 0; i < fdp.iterations; ++i)
		update(fdp);

	std::cout <<
		duration_cast<milliseconds>(steady_clock::now() - start).count()
		<< "ms" <<
		std::endl;


	for (auto& node : fdp.nodes)
	{
		node->position->x += fdp.windowSize.x / 2;
		node->position->y += fdp.windowSize.y / 2;
	}
}