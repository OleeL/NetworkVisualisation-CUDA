#include <iostream>
#include "forceDirectedPlacement.hpp"
#include "vector2.hpp"
#define W = 800;
#define H = 600;

// std::max doesn't handle floats
inline float max(float a, float b) {
	return (a > b) ? a : b;
}

// std::min doesn't handle floats
inline float min(float a, float b) {
	return (a < b) ? a : b;
}

inline float repulsiveForce(float dist, int numberOfNodes, float spread) {
	return spread * spread / dist / ((float) numberOfNodes) / 100.0f;
}

inline float attractiveForce(float dist, int numberOfNodes, float spread) {
    return dist * dist / spread / ((float) numberOfNodes);
}

/// <summary>
/// force directed placement algorithm
/// </summary>
/// <param name="nodes">Nodes you want to run the alg on</param>
/// <param name="damping">rate at which spring goes back to normal</param>
/// <param name="springLength">length of spring</param>
/// <param name="maxIterations">number of iterations you want
/// the alg to run. Higher = More accuracy</param>
void update(std::vector<Node>& nodes, float scale, float spread)
{
    float numberOfNodes = nodes.size();
    for (auto& node : nodes)
    {
        // Repulsive Force
        for (auto& cNode : nodes) {
            auto d = *node.position - *cNode.position;
            auto dist = sqrtf(d.x * d.x + d.y * d.y);
            dist = max(dist, 0.001f);
            auto rf = repulsiveForce(dist, numberOfNodes, spread);
            //((*node.displayPosition) += ((Vector2) d)) / dist * rf;
            node.displayPosition->x += d.x / dist * rf;
            node.displayPosition->y += d.y / dist * rf;
        }

        // Attractive Force
        for (auto& cNode : node.connectedNodes) {
            auto d_x = node.position->x - cNode->position->x;
            auto d_y = node.position->y - cNode->position->y;
            auto dist = sqrtf(d_x * d_x + d_y * d_y);
            dist = max(dist, 0.001f);
            auto af = attractiveForce(dist, numberOfNodes, spread);
            node.displayPosition->x -= d_x / dist * af;
            node.displayPosition->y -= d_y / dist * af;
        }

    }
    // Update
    for (Node& node : nodes) {
        float dist = sqrtf(node.displayPosition->x * node.displayPosition->x + node.displayPosition->y * node.displayPosition->y);
        node.position->x += (dist > scale) ? node.displayPosition->x / dist * scale : node.displayPosition->x;
        node.position->y += (dist > scale) ? node.displayPosition->y / dist * scale : node.displayPosition->x;
        node.position->x = min(800.0f / 2.0f, max(-800.0f / 2.0f, node.displayPosition->x));
        node.position->y = min(600.0f / 2.0f, max(-600.0f / 2.0f, node.displayPosition->y));
        node.displayPosition->x = 0;
        node.displayPosition->y = 0;
    }
    scale *= 0.99f;
};

/// <summary>
/// Run force directed placement algorithm on a set number of nodes for i iterations
/// </summary>
/// <param name="nodes">set of nodes</param>
/// <param name="iterations">number of iterations i (higher is more accurate)</param>
void forceDirectedPlacement(std::vector<Node>& nodes, int iterations, float scale, float spread)
{
    std::cout << "iterations: " << iterations << std::endl;

	for (int i = 0; i < iterations; ++i)
	{
		update(nodes, scale, spread);
	}

    for (Node& node : nodes)
    {
        node.position->x += 400;
        node.position->y += 300;
    }
}