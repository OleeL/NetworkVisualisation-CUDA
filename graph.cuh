#pragma once
#include "vector2.cuh"

/// <summary>
/// Return a graph from the nodes generated.
/// </summary>
class Graph {
public:
	Vector2f* nodes;
	Vector2f* displacement;  // stores x and y displacement for all nodes
	Vector2i* distinctEdges; // useful for drawing the nodes.
	bool* adjacencyMatrix;  // adjacencyMatrix of edges
	unsigned int numberOfNodes;
	unsigned int numberOfEdges;
	
	Graph();

	Graph(
		Vector2f* nodes,
		Vector2f* displacement,
		Vector2i* distinctEdges,
		bool* adjacencyMatrix,
		unsigned int numberOfNodes,
		unsigned int numberOfEdges
	);

	/// <summary>
	/// Prints all nodes data (useful for debugging)
	/// </summary>
	void printNodes();

	/// <summary>
	/// Prints all nodes and connections (useful for debugging)
	/// </summary>
	void printNodesAndConnections();

	void destroy();
};