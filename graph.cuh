#pragma once

/// <summary>
/// Return a graph from the nodes generated.
/// </summary>
class Graph {
public:
	float2* nodes;
	float2* displacement;  // stores x and y displacement for all nodes
	int2* distinctEdges; // useful for drawing the nodes.
	int* adjacencyMatrix;  // adjacencyMatrix of edges
	unsigned int numberOfNodes;
	unsigned int numberOfEdges;
	
	Graph();

	Graph(
		float2* nodes,
		float2* displacement,
		int2* distinctEdges,
		int* adjacencyMatrix,
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