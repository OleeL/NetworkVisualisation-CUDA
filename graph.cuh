﻿#pragma once
#include "node.cuh"
#include "vector2.cuh"

/// <summary>
/// Return a graph from the nodes generated.
/// </summary>
class Graph {
public:
	Node* nodes;
	Vector2i* edges;   // stores all of the
	Vector2i* distinctEdges; // useful for drawing the nodes.

	// index represents the nodes, value represents where edges end
	// e.g.
	// ...
	// connectionIndex[2] = 13.
	// connectionIndex[3] = 14. Connection will have (14 - 13) = 1 connections
	// ...
	unsigned int* connectionIndex;
	unsigned int numberOfNodes;
	unsigned int numberOfEdges;
	
	Graph();

	Graph(
		Node* nodes,
		Vector2i* edges,
		Vector2i* distinctEdges,
		unsigned int* connectionIndex,
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
	void printNodesAndConnections(Graph& graph);
};