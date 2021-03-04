#pragma once

#include "math.h"
#include <vector>
#include "vector2.hpp"

class Node {
public:
	Vector2* position;
	Vector2* velocity;
	int id;
	int radius;
	std::vector<Node*> connectedNodes;

	/// <summary>
	/// Blank Constructor 
	/// </summary>
	Node();

	~Node();

	/// <summary>
	/// Create a Node
	/// </summary>
	/// <param name="id">identify the node in undordered list</param>
	/// <param name="x">coordinate</param>
	/// <param name="y">coordinate</param>
	Node(const int id, const double x, const double y);

	/// <summary>
	/// Prints the euclidean distance between 2 nodes
	/// </summary>
	/// <param name="node1">Node 1</param>
	/// <param name="node2">Node 2</param>
	/// <returns></returns>
	double distance(Node& node);

	/// <summary>
	/// Prints all nodes data (useful for debugging)
	/// </summary>
	static void printNodes(Node nodes[], const int numOfNodes);

	/// <summary>
	/// Prints all nodes and connections (useful for debugging)
	/// </summary>
	static void printNodesAndConnections(std::vector<Node>& nodes);
};