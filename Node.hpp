#pragma once
#include "math.h"
#include <vector>

class Node {
	public:
		double x;
		double y;
		int id;
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
		double distance(const Node node1, const Node node2);

		static void printNodes(Node nodes[], const int numOfNodes);

		static void printNodesAndConnections(std::vector<Node> &nodes);
};