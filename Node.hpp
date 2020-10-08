#pragma once
#include "math.h"

class Node {
	public:
		double x;
		double y;
		int numConnectedNodes;
		Node* connectedNodes;

		/// <summary>
		/// Blank Constructor 
		/// </summary>
		Node();
		~Node();

		/// <summary>
		/// Node Constructor
		/// </summary>
		/// <param name="x">node coord x</param>
		/// <param name="y">node coord y</param>
		/// <param name="connectedNodes">array of connected nodes</param>
		Node(const double x, const double y);

		/// <summary>
		/// Prints the euclidean distance between 2 nodes
		/// </summary>
		/// <param name="node1">Node 1</param>
		/// <param name="node2">Node 2</param>
		/// <returns></returns>
		double distance(const Node node1, const Node node2);

		void setConnectedNodes(Node* connected, int numOfNodes);

		static void printNodes(Node* nodes, const int numberOfNodes);
};