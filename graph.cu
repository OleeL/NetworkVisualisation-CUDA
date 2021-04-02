#include "graph.cuh"
#include "vector2.cuh"
#include <iostream>

Graph::Graph() {
	this->nodes = nullptr;
	this->edges = nullptr;
	this->distinctEdges = nullptr;
	this->connectionIndex = nullptr;
	this->numberOfNodes = -1;
	this->numberOfEdges = -1;
};

Graph::Graph(Node* nodes,
	Vector2i* edges,
	Vector2i* distinctEdges,
	unsigned int* connectionIndex,
	unsigned int numberOfNodes,
	unsigned int numberOfEdges) :
	nodes(nodes),
	edges(edges),
	distinctEdges(distinctEdges),
	connectionIndex(connectionIndex),
	numberOfNodes(numberOfNodes),
	numberOfEdges(numberOfEdges)
{};

void Graph::printNodes() {
	for (auto i = 0; i < this->numberOfNodes; ++i)
	{
		std::cout
			<< i
			<< ":\t"
			<< this->nodes[i].x
			<< ", "
			<< this->nodes[i].y
			<< std::endl;
	}
}

inline void Graph::printNodesAndConnections(Graph& graph) {
	for (auto i = 0; i < graph.numberOfEdges * 2; ++i) {
		std::cout
			<< graph.edges[i].x
			<< " connected to "
			<< graph.edges[i].y
			<< std::endl;
	}
}
