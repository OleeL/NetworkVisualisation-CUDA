#include "graph.cuh"
#include "vector2.cuh"
#include <iostream>

Graph::Graph() {
	this->nodes = nullptr;
	this->edges = nullptr;
	this->displacement = nullptr;
	this->distinctEdges = nullptr;
	this->connectionIndex = nullptr;
	this->numberOfNodes = 0;
	this->numberOfEdges = 0;
};

Graph::Graph(Vector2f* nodes,
	Vector2f* displacement,
	Vector2i* edges,
	Vector2i* distinctEdges,
	unsigned int* connectionIndex,
	unsigned int numberOfNodes,
	unsigned int numberOfEdges) :
	nodes(nodes),
	displacement(displacement),
	edges(edges),
	distinctEdges(distinctEdges),
	connectionIndex(connectionIndex),
	numberOfNodes(numberOfNodes),
	numberOfEdges(numberOfEdges)
{};

void Graph::printNodes() {
	for (unsigned int i = 0; i < this->numberOfNodes; ++i)
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
	for (unsigned int i = 0; i < graph.numberOfEdges * 2; ++i) {
		std::cout
			<< graph.edges[i].x
			<< " connected to "
			<< graph.edges[i].y
			<< std::endl;
	}
}
