#include "graph.cuh"
#include <iostream>

Graph::Graph() {
	this->nodes = NULL;
	this->displacement = NULL;
	this->distinctEdges = NULL;
	this->adjacencyMatrix = NULL;
	this->numberOfNodes = 0;
	this->numberOfEdges = 0;
};

Graph::Graph(float2* nodes,
	float2* displacement,
	int2* distinctEdges,
	int* adjacencyMatrix,
	unsigned int numberOfNodes,
	unsigned int numberOfEdges) :
	nodes(nodes),
	displacement(displacement),
	distinctEdges(distinctEdges),
	adjacencyMatrix(adjacencyMatrix),
	numberOfNodes(numberOfNodes),
	numberOfEdges(numberOfEdges)
{};

void Graph::destroy() {
	delete[] this->nodes;
	delete[] this->displacement;
	delete[] this->distinctEdges;
	delete[] this->adjacencyMatrix;
}

void Graph::printNodes() {
	for (unsigned int i = 0; i < numberOfNodes; ++i)
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

void Graph::printNodesAndConnections() {
	for (unsigned int i = 0; i < numberOfNodes; ++i) {
		std::cout << i << "\t- ";
		for (unsigned int v = 0; v < numberOfNodes; ++v) {
			std::cout << this->adjacencyMatrix[i * numberOfNodes + v] << " ";
		}
		std::cout << std::endl;
	}
}