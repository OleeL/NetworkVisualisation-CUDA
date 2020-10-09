#include "Node.hpp"
#include <iostream>

Node::Node() {
	this->x = 0;
	this->y = 0;
	this->id = -1;
	this->connectedNodes = nullptr;
	this->numConnectedNodes = -1;
}

Node::~Node() {

}

Node::Node(const int id, const double x, const double y) {
	this->x = x;
	this->y = y;
	this->id = id;
	this->connectedNodes = nullptr;
	this->numConnectedNodes = -1;
}

void Node::setConnectedNodes(Node connected[], int numOfNodes) {
	this->connectedNodes = connected;
	this->numConnectedNodes = numOfNodes;
}

double Node::distance(const Node node1, const Node node2) {
	return sqrt(pow(node1.x - node2.x, 2) + pow(node1.y - node2.y, 2));
}

void Node::printNodes(Node nodes[], const int numOfNodes) {
	for (auto i = 0; i < numOfNodes; i++) {
		std::cout << nodes[i].x << ", " << nodes[i].y << std::endl;
	}	
	std::cout << std::endl;
}

void Node::printNodesAndConnections(Node nodes[], const int numOfNodes) {
	for (auto i = 0; i < numOfNodes; i++) {
		std::cout
			<< nodes[i].id
			<< " - (" 
			<< nodes[i].x << ", " << nodes[i].y
			<< ") Connected to\t- ";

		for (auto j = 0; j < nodes[i].numConnectedNodes; j++) {
			std::cout << nodes[i].connectedNodes[j].id << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}