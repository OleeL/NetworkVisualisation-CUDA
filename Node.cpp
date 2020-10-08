#include "Node.hpp"
#include <iostream>

Node::Node() {
	this->x = 0;
	this->y = 0;
	this->connectedNodes = NULL;
	this->numConnectedNodes = 0;
}

Node::~Node() {

}
		
Node::Node(const double x, const double y) {
	this->x = x;
	this->y = y;
	this->connectedNodes = NULL;
	this->numConnectedNodes = 0;
}

void Node::setConnectedNodes(Node* connected, int numOfNodes) {
	this->connectedNodes = connected;
	this->numConnectedNodes = numOfNodes;
}

double Node::distance(const Node node1, const Node node2) {
	return sqrt(pow(node1.x - node2.x, 2) + pow(node1.y - node2.y, 2));
}

void Node::printNodes(Node* nodes, int numberOfNodes) {
	for (auto i = 0; i < numberOfNodes; i++) {
		std::cout << nodes[i].x << ", " << nodes[i].y << std::endl;
	}	
	std::cout << std::endl;
}