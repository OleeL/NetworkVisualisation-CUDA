#include "Node.hpp"
#include <iostream>
#include <vector>
#include <iostream>

Node::Node() {
	this->x = 0;
	this->y = 0;
	this->id = -1;
}

Node::~Node() {

}

Node::Node(const int id, const double x, const double y) {
	this->x = x;
	this->y = y;
	this->id = id;
}

void Node::setConnectedNodes(const std::vector<Node>& connected) {
	this->connectedNodes = connected;
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

void Node::printNodesAndConnections(std::vector<Node> &nodes) {
	for (auto node : nodes) {
		std::cout
			<< node.id
			<< " (" 
			<< node.x << ", " << node.y
			<< ")\n\tConnections: ";

		auto i = 0;
		for (auto cNode : node.connectedNodes) {
			i++;
			//std::cout << i << " ";
			std::cout << cNode.id << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}