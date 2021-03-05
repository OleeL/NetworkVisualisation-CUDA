#include "Node.hpp"
#include <iostream>
#include <vector>
#include "vector2.hpp"
#include <iostream>

Node::Node() {
	this->position        = new Vector2(0.0, 0.0);
	this->velocity        = new Vector2(0.0, 0.0);
	this->displayPosition = new Vector2(0.0f, 0.0f);
	this->id              = -1;
	this->radius          = 6;
}

Node::~Node() {
	delete this->position;
	delete this->displayPosition;
	delete this->velocity;
}

Node::Node(const int id, const float x, const float y) {
	this->position        = new Vector2(x, y);
	this->displayPosition = new Vector2(x, y);
	this->velocity        = new Vector2(0.0, 0.0);
	this->id              = id;
	this->radius          = 6;
}

inline float Node::distance(Node& node) {
	return this->position->distance(*node.position);
}

void Node::printNodes(Node nodes[], const int numOfNodes) {
	for (auto i = 0; i < numOfNodes; ++i) {
		std::cout << nodes[i].position->x << ", " << nodes[i].position->y << std::endl;
	}
	std::cout << std::endl;
}

void Node::printNodesAndConnections(std::vector<Node>& nodes) {
	for (auto& node : nodes) {
		std::cout
			<< node.id
			<< " ("
			<< node.position->x << ", " << node.position->y
			<< ")\n\tConnections: ";

		auto i = 0;
		for (auto& cNode : node.connectedNodes) {
			++i;
			//std::cout << i << " ";
			std::cout << cNode->id << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}