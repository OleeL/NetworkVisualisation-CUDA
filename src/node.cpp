#include "Node.hpp"
#include <iostream>
#include <vector>
#include "vector2.hpp"
#include <iostream>

Node::Node() {
	this->position         = new Vector2(0.0f, 0.0f);
	this->displacePosition = new Vector2(0.0f, 0.0f);
	this->id               = -1;
}

Node::~Node() {
	delete this->position;
	delete this->displacePosition;
}

Node::Node(const int id, const float x, const float y) {
	this->position         = new Vector2(x, y);
	this->displacePosition = new Vector2(0.0f, 0.0f);
	this->id               = id;
}

inline float Node::distance(Node& node) {
	return this->position->distance(*node.position);
}

void Node::printNodes(std::vector<Node*>& nodes) {
	for (auto i = 0; i < nodes.size(); ++i) {
		std::cout << nodes[i]->position->x << ", " << nodes[i]->position->y << std::endl;
	}
	std::cout << std::endl;
}

void Node::printNodesAndConnections(std::vector<Node*>& nodes) {
	for (auto& node : nodes) {
		std::cout
			<< node->id
			<< " ("
			<< node->position->x << ", " << node->position->y
			<< ")\n\tConnections: ";

		auto i = 0;
		for (auto& cNode : node->connectedNodes) {
			++i;
			//std::cout << i << " ";
			std::cout << cNode->id << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}
