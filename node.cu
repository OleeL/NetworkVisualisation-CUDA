#include "node.cuh"
#include <cmath>

Node::Node() {
	this->x = 0.0f;
	this->y = 0.0f;
};

Node::Node(float x, float y) {
	this->x = x;
	this->y = y;
};

inline float Node::distance(Node& node) {
	return sqrtf(powf(this->y - node.y, 2) + powf(this->x - node.x, 2));
}