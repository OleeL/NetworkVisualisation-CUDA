#pragma once

#include "Node.hpp"
#include "vector2.hpp"

Vector2 repulsionForce(Node& node1, Node& node2);

Vector2 calcAttractionForce(Node& node1, Node& node2, double lengthOfSpring);

void arrangeNodes(std::vector<Node>& nodes);

void update(std::vector<Node>& nodes);