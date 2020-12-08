#include "Node.hpp"
#include <vector>

#pragma once
std::vector<Node> getNodes(int nNodes);
void getNodesRandom(std::vector<Node> &nodes, const int width, const int height, unsigned int seed = NULL);