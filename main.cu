#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Node.hpp"
#include "Draw.hpp"
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <vector>

#define gpuErrchk(ans) gpuAssert((ans), __FILE__, __LINE__);
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code == cudaSuccess) return;
	fprintf(stderr, "Error: %s %s Ln: %d\n", cudaGetErrorString(code), file, line);
	if (abort) exit(code);
}

std::vector<Node> getNodes(const int nNodes);

int main(void)
{
    auto nodes = getNodes(8);

    std::cout << "size: " << nodes[0].connectedNodes.size() << std::endl;

    Node::printNodesAndConnections(nodes);

    auto drawNodes = Draw();
    drawNodes.draw(nodes);

    return 0;
}

std::vector<Node> getNodes(const int nNodes)
{
    const auto connectedNodesN = nNodes - 1;
    std::vector<Node> nodes;
    nodes.reserve(nNodes);

    for (auto i = 0; i < nNodes; i++) {
        nodes.emplace_back(i, i * i * i, i * 50);
    }

    for (auto &node : nodes) {
        std::vector<Node> connections;
        connections.reserve(connectedNodesN);
        for (auto &cNode : nodes) {
            if (cNode.id == node.id) {
                continue;
            }
            connections.push_back(cNode);
        }
        node.setConnectedNodes(connections);
    }
    return nodes;
}