#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Node.hpp"
#include "Draw.hpp"
#include <stdio.h>
#include <cstdlib>
#include <iostream>

#define gpuErrchk(ans) gpuAssert((ans), __FILE__, __LINE__);
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code == cudaSuccess) return;
	fprintf(stderr, "Error: %s %s Ln: %d\n", cudaGetErrorString(code), file, line);
	if (abort) exit(code);
}

int main(void)
{
    const auto numNodes = 8;

    Node nodes[numNodes];

    for (auto i = 0; i < numNodes; i++)
    {
        nodes[i] = Node(i, i*i*i, i*50);
    }

    for (auto i = 0; i < numNodes; i++)
    {
        const int connectedNodesN = numNodes - 1;
        Node connections[connectedNodesN];
        auto nodesToAdd = 0;
        for (auto j = 0; j < connectedNodesN; j++)
        {
            if (j == i) continue;
            connections[nodesToAdd] = nodes[j];
            nodesToAdd++;
        }
        nodes[i].setConnectedNodes(connections, 7);
    }


    //Node::printNodes(nodes, numNodes);
    Node::printNodesAndConnections(nodes, numNodes);

    auto drawNodes = Draw();
    drawNodes.draw(nodes, numNodes);

    return 0;
}