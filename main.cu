#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Node.hpp"
#include "window.hpp"
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
    auto n1 = Node(69, 100, 10);
    auto n2 = Node(70, 200, 20);

    Node nc1[1] = { n2 };
    Node nc2[1] = { n1 };

    n1.setConnectedNodes(nc1, 1);
    n2.setConnectedNodes(nc2, 1);

    const auto numNodes = 2;
    Node nodes[] = { n1, n2 };

    //Node::printNodes(nodes, numNodes);
    Node::printNodesAndConnections(nodes, numNodes);

    DrawWindow();

    return 0;
}