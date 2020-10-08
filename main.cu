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

void printNodes(Node* nodes, const int numberOfNodes)
{
    for (auto i = 0; i < numberOfNodes; i++) {
        std::cout 
            << "("
            << nodes[i].x
            << ", "
            << nodes[i].y
            << ")"
            << std::endl;
    }
    std::cout << std::endl;
}

int main(void)
{
    const auto numNodes = 2;
    auto n1 = Node(-10, 0);
    auto n2 = Node(0, 10);

    Node nc1[1] = { n2 };
    Node nc2[1] = { n1 };

    n1.setConnectedNodes(nc1, 1);
    n2.setConnectedNodes(nc2, 1);

    Node nodes[numNodes] = { n1, n2 };
    printNodes(nodes, numNodes);

    DrawWindow();

    return 0;
}