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
    auto n1 = Node(0, 100, 10);
    auto n2 = Node(1, 200, 20);
    auto n3 = Node(2, 300, 70);
    auto n4 = Node(3, 400, 500);
    auto n5 = Node(4, 700, 100);
    auto n6 = Node(5, 50, 200);
    auto n7 = Node(6, 400, 150);
    auto n8 = Node(7, 20, 400);

    Node nc1[7] = { n2, n3, n4, n5, n6, n7, n8 };
    Node nc2[2] = { n1, n3 };
    Node nc3[2] = { n1, n2 };
    Node nc4[1] = { n1 };
    Node nc5[1] = { n1 };
    Node nc6[1] = { n1 };
    Node nc7[1] = { n1 };
    Node nc8[1] = { n1 };

    n1.setConnectedNodes(nc1, 7);
    n2.setConnectedNodes(nc2, 2);
    n3.setConnectedNodes(nc3, 2);
    n4.setConnectedNodes(nc4, 1);
    n5.setConnectedNodes(nc5, 1);
    n6.setConnectedNodes(nc6, 1);
    n7.setConnectedNodes(nc7, 1);
    n8.setConnectedNodes(nc8, 1);

    const auto numNodes = 8;
    Node nodes[] = { n1, n2, n3, n4, n5, n6, n7, n8 };

    //Node::printNodes(nodes, numNodes);
    Node::printNodesAndConnections(nodes, numNodes);

    auto drawNodes = new Draw();
    drawNodes->draw(nodes, numNodes);
    delete drawNodes;

    return 0;
}