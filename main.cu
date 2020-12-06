#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Node.hpp"
#include "Draw.hpp"
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <time.h>
#include <sstream>

#define gpuErrchk(ans) gpuAssert((ans), __FILE__, __LINE__);
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code == cudaSuccess) return;
	fprintf(stderr, "Error: %s %s Ln: %d\n", cudaGetErrorString(code), file, line);
	if (abort) exit(code);
}

std::vector<Node> getNodes(int nNodes)
{
    
    const auto connectedNodesN = nNodes - 1;
    std::vector<Node> nodes;
    nodes.reserve(nNodes); // Dedicates space to the nodes vector array

    
    // Creating the nodes
    for (auto i = 0; i < nNodes; i++) {
        nodes.emplace_back(i, i * i * i, i * 50);
    }

    // Creating the node connections
    for (auto &node : nodes) {
        std::vector<Node> connections;
        connections.reserve(connectedNodesN);
        for (auto &connectedNode : nodes) {
            if (connectedNode.id == node.id) continue;
            connections.push_back(connectedNode);
        }
        node.setConnectedNodes(connections);
    }
    return nodes;
}

std::vector<Node> getNodesRandom(const int nNodes, unsigned int seed = NULL)
{
    srand((seed == NULL) ? time(NULL) : 10);
    for (auto i = 0; i < 10; i++) {
        std::cout << rand() % 100 << std::endl;
    }

    //const auto connectedNodesN = nNodes - 1;
    std::vector<Node> nodes;
    //nodes.reserve(nNodes); // Dedicates space to the nodes vector array


    //for (auto i = 0; i < nNodes; i++) {
    //    nodes.emplace_back(i, i * i * i, i * 50);
    //}

    //for (auto& node : nodes) {
    //    std::vector<Node> connections;
    //    connections.reserve(connectedNodesN);
    //    for (auto& connectedNode : nodes) {
    //        if (connectedNode.id == node.id) {
    //            continue;
    //        }
    //        connections.push_back(connectedNode);
    //    }
    //    node.setConnectedNodes(connections);
    //}
    return nodes;
}

int charToInt(char *str)
{
    std::istringstream iss(str);
    int val;
    return (iss >> val) ? val : -1;
}

typedef struct ParamLaunch {

    ParamLaunch(int nodes, int s) {
        this->numNodes = nodes;
        this->seed = s;
    }

    ParamLaunch();

    int numNodes;
    int seed;

} ParamLaunch;

void PrintArgErrorMessage()
{
    std::cout << "Usage: fyp_cuda_nodes [numOfNodes] [seed]" << std::endl;
    std::cout << "Using default parameters..." << std::endl;
}

// Takes the arguments passed into the program
// Args: [int NumberOfNodes, int Seed]
// Returns the seed and number of nodes to be handled.
ParamLaunch handleArgs(int argc, char *argv[])
{
    const auto defaultNodes = 10;
    const auto defaultSeed = NULL;
    const auto defaultLaunchParam = ParamLaunch(defaultNodes, defaultSeed);



    if (argc > 3 || argc < 2) {
        PrintArgErrorMessage();
        return defaultLaunchParam;
    }

    // Nodes
    auto nodes = charToInt(argv[1]);
    if (nodes == -1) {
        PrintArgErrorMessage();
        nodes = defaultNodes;
    }

    // Seeds
    auto seed = charToInt(argv[2]);
    if (seed < 0) seed = defaultSeed;

    return ParamLaunch(nodes, seed);
}

int main(int argc, char* argv[])
{
    auto args = handleArgs(argc, argv);
    std::cout << "===============" <<                  std::endl;
    std::cout << "Nodes: "         << args.numNodes << std::endl;
    std::cout << "Seed: "          << args.seed     << std::endl;
    std::cout << "===============" <<                  std::endl;

    auto nodes = getNodes(args.numNodes);

    Node::printNodesAndConnections(nodes);

    auto drawNodes = Draw();
    drawNodes.draw(nodes);

    return 0;
}