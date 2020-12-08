#include <vector>
#include <iostream>
#include <time.h>
#include <algorithm>    // std::random_shuffle
#include <numeric>
#include "Node.hpp"

static std::vector<int> node_cache;

// Returns an array of unique node ids that which your node id could be connected to
void getConnectedNodes(std::vector<int>& values, const int maxConnections, int excludedNode)
{

    const auto nodesLength = node_cache.size();
    values = node_cache;
    if (excludedNode != nodesLength-1)
        values[excludedNode] = nodesLength-1;

    std::random_shuffle(values.begin(), values.end());

    values.resize(maxConnections);
    values.shrink_to_fit();
}

std::vector<Node> getNodes(int nNodes)
{
    const auto connectedNodesN = nNodes - 1;
    std::vector<Node> nodes;
    nodes.reserve(nNodes); // Dedicates space to the nodes vector array

    // Creating the nodes
    for (auto i = 0; i < nNodes; i++) {
        nodes.emplace_back(i, i * i, i * 10);
    }

    // Creating the node connections
    for (auto& node : nodes) {
        auto& connections = node.connectedNodes;
        connections.reserve(connectedNodesN);

        for (auto& connectedNode : nodes) {
            if (connectedNode.id == node.id) continue;
            connections.emplace_back(&connectedNode);
        }
    }
    return nodes;
}

void getNodesRandom(std::vector<Node> &nodes, const int width, const int height, unsigned int seed = NULL)
{

    const int nNodes = nodes.capacity();
    node_cache.resize(nNodes);
    std::iota(node_cache.begin(), node_cache.end(), 0); // elements are set to [0...n-1]
    
    // Setting the seed
    srand(seed);

    // Creating the nodes
    for (auto i = 0; i < nNodes; i++) {
        nodes.emplace_back(i, rand() % width, rand() % height);

        // Get lengths of connections (divided by 2 to reduce connections)
        const auto connectedNodesN = rand() % (nNodes / 2) + 1;
        // Create connections to nodes
        nodes[i].connectedNodes.reserve(connectedNodesN);
    }



    // Creating the node connections
    for (auto& node : nodes) {
        // Min number of connections a node can have is 1.
        // Max number of connections a node can have is n-1 because...
        // a node cannot be connected to itself and...
        // cannot be connected to the same node many times
        // Nodes are bidirectional (A -> B, B -> A)

        const auto connectedNodesN = node.connectedNodes.capacity();
        const auto connectedNodesCurrent = node.connectedNodes.size();
        std::vector<int> connections;
        getConnectedNodes(connections, connectedNodesN, node.id);

        for (auto i = 0; i < (connectedNodesN - connectedNodesCurrent); i++) {
            node.connectedNodes.emplace_back(&nodes[connections[i]]);
            nodes[connections[i]].connectedNodes.emplace_back(&node);
        }
    }
}