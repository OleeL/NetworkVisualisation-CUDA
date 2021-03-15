#pragma once

#include "Node.hpp"
#include <vector>

/// <summary>
/// Generates the nodes randomly with a given seed.
/// This doesn't usually make attractive graphs
/// </summary>
/// <param name="numNodes"></param>
/// <param name="seed"></param>
/// <returns>a randomly connected undirected graph</returns>
std::vector<Node*> getNodesRandom(int numNodes, unsigned int seed = NULL);

/// <summary>
/// Processes a set of nodes from a file
/// </summary>
/// <param name="fileName">File name</param>
/// <returns>an undirected graph as the file describes</returns>
std::vector<Node*> handleFile(char* fileName);