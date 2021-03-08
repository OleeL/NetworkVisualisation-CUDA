#include <vector>
#include <iostream>
#include <time.h>
#include <algorithm>    // std::random_shuffle
#include <numeric>
#include "Node.hpp"
#include <fstream>
#include <sstream>

static std::vector<int> node_cache;

// Returns an array of unique node ids that which your node id could be connected to
void getConnectedNodes(std::vector<Node*> &nodes, Node* node)
{
	// Setting constants and helpers
	auto excludedNode = node->id;
	auto& alreadyConnected = node->connectedNodes;
	const auto maxConnections = alreadyConnected.capacity() - alreadyConnected.size();

	// Getting all the connected node ids and putting them into an array 
	std::vector<int> valuesAlreadyConnected(alreadyConnected.size());
	for (auto& node : alreadyConnected) valuesAlreadyConnected.emplace_back(node->id);
	std::sort(valuesAlreadyConnected.begin(), valuesAlreadyConnected.end());

	std::vector<int> values;

	// Getting a list of all ids except for this node's id and connected ids
	std::set_difference(
		node_cache.begin(),
		node_cache.end(),
		valuesAlreadyConnected.begin(),
		valuesAlreadyConnected.end(),
		std::inserter(values, values.begin()));
	const auto itr = std::find(values.begin(), values.end(), excludedNode);
	if (itr != values.end())
		values.erase(values.begin() + (itr - values.begin()));

	// Randomizing the array and taking the amount that we need.
	std::random_shuffle(values.begin(), values.end());
	values.resize(maxConnections);
	values.shrink_to_fit();

	// Connect the random nodes to this node and this node to the random nodes
	for (auto i = 0; i < maxConnections; ++i) {
		alreadyConnected.emplace_back(nodes[values[i]]);
		nodes[values[i]]->connectedNodes.emplace_back(node);
	}
}

std::vector<Node*> getNodesRandom(int numNodes, unsigned int seed = NULL)
{
	std::vector<Node*> nodes;
	nodes.resize(numNodes); // Dedicates space to the nodes vector array

	long long int nNodes = nodes.capacity();
	node_cache.resize(nNodes - 1);
	std::iota(node_cache.begin(), node_cache.end(), 0); // elements are set to [0...n-1]

	// Setting the seed
	srand(seed);

	// Creating the nodes
	for (auto i = 0; i < nNodes; ++i) {
		nodes[i] = new Node(i, float(rand()) / RAND_MAX - 0.5, float(rand()) / RAND_MAX - 0.5);

		// Get lengths of connections (divided by 2 to reduce connections)
		const int connectedNodesN = rand() % (nNodes / 2) + 1;
		// Create connections to nodes
		nodes[i]->connectedNodes.reserve(connectedNodesN);
	}

	// - Creating the node connections
	// Min number of connections a node can have is 1.
	// Max number of connections a node can have is n-1 because...
	// a node cannot be connected to itself and...
	// cannot be connected to the same node many times
	// Nodes are bidirectional (A -> B, B -> A)
	// Graph must be complete
	for (auto& node : nodes) getConnectedNodes(nodes, node);

	return nodes;
}

inline void goToLine(std::ifstream& file, int line)
{
	std::string s;
	file.clear();
	file.seekg(0);
	for (auto i = 0; i < line; i++)
		std::getline(file, s);
}

std::vector<Node*> handleFile(char* fileName) {
	std::vector<Node*> nodes;
	std::ifstream file;
	file.open(fileName);
	int nNodes, lines, node, edge;
	file >> nNodes >> lines;

	int* edgeDictionary = new int[nNodes]();
	nodes.resize(nNodes);

	// Looping through all nodes
	for (auto i = 0; i < nNodes; ++i)
		nodes[i] = new Node(i, float(rand()) / RAND_MAX - 0.5, float(rand()) / RAND_MAX - 0.5);

	// Looping through all edges
	for (auto i = 0; i < lines; ++i)
	{
		file >> node >> edge;
		if (edge == -1) continue;

		// figuring out how many edges each node has 
		++edgeDictionary[node];
		++edgeDictionary[edge];
	}

	// Reserving space for all nodes
	for (auto i = 0; i < nNodes; ++i)
	{
		nodes[i]->connectedNodes.reserve(edgeDictionary[i]);
	}

	goToLine(file, 1);
	for (auto i = 0; i < lines; ++i) {
		file >> node >> edge;
		if (edge == -1) continue;
		nodes[node]->connectedNodes.emplace_back(nodes[edge]);
		nodes[edge]->connectedNodes.emplace_back(nodes[node]);
	}
	file.close();
	delete[] edgeDictionary;
	return nodes;
}