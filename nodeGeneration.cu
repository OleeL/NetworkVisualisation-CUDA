#include <algorithm>    // std::random_shuffle std::sort
#include <fstream>
#include <sstream>
#include <iostream>
#include "node.cuh"
#include "nodeGeneration.cuh"
#include "graph.cuh"
#include "vector2.cuh"

inline void goToLine(std::ifstream& file, int line)
{
	std::string s;
	file.clear();
	file.seekg(0);
	for (auto i = 0; i < line; i++)
		std::getline(file, s);
};

// Before edges looked like:
// 4 1
// 9 2
// 5 3
// ...
// Now, they look like
// 0 1
// 0 2
// 0 3
// 1 0
// 1 2
// ...
bool order(Vector2i &a, Vector2i &b)
{
	if (a.x < b.x)
		return true;
	else if (a.y < b.y && a.x == b.x)
		return true; // If they're the same way round then true
	return false;
}

Graph handleFile(char* fileName) {
	unsigned int lines, nNodes;

	std::ifstream file;
	file.open(fileName);
	file >> nNodes >> lines;

	auto nodes = new Node[nNodes];
	auto edges = new Vector2i[lines * 2];
	auto distinctEdges = new Vector2i[lines];
	auto connectionIndex = new unsigned int[nNodes]();

	// Looping through all nodes
	for (auto i = 0; i < nNodes; ++i)
	{
		nodes[i] = Node(
			static_cast<float>(rand()) / RAND_MAX - 0.5,
			static_cast<float>(rand()) / RAND_MAX - 0.5
		);
	}
	{
		auto v = Vector2i();
		auto inc = 0;
		// Looping through all distinct edges
		for (auto i = 0; i < lines; ++i)
		{
			file >> v.x >> v.y;
			distinctEdges[i] = v;
			edges[inc] = v;
			std::swap(v.x, v.y);
			edges[inc + 1] = v;
			inc += 2;
		}
	}
	file.close();
	std::sort(edges, edges + (2 * lines), order);
	for (auto i = 0; i < 2 * lines; ++i) {
		auto e = edges[i].x;
		connectionIndex[e] += 1;
	}
	for (auto i = 1; i < nNodes; ++i) connectionIndex[i] += connectionIndex[i - 1]; //End Index

	return Graph(nodes, edges, distinctEdges, connectionIndex, nNodes, lines);
}