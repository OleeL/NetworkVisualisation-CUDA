#include <algorithm>    // std::random_shuffle std::sort
#include <fstream>
#include <sstream>
#include <iostream>
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
	if (a.x < b.x) return true;
	return (a.y < b.y && a.x == b.x);
}

Graph handleFile(char* fileName) {
	unsigned int lines, nNodes;

	std::ifstream file;
	file.open(fileName);
	file >> nNodes >> lines;

	auto nodes = new Vector2f[nNodes];
	auto displacement = new Vector2f[nNodes]();
	auto distinctEdges = new Vector2i[lines];
	auto adjacencyMatrix = new bool[nNodes * nNodes]();

	// Looping through all nodes
	for (unsigned int i = 0; i < nNodes; ++i)
	{
		nodes[i] = Vector2f(
			static_cast<float>(rand()) / RAND_MAX - 0.5f,
			static_cast<float>(rand()) / RAND_MAX - 0.5f
		);
	}
	{
		auto v = Vector2i();
		// Looping through all distinct edges
		for (unsigned int i = 0; i < lines; ++i)
		{
			file >> v.x >> v.y;
			distinctEdges[i] = v;
			adjacencyMatrix[v.x * nNodes + v.y] = true;
			adjacencyMatrix[v.y * nNodes + v.x] = true;
		}
	}

	file.close();

	return Graph(nodes, displacement, distinctEdges, adjacencyMatrix, nNodes, lines);
}