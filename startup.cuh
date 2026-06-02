#pragma once
#include <time.h>
#include "vector2.cu"

enum RepulsionMode {
	REPULSION_EXACT = 0,
	REPULSION_GRID = 1,
	REPULSION_HYBRID = 2
};

class ParamLaunch {
public:
	static const int DEFAULT_NODES = 10;
	static const int DEFAULT_ITERATIONS = 100;

	int numNodes = 10;
	int numEdges = 0;
	char* fileName = nullptr;
	long long int seed = 0;
	int iterations = 100;
	int repulsionMode = REPULSION_EXACT;

	// Try to keep size W * H
	Vector2i windowSize;

	ParamLaunch()
		: seed(time(NULL)),
		windowSize(600, 600) {};

	ParamLaunch(char* fileName, int iterations, int repulsionMode = REPULSION_EXACT)
		:fileName(fileName),
		iterations(iterations),
		repulsionMode(repulsionMode),
		windowSize(600, 600) {};
	ParamLaunch(int nodes, long int s, int iterations, int repulsionMode = REPULSION_EXACT)
		:numNodes(nodes),
		seed(s),
		iterations(iterations),
		repulsionMode(repulsionMode),
		windowSize(600, 600) {};
};

// Takes the arguments passed into the program
// Args: [int NumberOfNodes, int Seed]
// Returns the seed and number of nodes to be handled.
ParamLaunch handleArgs(int argc, char* argv[]);