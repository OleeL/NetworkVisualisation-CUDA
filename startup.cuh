#pragma once
#include <time.h>
#include "vector2.cu"

class ParamLaunch {
public:
	static const int DEFAULT_NODES = 10;
	static const int DEFAULT_ITERATIONS = 100;

	int numNodes = 10;
	int numEdges = 0;
	char* fileName = nullptr;
	long long int seed = 0;
	int iterations = 100;

	// Try to keep size W * H
	Vector2i windowSize = Vector2i(600, 600);

	ParamLaunch() {
		this->seed = time(NULL);
	};
	ParamLaunch(char* fileName, int iterations, int nodes)
		:fileName(fileName),
		iterations(iterations),
		numNodes(nodes) {};
	ParamLaunch(char* fileName, int iterations)
		:fileName(fileName),
		iterations(iterations) {};
	ParamLaunch(int nodes, long int s, int iterations)
		:numNodes(nodes),
		seed(s),
		iterations(iterations) {};
};

// Takes the arguments passed into the program
// Args: [int NumberOfNodes, int Seed]
// Returns the seed and number of nodes to be handled.
ParamLaunch handleArgs(int argc, char* argv[]);