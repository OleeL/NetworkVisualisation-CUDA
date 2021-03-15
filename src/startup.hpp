#pragma once
#include <time.h>

class ParamLaunch {
public:
	static const int DEFAULT_NODES = 10;
	static const int DEFAULT_ITERATIONS = 100;

	int numNodes = 10;
	char* fileName = nullptr;
	long long int seed = 0;
	int iterations = 100;

	ParamLaunch() {
		this->seed = time(NULL);
	};
	ParamLaunch(char* fileName, int iterations, int nodes)
		:fileName(fileName), iterations(iterations), numNodes(nodes) {};
	ParamLaunch(int nodes, long int s, int iterations)
		:numNodes(nodes), seed(s), iterations(iterations) {};


};

// Takes the arguments passed into the program
// Args: [int NumberOfNodes, int Seed]
// Returns the seed and number of nodes to be handled.
ParamLaunch handleArgs(int argc, char* argv[]);