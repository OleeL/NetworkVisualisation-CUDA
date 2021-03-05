#pragma once

typedef struct ParamLaunch {

	ParamLaunch(int nodes, long int s, int iterations)
		:numNodes(nodes), seed(s), iterations(iterations) {};
	ParamLaunch();

	int numNodes;
	long int seed;
	int iterations;

} ParamLaunch;

// Takes the arguments passed into the program
// Args: [int NumberOfNodes, int Seed]
// Returns the seed and number of nodes to be handled.
ParamLaunch handleArgs(int argc, char* argv[]);