#include "Startup.hpp"
#include <sstream>
#include <iostream>
#include <time.h>

int charToInt(char* str)
{
	std::istringstream iss(str);
	int val;
	return (iss >> val) ? val : -1;
}

void PrintDefault(const ParamLaunch& param)
{
	std::cout << "Usage: fyp_cuda_nodes [numOfNodes] [seed] [iterations]" << std::endl << std::endl;
	std::cout << "Using default parameters..." << std::endl;
	std::cout << "\t - Using time based seed: " << param.seed << std::endl;
	std::cout << "\t - Using default iterations: " << param.iterations << std::endl;
	std::cout << "\t - Using default nodes: " << param.numNodes << std::endl;
}

// By default, 1 argument is passed into the program
// Takes the arguments passed into the program
// Args: [int NumberOfNodes, int Seed (optional)]
// Returns the seed and number of nodes to be handled.
ParamLaunch handleArgs(int argc, char* argv[])
{
	const auto DEFAULT_NODES = 10;
	const auto DEFAULT_SEED = time(NULL);
	const auto DEFAULT_ITERATIONS = 100;
	const auto DEFAULT_LAUNCHPARAM = ParamLaunch(DEFAULT_NODES, DEFAULT_SEED, DEFAULT_ITERATIONS);

	// Checking number of args
	if (argc != 4) {
		PrintDefault(DEFAULT_LAUNCHPARAM);
		return DEFAULT_LAUNCHPARAM;
	}

	// Nodes
	auto nodes = charToInt(argv[1]);
	if (nodes < 2) {
		PrintDefault(DEFAULT_LAUNCHPARAM);
		return DEFAULT_LAUNCHPARAM;
	}

	// Seed
	auto seed = (charToInt(argv[2]) < 0) ? DEFAULT_SEED : charToInt(argv[2]);

	// Iterations
	auto iterations = charToInt(argv[3]);
	if (iterations < 0) {
		PrintDefault(DEFAULT_LAUNCHPARAM);
		return DEFAULT_LAUNCHPARAM;
	}

	return ParamLaunch(nodes, seed, iterations);
}