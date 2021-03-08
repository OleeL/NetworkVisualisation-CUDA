#include "Startup.hpp"
#include <sstream>
#include <iostream>
#include <time.h>
#include <fstream>

/// <summary>
/// Converts a set of chars to an integer
/// </summary>
/// <param name="str"></param>
/// <returns></returns>
int charToInt(char* str)
{
	std::istringstream iss(str);
	int val;
	return (iss >> val) ? val : -1;
}

/// <summary>
/// Prints a default message for bad input
/// </summary>
/// <param name="param"></param>
void PrintDefault(const ParamLaunch& param)
{
	std::cout << "Usage: fyp_cuda_nodes [numOfNodes] [seed] [iterations]" << std::endl;
	std::cout << "or" << std::endl;
	std::cout << "Usage: fyp_cuda_nodes [fileName] [iterations]" << std::endl << std::endl;
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
	auto DEFAULT_LAUNCHPARAM = ParamLaunch();

	// Checking number of args
	if (argc < 3 || argc > 4) {
		PrintDefault(DEFAULT_LAUNCHPARAM);
		return DEFAULT_LAUNCHPARAM;
	}
	if (argc == 3) {
		std::ifstream file;
		file.open(argv[1]);
		int nodes;
		file >> nodes;
		file.close();
		return ParamLaunch(argv[1], charToInt(argv[2]), nodes);
	}

	// Nodes
	auto nodes = charToInt(argv[1]);
	if (nodes < 3) {
		PrintDefault(DEFAULT_LAUNCHPARAM);
		return DEFAULT_LAUNCHPARAM;
	}

	// Seed
	int seed = (charToInt(argv[2]) < 0) ? time(NULL) : charToInt(argv[2]);

	// Iterations
	auto iterations = charToInt(argv[3]);
	if (iterations < 0) {
		PrintDefault(DEFAULT_LAUNCHPARAM);
		return DEFAULT_LAUNCHPARAM;
	}

	return ParamLaunch(nodes, seed, iterations);
}