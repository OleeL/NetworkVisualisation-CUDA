#include "Startup.hpp"
#include <sstream>
#include <iostream>
#include <time.h>

ParamLaunch::ParamLaunch(int nodes, long int s) {
    this->numNodes = nodes;
    this->seed = s;
}

int charToInt(char* str)
{
    std::istringstream iss(str);
    int val;
    return (iss >> val) ? val : -1;
}

void PrintDefault()
{
    std::cout << "Using default parameters... Using time based seed" << std::endl;
}

void PrintArgErrorMessage()
{
    std::cout << "Usage: fyp_cuda_nodes [numOfNodes] [seed]" << std::endl;
    PrintDefault();
}

void PrintNoSeed()
{
    std::cout << "No seed provided... Using time based seed" << std::endl;
}

// Takes the arguments passed into the program
// Args: [int NumberOfNodes, int Seed]
// Returns the seed and number of nodes to be handled.
ParamLaunch handleArgs(int argc, char* argv[])
{
    const auto defaultNodes = 10;
    const auto defaultSeed = time(NULL);
    const auto defaultLaunchParam = ParamLaunch(defaultNodes, defaultSeed);

    if (argc > 3) {
        PrintArgErrorMessage();
        return defaultLaunchParam;
    }
    if (argc == 2) {
        PrintNoSeed();
        return ParamLaunch(charToInt(argv[1]), defaultSeed);
    }

    // Nodes
    auto nodes = charToInt(argv[1]);
    if (nodes == -1) {
        PrintArgErrorMessage();
        nodes = defaultNodes;
    }

    // Seeds
    auto seed = charToInt(argv[2]);
    if (seed < 0) seed = defaultSeed;

    return ParamLaunch(nodes, seed);
}