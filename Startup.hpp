#pragma once

typedef struct ParamLaunch {

    ParamLaunch(int nodes, long int s);
    ParamLaunch();

    int numNodes;
    long int seed;

} ParamLaunch;

// Takes the arguments passed into the program
// Args: [int NumberOfNodes, int Seed]
// Returns the seed and number of nodes to be handled.
ParamLaunch handleArgs(int argc, char* argv[]);