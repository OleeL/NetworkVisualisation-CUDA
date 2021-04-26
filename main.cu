#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cstdlib>
#include "startup.cuh"
#include "nodeGeneration.cuh"
#include "forceDirectedPlacement.cuh"
#include "graph.cuh"

int main(int argc, char* argv[])
{
	// Handling parameters
	ParamLaunch* args = handleArgs(argc, argv);

	// Initialisation
	Graph* graph = handleFile(args->fileName);

	// Running algorithm
	forceDirectedPlacement(args, graph);

	graph->destroy();
	free(args);
	free(graph);
	return 0;
}