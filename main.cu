#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cstdlib>
#include "draw.cuh"
#include "graph.cuh"
#include "startup.cuh"
#include "nodeGeneration.cuh"
#include "forceDirectedPlacement.cuh"

int main(int argc, char* argv[])
{
	// Handling parameters
	auto args = handleArgs(argc, argv);

	// Initialisation
	auto graph = handleFile(args.fileName);

	forceDirectedPlacement(args, graph);

	// Setting up window
	auto draw = new Draw((char*) "GPU Implementation", args.windowSize.x, args.windowSize.y);
	draw->draw(graph);

	// Cleanup
	delete draw;
	return 0;
}