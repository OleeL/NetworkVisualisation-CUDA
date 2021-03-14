//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include "Draw.hpp"
#include "Startup.hpp"
#include "NodeGeneration.hpp"
#include <iostream>
#include <cstdlib>

//#define gpuErrchk(ans) gpuAssert((ans), __FILE__, __LINE__);
//inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
//{
//	if (code == cudaSuccess) return;
//	fprintf(stderr, "Error: %s %s Ln: %d\n", cudaGetErrorString(code), file, line);
//	if (abort) exit(code);
//}

int main(int argc, char* argv[])
{
    const auto windowWidth = 800;
    const auto windowHeight = 600;

    auto args = handleArgs(argc, argv);
    std::cout << "===============" <<                  std::endl;
    std::cout << "Nodes: "         << args.numNodes << std::endl;
    std::cout << "Seed: "          << args.seed     << std::endl;
    std::cout << "===============" <<                  std::endl;

    //auto nodes = getNodes(args.numNodes);
    std::vector<Node> nodes;
    nodes.reserve(args.numNodes); // Dedicates space to the nodes vector array
    getNodesRandom(nodes, windowWidth, windowHeight, args.seed);

    Draw((char*) "CPU Implementation", windowWidth, windowHeight).draw(nodes);


    return 0;
}