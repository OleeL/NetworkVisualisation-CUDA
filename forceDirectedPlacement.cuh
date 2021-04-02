#pragma once
#include "startup.cuh"
#include "graph.cuh"

/// <summary>
/// Run force directed placement algorithm on a set number of nodes for i iterations
/// </summary>
/// <param name="args">Passed user provided parameters</param>
/// <param name="graph">The force directed placement context to prevent huge args</param>
void forceDirectedPlacement(ParamLaunch& args, Graph& graph);