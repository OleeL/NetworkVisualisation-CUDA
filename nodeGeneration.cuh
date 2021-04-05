#pragma once

#include "graph.cuh"

/// <summary>
/// Processes a set of nodes from a file
/// </summary>
/// <param name="fileName">File name</param>
/// <returns>an undirected graph as the file describes</returns>
Graph handleFile(char* fileName);