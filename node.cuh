#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

class Node {
public:
	float x;
	float y;
	float dx;
	float dy;

	/// <summary>
	/// Blank Constructor 
	/// </summary>
	Node();

	/// <summary>
	/// Create a Node
	/// </summary>
	/// <param name="id">identify the node in undordered list</param>
	/// <param name="x">coordinate</param>
	/// <param name="y">coordinate</param>
	Node(float x, float y);

	/// <summary>
	/// Prints the euclidean distance between 2 nodes
	/// </summary>
	/// <param name="node1">Node 1</param>
	/// <param name="node2">Node 2</param>
	/// <returns></returns>
	float distance(Node& node);
};