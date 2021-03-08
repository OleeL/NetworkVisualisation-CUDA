#pragma once

#include "Node.hpp"
#include "vector2.hpp"
#include "startup.hpp"

/// <summary>
/// Force directed placement context
/// </summary>
typedef struct FdpContext {
public:
	std::vector<Node*>& nodes;
	Vector2& windowSize;
	float scale;  // This is known as k in reingold's
	float spread; // 
	int iterations;

	/// <summary>
	/// 
	/// </summary>
	/// <param name="n">nodes</param>
	/// <param name="w">windowSize</param>
	/// <param name="sc">scale</param>
	/// <param name="sp">spread</param>
	/// <param name="i">iterations</param>
	FdpContext(std::vector<Node*>& n, Vector2& w, float sc, float sp, int i)
		: nodes(n), windowSize(w), scale(sc), spread(sp), iterations(i){}
} FdpContext;

/// <summary>
/// Run force directed placement algorithm on a set number of nodes for i iterations
/// </summary>
/// <param name="fdp">The force directed placement context to prevent huge arguments</param>
/// <param name="args">Passed user provided parameters</param>
void forceDirectedPlacement(FdpContext& fdp, ParamLaunch& args);