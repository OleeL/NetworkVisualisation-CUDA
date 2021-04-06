#pragma once

#include "window.cuh"
#include "graph.cuh"

class Draw : public Window
{
	public:
		Draw(char *title, int w, int h);
		void draw(Graph& graph);
		void redraw(Graph& graph);
		void drawNodes(Graph& graph);
		void* pixels;
};

