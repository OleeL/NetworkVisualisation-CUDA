#pragma once

#include "window.hpp"
#include "Node.hpp"

class Draw : public Window
{
	public:
		Draw(char *title, int w, int h);
		void draw(std::vector<Node*> &nodes);

	private:
		void drawNodes(std::vector<Node*> &nodes);
};

