#pragma once
#include "window.hpp"
#include "Node.hpp"

class Draw : public Window
{
	public:
		Draw();
		void draw(Node nodes[], const int numOfNodes);

	private:
		void drawNodes(Node nodes[], const int numOfNodes);
		//void drawCircle(bool fill, GLfloat cx, GLfloat cy, GLfloat r, int n_seg);
		//void rectangle(GLfloat x, GLfloat y, GLfloat w, GLfloat h);
		//void drawLine(GLfloat x1, GLfloat y1, GLfloat x2, GLfloat y2);
		//void setColour(GLfloat r, GLfloat g, GLfloat b, GLfloat a);
};

