#include <math.h>
#include <iostream>
#include <algorithm>
#include <GLFW/glfw3.h>
#include "draw.cuh"
#include "window.cuh"
#include "graph.cuh"
#include "vector2.cuh"

const float PI = 3.1415927f;

void drawCircle(bool fill, GLfloat cx, GLfloat cy, GLfloat r, int n_seg)
{
	if (!fill) {
		glEnable(GL_LINE_SMOOTH);
		glBegin(GL_LINE_LOOP);

		for (auto i = 0; i < n_seg; ++i) {
			GLfloat theta = 2.0f * PI * i / n_seg; // get the current angle 
			GLfloat x = r * cos(theta); // calculate the x component 
			GLfloat y = r * sin(theta); // calculate the y component 
			glVertex2f(x + cx, y + cy); // output vertex 
		}
		glEnd();
		glDisable(GL_LINE_SMOOTH);
		return;
	}
	glPointSize(r);
	glEnable(GL_POINT_SMOOTH);
	glBegin(GL_POINTS);
	glVertex2f(cx, cy);
	glEnd();
}

// Draws a rectangle (x, y, w, h)
void rectangle(GLfloat x, GLfloat y, GLfloat w, GLfloat h)
{
	glRectf(x, y, x + w, y + h);
}

// Draws a line (x1, y1, x2, y2)
void drawLine(GLfloat x1, GLfloat y1, GLfloat x2, GLfloat y2)
{
	glBegin(GL_LINES);
	glVertex2f(x1, y1);
	glVertex2f(x2, y2);
	glEnd();
}

// Sets the color (r, g, b, a)
void setColour(GLfloat r, GLfloat g, GLfloat b, GLfloat a)
{
	glColor4f(r, g, b, a);
}


Draw::Draw(char* title, int w, int h) : Window(title, w, h)
{
	pixels = malloc(size_t(4 * width * height));
};

// Draws nodes and connections
void Draw::drawNodes(Graph& graph)
{
	const auto segments = 16;
	const auto numOfNodes = graph.numberOfNodes;
	const auto numOfEdges = graph.numberOfEdges;

	for (unsigned int i = 0; i < numOfEdges; ++i)
	{
		setColour(1.0f, 1.0f, 1.0f, 0.7f);
		drawLine(
			static_cast<GLfloat>(graph.nodes[graph.distinctEdges[i].x].x),
			static_cast<GLfloat>(graph.nodes[graph.distinctEdges[i].x].y),
			static_cast<GLfloat>(graph.nodes[graph.distinctEdges[i].y].x),
			static_cast<GLfloat>(graph.nodes[graph.distinctEdges[i].y].y));
	}

	// Iterating over all nodes
	for (unsigned int i = 0; i < numOfNodes; ++i) {
		// Drawing node
		setColour(1, 1, 1, 1);
		drawCircle(true, graph.nodes[i].x, graph.nodes[i].y, 6.0f, segments);
	}
}

inline void drawTestFrame(int x1, int y1, int& x2, int& y2)
{
	drawLine(x1, y1, x1, y2);
	drawLine(x1, y1, x2, y1);
};

void Draw::redraw(Graph& graph) {
	pollDraw = false;
	glClearColor(0.0, 0.0, 0.0, 1.0);  //clear screen by white pixel
	glClear(GL_COLOR_BUFFER_BIT);

	drawNodes(graph);
	glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
}

void Draw::draw(Graph& graph)
{
	drawNodes(graph);
	glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
	while (!glfwWindowShouldClose(window))
	{
		if (pollDraw) redraw(graph);
		glClear(GL_COLOR_BUFFER_BIT);
		glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	glfwTerminate();
	free(pixels);
	return;
};