#include "window.hpp"
#include <GLFW/glfw3.h>

Window::Window(const char* windowName, const int width, const int height)
{
	/* Initialize the library */
	if (!glfwInit())
		return;

	/* Create a windowed mode window and its OpenGL context */
	this->width = width;
	this->height = height;
	this->window = glfwCreateWindow(width, height, windowName, NULL, NULL);
	if (!this->window)
	{
		glfwTerminate();
		return;
	}

	/* Make the window's context current */
	glfwMakeContextCurrent(this->window);

	// Setting the window ready for drawing.
	glEnable(GL_BLEND);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
	glfwWindowHint(GLFW_SAMPLES, 4);
	glLineWidth(0.5f);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity(); // Resets any previous projection matrices
	glMatrixMode(GL_MODELVIEW);
	glClear(GL_COLOR_BUFFER_BIT);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	GLfloat aspect = width / height;
	glViewport(0, 0, width, height);
	glOrtho(0, width / aspect, height / aspect, 0.0, 0.0, 1.0);
};