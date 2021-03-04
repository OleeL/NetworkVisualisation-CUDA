#pragma once

#include <GLFW/glfw3.h>

class Window
{
public:
	int width;
	int height;

	Window(const char* windowName, const int width, const int height);

	GLFWwindow* window;
};