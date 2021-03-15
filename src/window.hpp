#pragma once

#include <GLFW/glfw3.h>

class Window
{
public:
	int width;
	int height;

	Window(const char* windowName, const int width, const int height);

	/// <summary>
	/// Window context
	/// </summary>
	GLFWwindow* window;
};