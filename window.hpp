#pragma once
#include <GLFW/glfw3.h>

class Window
{

	public:
		Window(const char* windowName, const int width, const int height);

		int width, height;
		GLFWwindow* window;
};