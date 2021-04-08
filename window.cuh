#pragma once

#include <GLFW/glfw3.h>
#include "vector2.cuh"

/// <summary>
/// Singleton class for operations on GLFW window
/// </summary>
class Window
{
protected:
	/// <summary>
	/// Window context
	/// </summary>
	GLFWwindow* window;
	int speed;
	Vector2i position;
	float zoomAmount = 0.0625f;
	float scale = 1;
	GLfloat aspect;
	float* projMatrix = nullptr;

	// this method is specified as glfw callback
	static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

	void key_callback_imp(GLFWwindow* window, int key, int scancode, int action, int mods);

public:
	int width;
	int height;
	bool pollDraw = false;

	Window(const char* windowName, const int width, const int height);
	~Window();

	// Singleton is accessed via getInstance()
	static Window* getInstance();
};