#pragma once

#include <GLFW/glfw3.h>
#include "vector2.cuh"

/// <summary>
/// Singleton class for operations on GLFW window
/// </summary>
class Window
{

public:
	int width;
	int height;
	int speed;

	Vector2i position;

	float zoomAmount = 0.0625f;
	float scale = 1;

	Window(const char* windowName, const int width, const int height);
	~Window();

	void zoomIn();
	void zoomOut();

	GLfloat aspect;

	float* projMatrix = nullptr;
	bool pollDraw = false;
	
	// Singleton is accessed via getInstance()
	static Window* getInstance();

	// this method is specified as glfw callback
	static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

	void key_callback_imp(GLFWwindow* window, int key, int scancode, int action, int mods);

	/// <summary>
	/// Window context
	/// </summary>
	GLFWwindow* window;

};