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
	GLFWwindow* window = nullptr;
	float speed;
	Vector2f position;
	float zoomAmount = 1.1f;
	float scale = 1.0f;
	bool dragging = false;
	double lastMouseX = 0.0;
	double lastMouseY = 0.0;
	GLfloat aspect;
	float* projMatrix = nullptr;

	// this method is specified as glfw callback
	static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
	static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
	static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
	static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);

	void key_callback_imp(GLFWwindow* window, int key, int scancode, int action, int mods);
	void scroll_callback_imp(GLFWwindow* window, double xoffset, double yoffset);
	void mouse_button_callback_imp(GLFWwindow* window, int button, int action, int mods);
	void cursor_position_callback_imp(GLFWwindow* window, double xpos, double ypos);
	void applyCamera();

public:
	int width;
	int height;
	bool pollDraw = false;

	Window(const char* windowName, const int width, const int height);
	~Window();

	void setCamera(Vector2f position, float scale);
	void resetCamera();

	// Singleton is accessed via getInstance()
	static Window* getInstance();
};