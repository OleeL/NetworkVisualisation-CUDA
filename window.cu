#include <GLFW/glfw3.h>
#include <iostream>
#include "window.cuh"
#include "vector2.cuh"
#include "draw.cuh"

Window* instance;

Window* Window::getInstance()
{
	return instance;
}

void Window::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	// here we access the instance via the singleton pattern and forward the callback to the instance method
	getInstance()->key_callback_imp(window, key, scancode, action, mods);
}

const auto LEFT = Vector2i(1, 0);
const auto RIGHT = Vector2i(-1, 0);
const auto UP = Vector2i(0, -1);
const auto DOWN = Vector2i(0, 1);

float scale = 1;

// size of 16
float* getMatrix(void)
{
	const auto s = 16;
	float* arr = new float[s]();

	arr[0] = 1.0f;
	arr[5] = 1.0f;
	arr[10] = 1.0f;
	arr[15] = 1.0f;

	return arr;
}

void Window::zoomIn()
{
	scale = std::max(0.125f, scale - zoomAmount);
	projMatrix[0] = scale;
	projMatrix[5] = scale;
}

void Window::zoomOut()
{
	scale = std::min(10.0f, scale + zoomAmount);
	projMatrix[0] = scale;
	projMatrix[5] = scale;
}

void Window::key_callback_imp(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	bool moved = false, zoomed = false;

	if (key == GLFW_KEY_W && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
		moved = true;
		this->position += UP * this->speed;
	}
	if (key == GLFW_KEY_A && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
		moved = true;
		this->position += LEFT * this->speed;
	}
	if (key == GLFW_KEY_S && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
		moved = true;
		this->position += DOWN * this->speed;
	}
	if (key == GLFW_KEY_D && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
		moved = true;
		this->position += RIGHT * this->speed;
	}
	if (key == GLFW_KEY_EQUAL && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
		zoomed = true;
		scale = std::max(0.0009765625f, scale - zoomAmount);
	}
	if (key == GLFW_KEY_MINUS && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
		zoomed = true;
		scale = std::min(64.0f, scale + zoomAmount);
	}

#if defined(DEBUG) || defined(_DEBUG)
	if (moved)
	{
		printf("(%d, %d)\n", position.x, position.y);
	}
#endif
	if (zoomed || moved)
	{
		pollDraw = true;
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glTranslatef(position.x, position.y, 0.0f);
		glScalef(scale, scale, 1.0f);

		glViewport(0, 0, width, height);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(-width / 2, width / 2, -height / 2, height / 2, 0.0f, 1.0f);
	}

}

Window::Window(const char* windowName, const int width, const int height) :
	width(width),
	height(height),
	speed(16)
{
	if (instance != nullptr) return;
	instance = this;

	/* Initialize the library */
	if (!glfwInit())
		return;

	/* Create a windowed mode window and its OpenGL context */
	window = glfwCreateWindow(width, height, windowName, NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		return;
	}

	/* Make the window's context current */
	glfwMakeContextCurrent(window);

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


	position = Vector2i();
	aspect = static_cast<float>(width) / height;
	projMatrix = getMatrix();

	glViewport(0, 0, width, height);
	glOrtho(-width / 2, width / 2, -height / 2, height / 2, 0.0f, 1.0f);

	glfwSetKeyCallback(window, Window::key_callback);

	std::cout << "===================" << std::endl;
	std::cout << "Camera pan: w, a, s, d" << std::endl;
	std::cout << "Camera zoom: - / =" << std::endl;
	std::cout << "===================" << std::endl;

}

Window::~Window()
{
	free(projMatrix);
}
