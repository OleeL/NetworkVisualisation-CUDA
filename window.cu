#include <GLFW/glfw3.h>
#include <iostream>
#include "window.cuh"
#include "vector2.cuh"
#include "draw.cuh"

Window* instance;

void glfw_error_callback(int error, const char* description)
{
	std::cerr << "GLFW error " << error << ": " << description << std::endl;
}

Window* Window::getInstance()
{
	return instance;
}

void Window::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	// here we access the instance via the singleton pattern and forward the callback to the instance method
	getInstance()->key_callback_imp(window, key, scancode, action, mods);
}

void Window::scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	getInstance()->scroll_callback_imp(window, xoffset, yoffset);
}

void Window::mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	getInstance()->mouse_button_callback_imp(window, button, action, mods);
}

void Window::cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
	getInstance()->cursor_position_callback_imp(window, xpos, ypos);
}

const auto LEFT = Vector2f(1.0f, 0.0f);
const auto RIGHT = Vector2f(-1.0f, 0.0f);
const auto UP = Vector2f(0.0f, -1.0f);
const auto DOWN = Vector2f(0.0f, 1.0f);

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

void Window::applyCamera()
{
	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-width / 2, width / 2, -height / 2, height / 2, 0.0f, 1.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(position.x, position.y, 0.0f);
	glScalef(scale, scale, 1.0f);
	pollDraw = true;
}

void Window::setCamera(Vector2f position, float scale)
{
	this->position = position;
	this->scale = std::min(128.0f, std::max(0.01f, scale));
	applyCamera();
}

void Window::resetCamera()
{
	setCamera(Vector2f(0.0f, 0.0f), 1.0f);
}

void Window::key_callback_imp(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (!(action == GLFW_PRESS || action == GLFW_REPEAT)) return;

	bool changed = false;
	float panSpeed = this->speed * (mods & GLFW_MOD_SHIFT ? 3.0f : 1.0f);

	if (key == GLFW_KEY_W || key == GLFW_KEY_UP) {
		changed = true;
		this->position += UP * panSpeed;
	}
	if (key == GLFW_KEY_A || key == GLFW_KEY_LEFT) {
		changed = true;
		this->position += LEFT * panSpeed;
	}
	if (key == GLFW_KEY_S || key == GLFW_KEY_DOWN) {
		changed = true;
		this->position += DOWN * panSpeed;
	}
	if (key == GLFW_KEY_D || key == GLFW_KEY_RIGHT) {
		changed = true;
		this->position += RIGHT * panSpeed;
	}
	if (key == GLFW_KEY_EQUAL || key == GLFW_KEY_KP_ADD) {
		changed = true;
		scale = std::min(128.0f, scale * zoomAmount);
	}
	if (key == GLFW_KEY_MINUS || key == GLFW_KEY_KP_SUBTRACT) {
		changed = true;
		scale = std::max(0.01f, scale / zoomAmount);
	}
	if (key == GLFW_KEY_R || key == GLFW_KEY_HOME) {
		resetCamera();
		return;
	}

	if (changed) applyCamera();
}

void Window::scroll_callback_imp(GLFWwindow* window, double xoffset, double yoffset)
{
	if (yoffset > 0.0) scale = std::min(128.0f, scale * zoomAmount);
	if (yoffset < 0.0) scale = std::max(0.01f, scale / zoomAmount);
	applyCamera();
}

void Window::mouse_button_callback_imp(GLFWwindow* window, int button, int action, int mods)
{
	if (button != GLFW_MOUSE_BUTTON_LEFT) return;
	dragging = action == GLFW_PRESS;
	glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
}

void Window::cursor_position_callback_imp(GLFWwindow* window, double xpos, double ypos)
{
	if (!dragging) return;

	position.x += static_cast<float>(xpos - lastMouseX);
	position.y -= static_cast<float>(ypos - lastMouseY);
	lastMouseX = xpos;
	lastMouseY = ypos;
	applyCamera();
}

Window::Window(const char* windowName, const int width, const int height) :
	width(width > 0 ? width : 600),
	height(height > 0 ? height : 600),
	speed(24.0f)
{
	if (instance != nullptr) return;
	instance = this;

	std::cout << "Opening viewer..." << std::endl;
	std::cout << "Requested window size: " << width << "x" << height << std::endl;
	std::cout << "Actual window size: " << this->width << "x" << this->height << std::endl;

	glfwSetErrorCallback(glfw_error_callback);

	/* Initialize the library */
	if (!glfwInit()) {
		std::cerr << "Failed to initialize GLFW" << std::endl;
		return;
	}

	glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);

	/* Create a windowed mode window and its OpenGL context */
	window = glfwCreateWindow(this->width, this->height, windowName, NULL, NULL);
	if (!window)
	{
		std::cerr << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return;
	}

	glfwShowWindow(window);
	glfwFocusWindow(window);

	/* Make the window's context current */
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	// Setting the window ready for drawing.
	glEnable(GL_BLEND);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
	glLineWidth(0.5f);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity(); // Resets any previous projection matrices
	glMatrixMode(GL_MODELVIEW);
	glClear(GL_COLOR_BUFFER_BIT);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	position = Vector2f(0.0f, 0.0f);
	aspect = static_cast<float>(width) / height;
	projMatrix = getMatrix();

	glViewport(0, 0, width, height);
	glOrtho(-width / 2, width / 2, -height / 2, height / 2, 0.0f, 1.0f);

	glfwSetKeyCallback(window, Window::key_callback);
	glfwSetScrollCallback(window, Window::scroll_callback);
	glfwSetMouseButtonCallback(window, Window::mouse_button_callback);
	glfwSetCursorPosCallback(window, Window::cursor_position_callback);

	std::cout << "===================" << std::endl;
	std::cout << "Camera pan: w/a/s/d, arrow keys, or left-drag" << std::endl;
	std::cout << "Camera zoom: mouse wheel, - / =" << std::endl;
	std::cout << "Camera reset: r or home" << std::endl;
	std::cout << "===================" << std::endl;

}

Window::~Window()
{
	free(projMatrix);
}
