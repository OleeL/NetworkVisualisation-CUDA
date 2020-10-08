#include "window.hpp"
#include <GLFW/glfw3.h>

#define _USE_MATH_DEFINES
#include <math.h>

void drawCircle(bool fill, GLfloat cx, GLfloat cy, GLfloat r, int n_seg) {

    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

    if (fill) {
        glBegin(GL_POLYGON);
    }
    else {
        glBegin(GL_LINE_LOOP);
    }

    for (int i = 0; i < n_seg; i++) {
        GLfloat theta = 2.0f * 3.1415926f * i / n_seg; //get the current angle 
        GLfloat x = r * cos(theta); //calculate the x component 
        GLfloat y = r * sin(theta); //calculate the y component 
        glVertex2f(x + cx, y + cy); //output vertex 
    }

    glEnd();
    glDisable(GL_LINE_SMOOTH);
}

void rectangle(GLfloat x, GLfloat y, GLfloat w, GLfloat h) {
    glRectf(x, y, x + w, y + h);
}

void drawLine(GLfloat x1, GLfloat y1, GLfloat x2, GLfloat y2)
{
    glBegin(GL_LINES);
    glVertex2f(x1, y1);
    glVertex2f(x2, y2);
    glEnd();
}

void setColour(GLfloat r, GLfloat g, GLfloat b, GLfloat a) {
    glColor4f(r, g, b, a);
}

int DrawWindow(void)
{
    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit())
        return -1;

    const auto width = 640;
    const auto height = 480;

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(width, height, "Hello World", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    // Setting the window ready for drawing.
    glEnable(GL_BLEND);
    glLineWidth(0.5f);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity(); // Resets any previous projection matrices
    glMatrixMode(GL_MODELVIEW);
    glClear(GL_COLOR_BUFFER_BIT);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    GLfloat aspect = width / height;
    glViewport(0, 0, width, height);
    glOrtho(0, width / aspect, height / aspect, 0.0, 0.0, 1.0);

    const auto segments = 64;
    const auto r = 10;
    const auto cx = width / 2;
    const auto cy = height / 2;
    const auto distance = 20;

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT);

        drawLine(cx - distance, cy, cx + distance, cy);
        drawCircle(true, cx - distance, cy, r, segments);
        drawCircle(true, cx + distance, cy, r, segments);

        /* Renders graphics to screen */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}