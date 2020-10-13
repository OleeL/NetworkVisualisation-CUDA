#define _USE_MATH_DEFINES
#include <math.h>
#include "Draw.hpp"
#include "window.hpp"
#include <vector>
#include <iostream>

void drawCircle(bool fill, GLfloat cx, GLfloat cy, GLfloat r, int n_seg)
{
    if (!fill) {
        glEnable(GL_LINE_SMOOTH);
        glBegin(GL_LINE_LOOP);

        for (int i = 0; i < n_seg; i++) {
            GLfloat theta = 2.0f * M_PI * i / n_seg; // get the current angle 
            GLfloat x = r * cos(theta); // calculate the x component 
            GLfloat y = r * sin(theta); // calculate the y component 
            glVertex2f(x + cx, y + cy); // output vertex 
        }
        glEnd();
        glDisable(GL_LINE_SMOOTH);
        return;
    }
    const GLvoid* p;
    glPointSize(r);
    glEnable(GL_POINT_SMOOTH);
    glBegin(GL_POINTS);
    glVertex2i(cx, cy);
    glEnd();
}

void rectangle(GLfloat x, GLfloat y, GLfloat w, GLfloat h)
{
    glRectf(x, y, x + w, y + h);
}

void drawLine(GLfloat x1, GLfloat y1, GLfloat x2, GLfloat y2)
{
    glBegin(GL_LINES);
    glVertex2f(x1, y1);
    glVertex2f(x2, y2);
    glEnd();
}

void setColour(GLfloat r, GLfloat g, GLfloat b, GLfloat a)
{
    glColor4f(r, g, b, a);
}

bool arrayContainsN(int* arr, int elements, int n)
{
    for (auto i = 0; i < elements; i++) {
        if (arr[i] == n) return true;
    }
    return false;
}

void appendN(int* arr, int elements, int n)
{
    for (auto i = 0; i < elements; i++) {
        if (arr[i] == -1)
        {
            arr[i] = n;
            return;
        }
    }
}


Draw::Draw() : Window("Hello world", 800, 600)
{

};

void Draw::drawNodes(std::vector<Node> &nodes)
{
    const auto segments = 64;
    const auto r = 15;
    const auto numOfNodes = nodes.size();

    int** drawnLines;
    drawnLines = new int* [numOfNodes];
    for (auto i = 0; i < numOfNodes; i++) {
        drawnLines[i] = new int[nodes[i].connectedNodes.size()];
        for (auto j = 0; j < nodes[i].connectedNodes.size(); j++) {
            drawnLines[i][j] = -1;
        }
    }

    for (auto i = 0; i < numOfNodes; i++) {
        drawCircle(true, nodes[i].x, nodes[i].y, r, segments);
        for (auto j = 0; j < nodes[i].connectedNodes.size(); j++) {
            const auto endId = nodes[i].connectedNodes[j].id;
            if (arrayContainsN(drawnLines[i], nodes[i].connectedNodes.size(), endId))
                continue;

            appendN(drawnLines[endId], nodes[j].connectedNodes.size(), i);

            //std::cout
            //    << "Draw line from ("
            //    << nodes[i].id
            //    << " to "
            //    << nodes[i].connectedNodes[j].id
            //    << ")"
            //    << std::endl;

            drawLine(
                nodes[i].x,
                nodes[i].y,
                nodes[i].connectedNodes[j].x,
                nodes[i].connectedNodes[j].y);
        }
    }

    //std::cout << std::endl;

    for (int i = 0; i < numOfNodes; i++)
    {
        delete[] drawnLines[i];
    }
    delete[] drawnLines;
}

void Draw::draw(std::vector<Node>& nodes)
{

    while (!glfwWindowShouldClose(this->window))
    {
        glClear(GL_COLOR_BUFFER_BIT);
        this->drawNodes(nodes);
        glfwSwapBuffers(this->window);
        glfwPollEvents();
    }

    glfwTerminate();
    return;
};