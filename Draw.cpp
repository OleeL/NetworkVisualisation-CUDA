#define _USE_MATH_DEFINES
#include <math.h>
#include "Draw.hpp"
#include "window.hpp"
#include <vector>
#include <iostream>
#include <algorithm>

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

// Draws a rectangle (x, y, w, h)
void rectangle(GLfloat x, GLfloat y, GLfloat w, GLfloat h)
{
    glRectf(x, y, x + w, y + h);
}

// Draws a line (x1, y1, x2, y2)
void drawLine(GLfloat x1, GLfloat y1, GLfloat x2, GLfloat y2)
{
    glBegin(GL_LINES);
    glVertex2f(x1, y1);
    glVertex2f(x2, y2);
    glEnd();
}

// Sets the color (r, g, b, a)
void setColour(GLfloat r, GLfloat g, GLfloat b, GLfloat a)
{
    glColor4f(r, g, b, a);
}


Draw::Draw(char *title, int w, int h) : Window(title, w, h)
{

};

// Draws nodes and connections
void Draw::drawNodes(std::vector<Node> &nodes)
{
    const auto segments = 32;
    const auto r = 6;
    const auto numOfNodes = nodes.size();

    // Creating a 2D array to prevent redrawing connections
    std::vector<std::vector<int>> drawnLines(numOfNodes);

    for (auto i = 0; i < numOfNodes; i++) {
        const auto numConnectedNodes = nodes[i].connectedNodes.size();
        drawnLines[i].reserve(numConnectedNodes);
        for (auto j = 0; j < nodes[i].connectedNodes.size(); j++) {
            drawnLines[j].emplace_back(-1);
        }
    }

    // Iterating over all nodes
    for (auto &node : nodes) {
        // Drawing node
        setColour(1, 1, 1, 1);
        drawCircle(true, node.x, node.y, r, segments);

        // Iterating over connections
        for (auto j = 0; j < node.connectedNodes.size(); j++) {
            // The last ID of an array 
            const auto endId = node.connectedNodes[j]->id;
            if (std::find(drawnLines[node.id].begin(), drawnLines[node.id].end(), endId) != drawnLines[node.id].end())
                continue;
            drawnLines[endId].push_back(node.id);
            setColour(1, 1, 1, 0.4);
            drawLine(
                node.x,
                node.y,
                node.connectedNodes[j]->x,
                node.connectedNodes[j]->y);
        }
    }
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