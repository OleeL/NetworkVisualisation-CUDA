#define _USE_MATH_DEFINES

#include "vector2.hpp"
#include <math.h>
#include <cmath>

const Vector2 Vector2::ZERO = Vector2(0, 0);

Vector2 Vector2::operator+(Vector2& n)
{
	return Vector2(this->x+n.x, this->y+n.y);
};

Vector2 Vector2::operator*(Vector2& n)
{
	return Vector2(this->x * n.x, this->y * n.y);
};

Vector2 Vector2::operator+(double& n)
{
	return Vector2(this->x + n, this->y + n);
};

Vector2 Vector2::operator*(double& n)
{
	return Vector2(this->x * n, this->y * n);
};

Vector2& Vector2::operator+=(Vector2& node)
{
	this->x = this->x + node.x;
	this->y = this->y + node.y;
	return *this;
};

Vector2::Vector2(double x, double y) {
	this->x = x;
	this->y = y;
	this->direction = 0;
	this->magnitude = 0;
};

double Vector2::distance(Vector2& node)
{
	return sqrt(pow(this->y - node.y, 2) + pow(this->x - node.x, 2));
};

double Vector2::getBearingAngle(Vector2& position)
{
	Vector2 half = Vector2(position.x + ((this->x - position.x) / 2), position.y + ((this->y - position.y) / 2));

	double diffX = (double)(half.x - position.x);
	double diffY = (double)(half.y - position.y);

	if (diffX == 0) diffX = 0.001;
	if (diffY == 0) diffY = 0.001;

	double angle;
	if (abs(diffX) > abs(diffY)) {
		angle = tanh(diffY / diffX) * (180.0 / M_PI);
		if (((diffX < 0) && (diffY > 0)) || ((diffX < 0) && (diffY < 0))) angle += 180;
		return angle;
	}
	angle = tanh(diffX / diffY) * (180.0 / M_PI);
	if (((diffY < 0) && (diffX > 0)) || ((diffY < 0) && (diffX < 0))) angle += 180;
	angle = (180 - (angle + 90));
	return angle;
};