#define _USE_MATH_DEFINES

#include "vector2.hpp"
#include <math.h>
#include <cmath>

const Vector2 Vector2::ZERO = Vector2(0, 0);

Vector2::Vector2(float x, float y) {
	this->x = x;
	this->y = y;
};

Vector2 Vector2::operator+(Vector2& n)
{
	return Vector2(this->x + n.x, this->y + n.y);
};

Vector2 Vector2::operator-(Vector2& n)
{
	return Vector2(this->x - n.x, this->y - n.y);
}

Vector2 Vector2::operator*(Vector2& n)
{
	return Vector2(this->x * n.x, this->y * n.y);
};

Vector2& Vector2::operator+=(Vector2& node)
{
	this->x += node.x;
	this->y += node.y;
	return *this;
};

Vector2 Vector2::operator+(float& n)
{
	return Vector2(this->x + n, this->y + n);
};

Vector2 Vector2::operator-(float& n)
{
	return Vector2(this->x - n, this->y - n);
}

Vector2 Vector2::operator*(float& n)
{
	return Vector2(this->x * n, this->y * n);
};

Vector2 Vector2::operator/(float& n)
{
	return Vector2(this->x / n, this->y / n);
}

Vector2& Vector2::operator+=(float& n)
{
	this->x += n;
	this->y += n;
	return *this;
};

void Vector2::reset(void)
{
	this->x = 0;
	this->y = 0;
};

float Vector2::distance(Vector2& node)
{
	return sqrtf(powf(this->y - node.y, 2) + powf(this->x - node.x, 2));
};