#pragma once


class Vector2 {

public:
	float x;
	float y;

	static const Vector2 ZERO;

	Vector2(float x, float y);

	/// <summary>
	/// addition of 2 vectors
	/// </summary>
	Vector2 operator+(Vector2& n);

	/// <summary>
	/// subtraction of 2 vectors
	/// </summary>
	Vector2 operator-(Vector2& n);

	/// <summary>
	/// multiplication of 2 vectors
	/// </summary>
	Vector2 operator*(Vector2& n);

	/// <summary>
	/// increments vector based on opposite vector2.
	/// </summary>
	Vector2& operator+=(Vector2& node);

	/// <summary>
	/// adds number to both x and y
	/// </summary>
	Vector2 operator+(float& n);

	/// <summary>
	/// subtracts number to both x and y
	/// </summary>
	Vector2 operator-(float& n);

	/// <summary>
	/// multiplies number to both x and y
	/// </summary>
	Vector2 operator*(float& n);

	/// <summary>
	/// divides number to both x and y
	/// </summary>
	Vector2 operator/(float& n);

	/// <summary>
	/// increments number to vector
	/// </summary>
	Vector2& operator+=(float& n);

	/// <summary>
	/// Sets the vector to 0,0
	/// </summary>
	void reset(void);

	/// <summary>
	/// gets distance of this node to another node
	/// </summary>
	/// <returns>the distance from this node to another node</returns>
	float distance(Vector2& node);
};