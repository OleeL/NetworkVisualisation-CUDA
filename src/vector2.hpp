#pragma once


class Vector2 {

public:
	double x;
	double y;
	double direction;
	double magnitude;

	static const Vector2 ZERO;

	Vector2(double x, double y);

	/// <summary>
	/// addition of 2 vectors
	/// </summary>
	Vector2 operator+(Vector2& n);

	/// <summary>
	/// multiplication of 2 vectors
	/// </summary>
	Vector2 operator*(Vector2& n);

	/// <summary>
	/// adds number to both x and y
	/// </summary>
	Vector2 operator+(double& n);

	/// <summary>
	/// multiplies number to both x and y
	/// </summary>
	Vector2 operator*(double& n);

	/// <summary>
	/// increments vector based on opposite vector2.
	/// </summary>
	Vector2& operator+=(Vector2& node);

	/// <summary>
	/// gets distance of this node to another node
	/// </summary>
	/// <returns>the distance from this node to another node</returns>
	double distance(Vector2& node);

	/// <summary>
	/// gets the bearing angle of this node to another node
	/// </summary>
	/// <param name="position">the position of the vector you're comparing to</param>
	/// <returns>angle in radians</returns>
	double getBearingAngle(Vector2& position);
};