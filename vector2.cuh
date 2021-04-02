#pragma once

template <class T>
class Vector2 {

public:
	T x, y;

	Vector2(T x, T y) : x(x), y(y) {};

	Vector2() : x(0), y(0) { };

	/// <summary>
	/// Sets the vector to 0,0
	/// </summary>
	void reset(void);

	/// <summary>
	/// gets distance of this node to another node
	/// </summary>
	/// <returns>the distance from this node to another node</returns>
	float distance(Vector2<float>& node);
};

/// <summary>
/// Creating 2 fast types to access
/// </summary>
typedef Vector2<int> Vector2i;
typedef Vector2<float> Vector2f;

/// <summary>
/// negates a vector
/// </summary>
template <typename T>
Vector2<T> operator-(const Vector2<T>& right);

/// <summary>
/// increments vector on to vector
/// </summary>
template <typename T>
Vector2<T>& operator+=(Vector2<T>& left, const Vector2<T>& right);

/// <summary>
/// Decrements vector by another vector
/// </summary>
template <typename T>
Vector2<T>& operator-=(Vector2<T>& left, const Vector2<T>& right);

/// <summary>
/// Adds two vectors together
/// </summary>
template <typename T>
Vector2<T> operator+(const Vector2<T>& left, const Vector2<T>& right);

/// <summary>
/// Vector is equal to itself multiplied by a type
/// </summary>
template <typename T>
Vector2<T>& operator*=(Vector2<T>& left, T right);

/// <summary>
/// divides number to both x and y
/// </summary>
template <typename T>
Vector2<T> operator/(const Vector2<T>& left, T right);

/// <summary>
/// Divides self by a value
/// </summary>
template <typename T>
Vector2<T>& operator/=(Vector2<T>& left, T right);

/// <summary>
/// Checks if two vectors are equal
/// </summary>
template <typename T>
bool operator==(const Vector2<T>& left, const Vector2<T>& right);

/// <summary>
/// Checks if two values are not equal
/// </summary>
template <typename T>
bool operator!=(const Vector2<T>& left, const Vector2<T>& right);

/// <summary>
/// subtraction of 2 vectors
/// </summary>
template <typename T>
Vector2<T> operator-(const Vector2<T>& left, const Vector2<T>& right);

/// <summary>
/// multiplication of 2 types
/// </summary>
template <typename T>
Vector2<T> operator*(const Vector2<T>& left, T right);
template <typename T>
Vector2<T> operator*(T left, const Vector2<T>& right);