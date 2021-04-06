#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

template <class T>
class Vector2 {

public:
	T x, y;

	__device__ __host__
	Vector2(T x, T y) : x(x), y(y) {};

	__device__ __host__
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
/// increments vector on to vector
/// </summary>
template <typename T>
__inline__  __device__ __host__ inline Vector2<T>& operator+=(Vector2<T>& left, const Vector2<T>& right)
{
	left.x += right.x;
	left.y += right.y;

	return left;
}

/// <summary>
/// subtraction of 2 vectors
/// </summary>
template <typename T>
__inline__  __device__ __host__ inline Vector2<T> operator-(const Vector2<T>& left, const Vector2<T>& right) {
	return Vector2<T>(left.x - right.x, left.y - right.y);
}

/// <summary>
/// negates a vector
/// </summary>
template <typename T>
__inline__  __device__ __host__ inline Vector2<T> operator-(const Vector2<T>& right)
{
	return Vector2<T>(-right.x, -right.y);
}

/// <summary>
/// Decrements vector by another vector
/// </summary>
template <typename T>
__inline__  __device__ __host__ inline Vector2<T>& operator-=(Vector2<T>& left, const Vector2<T>& right)
{
	left.x -= right.x;
	left.y -= right.y;

	return left;
}

/// <summary>
/// Adds two vectors together
/// </summary>
template <typename T>
__inline__  __device__ __host__ inline Vector2<T> operator+(const Vector2<T>& left, const Vector2<T>& right)
{
	return Vector2<T>(left.x + right.x, left.y + right.y);
}

/// <summary>
/// multiplication of 2 types
/// </summary>
template <typename T>
__inline__  __device__ __host__ inline Vector2<T> operator*(const Vector2<T>& left, T right)
{
	return Vector2<T>(left.x * right, left.y * right);
}

template <typename T>
__inline__  __device__ __host__ inline Vector2<T> operator*(T left, const Vector2<T>& right)
{
	return Vector2<T>(right.x * left, right.y * left);
}

/// <summary>
/// Vector is equal to itself multiplied by a type
/// </summary>
template <typename T>
__inline__  __device__ __host__ inline Vector2<T>& operator*=(Vector2<T>& left, T right)
{
	left.x *= right;
	left.y *= right;

	return left;
}

/// <summary>
/// Vector is equal to itself plus a type
/// </summary>
template <typename T>
__inline__  __device__ __host__ inline Vector2<T>& operator+=(Vector2<T>& left, T right)
{
	left.x += right;
	left.y += right;

	return left;
}

/// <summary>
/// Vector is equal to itself minus a type
/// </summary>
template <typename T>
__inline__  __device__ __host__ inline Vector2<T>& operator-=(Vector2<T>& left, T right)
{
	left.x -= right;
	left.y -= right;

	return left;
}

/// <summary>
/// divides number to both x and y
/// </summary>
template <typename T>
__inline__  __device__ __host__ inline Vector2<T> operator/(const Vector2<T>& left, T right)
{
	return Vector2<T>(left.x / right, left.y / right);
}

/// <summary>
/// Divides self by a value
/// </summary>
template <typename T>
__inline__  __device__ __host__ inline Vector2<T>& operator/=(Vector2<T>& left, T right)
{
	left.x /= right;
	left.y /= right;

	return left;
}

/// <summary>
/// Checks if two vectors are equal
/// </summary>
template <typename T>
__inline__  __device__ __host__ inline bool operator==(const Vector2<T>& left, const Vector2<T>& right)
{
	return (left.x == right.x) && (left.y == right.y);
}

/// <summary>
/// Checks if two values are not equal
/// </summary>
template <typename T>
__inline__  __device__ __host__ inline bool operator!=(const Vector2<T>& left, const Vector2<T>& right)
{
	return (left.x != right.x) || (left.y != right.y);
}