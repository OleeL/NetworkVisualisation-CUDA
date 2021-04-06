#include "vector2.cuh"
#include <cmath>

template <class T>
inline void Vector2<T>::reset(void)
{
    this->x = 0;
    this->y = 0;
};

template <class T>
inline float Vector2<T>::distance(Vector2<float>& node)
{
    return sqrtf(powf(this->y - node.y, 2) + powf(this->x - node.x, 2));
};