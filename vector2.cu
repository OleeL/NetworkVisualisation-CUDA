#include "vector2.cuh"
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <cassert>

template <class T>
inline void Vector2<T>::reset(void)
{
    this->x = 0;
    this->y = 0;
};