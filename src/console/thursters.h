// thursters.h ___________________________________________________________________________________________________________________
#pragma once

#include "particle.h"
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <list>
#include <vector>

//_____________________________________________________________________________________________________________________________
// Clamps a value to the range [lo, hi]

template <typename T>
struct Clamp : public thrust::unary_function<T,T>
{
    T lo, hi;

    __host__ __device__
    Clamp(T _lo, T _hi) : lo(_lo), hi(_hi) {}

    __host__ __device__
    T operator()(T x)
    {
        if (x < lo)
            return lo;
        else if (x < hi)
            return x;
        else
            return hi;
    }
};

//_____________________________________________________________________________________________________________________________

class Thursters
{
public:
	typedef thrust::device_vector<int>								DVector; 

	Thursters( void);

	void	ClampTest( void);
	void	Fire( void);
};

//_____________________________________________________________________________________________________________________________



