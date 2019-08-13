// thursters.h ___________________________________________________________________________________________________________________
#pragma once

#include	"thrusters/tenor/th_includes.h"
#include	"particle.h"

//_____________________________________________________________________________________________________________________________
// Clamps a value to the range [lo, hi]

template <typename T>
struct Clamp : public thrust::unary_function<T,T>
{
    T lo, hi;

    TH_UBIQ	
	Clamp(T _lo, T _hi) 
		: lo(_lo), hi(_hi) 
	{}

    TH_UBIQ
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
	void	XFormOutTest(void);

	void	Fire( void);
};

//_____________________________________________________________________________________________________________________________



