// particle.h ___________________________________________________________________________________________________________________
#pragma once

#include	"thrusters/tenor/th_includes.h"

//_____________________________________________________________________________________________________________________________

class PointF3
{
public:
	float x;
	float y;
	float z;

	PointF3( void) 
	{}

	PointF3( float xIn, float yIn, float zIn) 
		: x(xIn), y(yIn), z(zIn)
	{}

	 void	randomize();

	TH_UBIQ void	Normalize();
	TH_UBIQ void	Scramble();

};

//_____________________________________________________________________________________________________________________________

struct PointF2 :  public thrust::tuple<float,float>
{
	typedef thrust::tuple<float,float>	Base;

	TH_UBIQ	PointF2( void)
	{}

	PointF2( float x, float y)
		: Base( x, y)
	{}

	friend	std::ostream	&operator<<( std::ostream &ostr, const PointF2 &pt2)
	{
		ostr << "[ " << thrust::get< 0>( pt2) << ", " << thrust::get< 1>( pt2) << "]";
		return ostr;
	}
};

//_____________________________________________________________________________________________________________________________

class Particle
{
	PointF3		position;
	PointF3		velocity;
	PointF3		totalDistance;

public:
	Particle( void)
		: totalDistance(1,0,0)
	{
		position.randomize();
		velocity.randomize();
	}

	TH_UBIQ	void		Advance(float dist);

	const PointF3		&TotalDistance() const 
	{
		return totalDistance;
	}

}; 

//_____________________________________________________________________________________________________________________________
