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
