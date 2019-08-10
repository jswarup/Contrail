// particle.h ___________________________________________________________________________________________________________________
#pragma once

#include	<math.h>
#include	<random>

//_____________________________________________________________________________________________________________________________

class PointF3
{
public:
	float x;
	float y;
	float z;

	PointF3();
	PointF3(float xIn, float yIn, float zIn);
	void randomize();
	__host__ __device__ void normalize();
	__host__ __device__ void scramble();

};

//_____________________________________________________________________________________________________________________________

class particle
{
	PointF3		position;
	PointF3		velocity;
	PointF3		totalDistance;

public:
	particle();

	__host__ __device__ void advance(float dist);

	const PointF3		&TotalDistance() const 
	{
		return totalDistance;
	}

}; 

//_____________________________________________________________________________________________________________________________
