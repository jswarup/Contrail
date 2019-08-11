// particle.h ___________________________________________________________________________________________________________________
#pragma once

#include	<math.h>
#include	<random>


#define Fl_CUDAERROR_CHECK()																		\
{																									\
	cudaError_t error = cudaGetLastError();															\
																									\
	if (error != cudaSuccess)																		\
	{																								\
		printf("%s, %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error));						\
		exit(1);																					\
	}																								\
}

//_____________________________________________________________________________________________________________________________

class PointF3
{
public:
	float x;
	float y;
	float z;

	PointF3();
	PointF3(float xIn, float yIn, float zIn);
	void					 randomize();
	__host__ __device__ void normalize();
	__host__ __device__ void scramble();

};

//_____________________________________________________________________________________________________________________________

class Particle
{
	PointF3		position;
	PointF3		velocity;
	PointF3		totalDistance;

public:
	Particle();

	__host__ __device__		void advance(float dist);

	const PointF3		&TotalDistance() const 
	{
		return totalDistance;
	}

}; 

//_____________________________________________________________________________________________________________________________
