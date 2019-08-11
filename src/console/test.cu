 // test.cu ___________________________________________________________________________________________________________________

#include "particle.h"
#include <stdlib.h>
#include <stdio.h>

//_____________________________________________________________________________________________________________________________

__global__ void advanceParticles(float dt, Particle * pArray, int nParticles)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if(idx < nParticles)
	{
		pArray[idx].advance(dt);
	}
}

//_____________________________________________________________________________________________________________________________


int main(int argc, char ** argv)
{

	int		n = 1000000;
	if (argc > 1) 
		n = atoi(argv[1]);      // Number of particles
	if (argc > 2) 
		srand(atoi(argv[2]));	// Random seed

	Fl_CUDAERROR_CHECK()

	Particle	*pArray = new Particle[n];
	Particle	*devPArray = NULL;
	cudaMalloc( &devPArray, n * sizeof(Particle));
	cudaDeviceSynchronize(); 
	Fl_CUDAERROR_CHECK()

	cudaMemcpy(devPArray, pArray, n * sizeof( Particle), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize(); 
	Fl_CUDAERROR_CHECK()

	for (int i = 0; i < 100; i++)
	{
		float dt = (float)rand() / (float)RAND_MAX; // Random distance each step
		advanceParticles << < 1 + n / 256, 256 >> > (dt, devPArray, n);
		 
		Fl_CUDAERROR_CHECK()
		cudaDeviceSynchronize();
	}
	cudaMemcpy(pArray, devPArray, n * sizeof(Particle), cudaMemcpyDeviceToHost);

	PointF3			totalDistance(0, 0, 0);
	PointF3			temp;
	for (int i = 0; i < n; i++)
	{
		temp = pArray[i].TotalDistance();
		totalDistance.x += temp.x;
		totalDistance.y += temp.y;
		totalDistance.z += temp.z;
	}
	float		avgX = totalDistance.x / (float)n;
	float		avgY = totalDistance.y / (float)n;
	float		avgZ = totalDistance.z / (float)n;
	float		avgNorm = sqrt(avgX * avgX + avgY * avgY + avgZ * avgZ);
	printf("Moved %d particles 100 steps. Average distance traveled is |(%f, %f, %f)| = %f\n", n, avgX, avgY, avgZ, avgNorm);
	return 0;
}
