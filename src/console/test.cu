 // test.cu ___________________________________________________________________________________________________________________

#include "thursters.h"

//_____________________________________________________________________________________________________________________________

__global__ void advanceParticles(float dt, Particle * pArray, int nParticles)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if(idx < nParticles)
	{
		pArray[idx].Advance(dt);
	}
}

//_____________________________________________________________________________________________________________________________


int main( int argc, char ** argv)
{

	int		n = 1000000;
	if (argc > 1) 
		n = atoi(argv[1]);      // Number of particles
	if (argc > 2) 
		srand(atoi(argv[2]));	// Random seed

	TH_CUDAERROR_CHECK()

	Thursters		thursters;
	thursters.Fire();
	return 0;

	thrust::device_vector< uint8_t>	devMem(  1024 *1024* 1024);
	 
	Particle	*pArray = new Particle[n];
	Particle	*devPArray = NULL;
	cudaMalloc( &devPArray, n * sizeof(Particle));
	cudaDeviceSynchronize(); 
	TH_CUDAERROR_CHECK()

	cudaMemcpy(devPArray, pArray, n * sizeof( Particle), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize(); 
	TH_CUDAERROR_CHECK()

	for (int i = 0; i < 100; i++)
	{
		float dt = (float)rand() / (float)RAND_MAX; // Random distance each step
		advanceParticles << < 1 + n / 256, 256 >> > (dt, devPArray, n);
		 
		TH_CUDAERROR_CHECK()
		cudaDeviceSynchronize();
	}
	cudaMemcpy(pArray, devPArray, n * sizeof(Particle), cudaMemcpyDeviceToHost);

	PointF3			totalDistance(0, 0, 0);
	PointF3			temp;
	for (int i = 0; i < n; i++)
	{
		temp = pArray[i].TotalDistance();
		totalDistance.X() += temp.X();
		totalDistance.Y() += temp.Y();
		totalDistance.Z() += temp.Z();
	}
	float		avgX = totalDistance.X() / (float)n;
	float		avgY = totalDistance.Y() / (float)n;
	float		avgZ = totalDistance.Z() / (float)n;
	float		avgNorm = sqrt(avgX * avgX + avgY * avgY + avgZ * avgZ);
	printf("Moved %d particles 100 steps. Average distance traveled is |(%f, %f, %f)| = %f\n", n, avgX, avgY, avgZ, avgNorm);
	return 0;
}
