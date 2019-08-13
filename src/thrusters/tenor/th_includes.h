// th_includes.h ___________________________________________________________________________________________________________________
#pragma once

#include	<math.h>
#include	<random> 
#include	<thrust/host_vector.h>
#include	<thrust/device_vector.h> 
#include	<thrust/copy.h>
#include	<thrust/fill.h>
#include	<thrust/sequence.h>
#include	<thrust/functional.h>
#include	<thrust/gather.h>
#include	<thrust/iterator/transform_output_iterator.h>
#include	<thrust/binary_search.h>
#include	<thrust/sort.h>

//_____________________________________________________________________________________________________________________________

#define TH_HOST			__host__
#define TH_DEVICE		__device__
#define TH_UBIQ			__host__ __device__ 

//_____________________________________________________________________________________________________________________________

#define TH_CUDAERROR_CHECK()																		\
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
