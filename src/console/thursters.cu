 // thursters.cu ___________________________________________________________________________________________________________________

#include	"thursters.h"
#include	"thrusters/hive/th_utils.h"
#include	<future>
#include	<sstream>
#include	<map>
#include <thrust/system/cuda/vector.h>

//_____________________________________________________________________________________________________________________________

Thursters::Thursters( void)
{
} 

//_____________________________________________________________________________________________________________________________

template <typename T>
struct simple_negate : public thrust::unary_function<T,T>
{
    TH_UBIQ T operator()(T x)
    {
        return -x;
    }
};
 
//_____________________________________________________________________________________________________________________________

void	Thursters::ClampTest(void)
{
    // clamp values to the range [1, 5]
    int lo = 1;
    int hi = 5; 

    // initialize values
    DVector		values(8);

    values[0] =  2;
    values[1] =  5;
    values[2] =  7;
    values[3] =  1;
    values[4] =  6;
    values[5] =  0;
    values[6] =  3;
    values[7] =  8;
     
	std::cout << "values :" << Th_Utils::IterOut( values.begin(), values.end(), "  ") << "\n";
    // define some more types

    // create a transform_iterator that applies clamp() to the values array
    auto		cv_begin = thrust::make_transform_iterator(values.begin(), Clamp<int>(lo, hi));
    auto		cv_end   = cv_begin + values.size();
    
    // now [clamped_begin, clamped_end) defines a sequence of clamped values 
	std::cout << "clamped	:"  << Th_Utils::IterOut( cv_begin, cv_end, "  ") << "\n";

    std::cout << "sum of clamped values :" << thrust::reduce( cv_begin, cv_end) << "\n";	    // compute the sum of the clamped sequence with reduce()

	thrust::counting_iterator< int> count_begin( 0);
    thrust::counting_iterator< int> count_end( 10);
     
	std::cout << "sequence :" << ": " << Th_Utils::IterOut( count_begin, count_end, "  ") << "\n";

    auto	cs_begin = thrust::make_transform_iterator(count_begin, Clamp<int>(lo, hi));
    auto	cs_end   = thrust::make_transform_iterator(count_end,   Clamp<int>(lo, hi));
	std::cout << "clamped sequence :" << Th_Utils::IterOut( cs_begin, cs_end, "  ") << "\n";
    
    auto	ncs_begin = thrust::make_transform_iterator(cs_begin, thrust::negate<int>());
    auto	ncs_end   = thrust::make_transform_iterator(cs_end,   thrust::negate<int>());
	std::cout << "negated sequence :" << Th_Utils::IterOut( ncs_begin, ncs_end, "  ") << "\n";  
    return;
}

//_____________________________________________________________________________________________________________________________

struct Functor 
{
	template<class Tuple>
	TH_UBIQ	float operator()(const Tuple& tuple) const
	{
		const float		x = thrust::get< 0>(tuple);
		const float		y = thrust::get< 1>(tuple);
		return x*y*2.0f / 3.0f;
	}
};

//_____________________________________________________________________________________________________________________________

void Thursters::XFormOutTest(void)
{
	float	u[] = { 4 , 3,  2,   1};
	float	v[] = {-1,  1,  1,  -1};
	int		idx[] = {3, 0, 1};
	float	w[] = {0, 0, 0};
	
	thrust::device_vector<float>	U(u, u + 4);
	thrust::device_vector<float>	V(v, v + 4);
	auto							zUVIt = thrust::make_zip_iterator( thrust::make_tuple( U.begin(), V.begin()));

	thrust::device_vector<int>		IDX(idx, idx + 3);
	thrust::device_vector<float>	W(w, w + 3);
	auto							outIt = thrust::make_transform_output_iterator( W.begin(), Functor());
	
	// gather multiple elements and apply a function before writing result in memory
	thrust::gather( IDX.begin(), IDX.end(), zUVIt, outIt);
	
	std::cout << "result= [ " << Th_Utils::IterOut( W.begin(), W.end(), "  ") << "] \n";
	
	return;
}


//_____________________________________________________________________________________________________________________________
/*
 * This example "welds" triangle vertices together by taking as input "triangle soup" and eliminating redundant vertex positions and shared edges.  
 * result is a connected mesh.
 * 
 *
 * Input: 9 vertices representing a mesh with 3 triangles
 *  
 *  Mesh              Vertices 
 *    ------           (2)      (5)--(4)    (8)      
 *    | \ 2| \          | \       \   |      | \
 *    |  \ |  \   <->   |  \       \  |      |  \
 *    | 0 \| 1 \        |   \       \ |      |   \
 *    -----------      (0)--(1)      (3)    (6)--(7)
 *
 *   (vertex 1 equals vertex 3, vertex 2 equals vertex 5, ...)
 *
 * Output: mesh representation with 5 vertices and 9 indices
 *
 *  Vertices            Indices
 *   (1)--(3)            [(0,2,1),
 *    | \  | \            (2,3,1), 
 *    |  \ |  \           (2,4,3)]
 *    |   \|   \
 *   (0)--(2)--(4)
 */

// define a 2d float vector

void	Thursters::WeldTest( void) 
{ 
    thrust::device_vector< PointF2>				input(9);

    input[ 0] = PointF2( 0, 0);					// First Triangle
    input[ 1] = PointF2( 1, 0);
    input[ 2] = PointF2( 0, 1);
    input[ 3] = PointF2( 1, 0);					// Second Triangle
    input[ 4] = PointF2( 1, 1);
    input[ 5] = PointF2( 0, 1);
    input[ 6] = PointF2( 1, 0);					// Third Triangle
    input[ 7] = PointF2( 2, 0);
    input[ 8] = PointF2( 1, 1);
		   
	std::cout << "Points: " << Th_Utils::IterOut( &input[ 0], &input[ 9], "  ") << " \n";

    // allocate space for output mesh representation
    thrust::device_vector<PointF2>				vertices = input;
    thrust::device_vector<unsigned int>			indices( input.size());

    thrust::sort( vertices.begin(), vertices.end());									// sort vertices to bring duplicates together
	std::cout << "Sorted: " << Th_Utils::IterOut( vertices.begin(), vertices.end(), "  ") << " \n"; 
    
    vertices.erase( thrust::unique(vertices.begin(), vertices.end()), vertices.end());	// find unique vertices and erase redundancies
	std::cout << "Unique: " << Th_Utils::IterOut( vertices.begin(), vertices.end(), "  ") << " \n";

    // find index of each input vertex in the list of unique vertices
    thrust::lower_bound( vertices.begin(), vertices.end(), input.begin(), input.end(), indices.begin());
	std::cout << "Index : " << Th_Utils::IterOut( indices.begin(), indices.end(), "  ") << " \n";
 
	auto				zipBegin = thrust::make_zip_iterator( thrust::make_tuple( indices.begin(), &input[ 0]));
	auto				zipEnd = thrust::make_zip_iterator( thrust::make_tuple( indices.end(), &input[ 9]));
	 std::for_each(zipBegin, zipEnd, []( auto x) {
			std::cout << thrust::get< 0>( x) << ": " << thrust::get< 1>( x) << "\n";
		});
    return;
}

//_____________________________________________________________________________________________________________________________

struct SaxpyFunctor 
{
    const float a;

    SaxpyFunctor(float _a) : a(_a) {}

TH_UBIQ
    float operator()(const float& x, const float& y) const { 
        return a * x + y;
    }
};

//_____________________________________________________________________________________________________________________________
 
void Thursters::SaxpyTest(void)
{
    // initialize host arrays
    float x[4] = {1.0, 1.0, 1.0, 1.0};
    float y[4] = {1.0, 2.0, 3.0, 4.0};

    // transfer to device
    thrust::device_vector<float> X(x, x + 4);
    thrust::device_vector<float> Y(y, y + 4);
 
	float	valA  = 2.0;

	thrust::transform( X.begin(), X.end(), Y.begin(), Y.begin(), SaxpyFunctor( valA) );
    std::cout << "X: " << Th_Utils::IterOut( X.begin(), X.end(), "  ") << " \n";
    std::cout << "Y: " << Th_Utils::IterOut( Y.begin(), Y.end(), "  ") << " \n";

    return;
}
 

//_____________________________________________________________________________________________________________________________

void	 Thursters::AsyncTest( void)
{
	size_t		n = 1 << 20;
	thrust::device_vector<unsigned int> data(n, 1);
	thrust::device_vector<unsigned int> result(1, 0);
 

	// method 2: use std::async to create asynchrony

	// copy all the algorithm parameters
	auto			begin        = data.begin();
	auto			end          = data.end();
	auto			binary_op    = thrust::plus< size_t>();

	// std::async captures the algorithm parameters by value,  ensure the creation of a new thread
	size_t			init = 0;
	std::future< size_t>		future_result = std::async(std::launch::async, [=] {
		return thrust::reduce( begin, end, init, binary_op);
	});

	// wait on the result and check that it is correct
	auto			res = future_result.get();
	assert( res == n); 
	
	std::cout << res << "\n";
	return;
}


//_____________________________________________________________________________________________________________________________

struct UnknownPointer
{
	std::string		message;
  
	UnknownPointer( void* p)
		: message()
	{
		std::stringstream	s;
		s << "Pointer `" << p << "` was not allocated by this allocator.";
		message = s.str();
	}

	virtual ~UnknownPointer() {}

	virtual const char	*what() const
	{
		return message.c_str();
	}
};

//_____________________________________________________________________________________________________________________________
// A simple allocator for caching cudaMalloc allocations.
struct CachedAllocator
{
	typedef char									value_type;
	typedef std::multimap<std::ptrdiff_t, char*>	free_blocks_type;
	typedef std::map<char*, std::ptrdiff_t>			allocated_blocks_type;

	free_blocks_type		free_blocks;
	allocated_blocks_type	allocated_blocks;

	void free_all()
	{
		std::cout << "cached_allocator::free_all()" << std::endl;
		for ( free_blocks_type::iterator i = free_blocks.begin() ; i != free_blocks.end() ; ++i)
			thrust::cuda::free( thrust::cuda::pointer<char>(i->second));				// Transform the pointer to cuda::pointer before calling cuda::free.

		for( allocated_blocks_type::iterator i = allocated_blocks.begin(); i != allocated_blocks.end(); ++i)
			thrust::cuda::free( thrust::cuda::pointer<char>(i->first));				// Transform the pointer to cuda::pointer before calling cuda::free.			
	}

	CachedAllocator() {}

	~CachedAllocator()
	{
		free_all();
	}

	char	*allocate( std::ptrdiff_t num_bytes)
	{
		std::cout << "CachedAllocator::allocate(): num_bytes == " << num_bytes << std::endl;

		char	*result = 0;

		// Search the cache for a free block.
		free_blocks_type::iterator			free_block = free_blocks.find( num_bytes);

		if ( free_block != free_blocks.end())
		{
			std::cout << "CachedAllocator::allocate(): found a free block" << std::endl;

			result = free_block->second;

			// Erase from the `free_blocks` map.
			free_blocks.erase( free_block);
		}
		else
		{
			// No allocation of the right size exists, so create a new one with
			// `thrust::cuda::malloc`.
			try
			{
				std::cout << "CachedAllocator::allocate(): allocating new block" << std::endl;
				// Allocate memory and convert the resulting `thrust::cuda::pointer` to a raw pointer.
				result = thrust::cuda::malloc<char>(num_bytes).get();
			}
			catch (std::runtime_error&)
			{
				throw;
			}
		}

		// Insert the allocated pointer into the `allocated_blocks` map.
		allocated_blocks.insert(std::make_pair(result, num_bytes));

		return result;
	}

	void deallocate(char *ptr, size_t)
	{
		std::cout << "CachedAllocator::deallocate(): ptr == " << reinterpret_cast<void*>(ptr) << std::endl;

		// Erase the allocated block from the allocated blocks map.
		allocated_blocks_type::iterator		iter = allocated_blocks.find(ptr);

		if ( iter == allocated_blocks.end())
			throw UnknownPointer( reinterpret_cast<void*>( ptr));

		std::ptrdiff_t	num_bytes = iter->second;
		allocated_blocks.erase(iter);

		// Insert the block into the free blocks map.
		free_blocks.insert(std::make_pair(num_bytes, ptr));
	}
 
};

//_____________________________________________________________________________________________________________________________

void	Thursters::AllocTest( void)
{
	std::size_t					num_elements = 32768; 
	thrust::host_vector< int>	h_input( num_elements);

	thrust::generate( h_input.begin(), h_input.end(), rand);		// Generate random input.

	thrust::cuda::vector<int>	d_input = h_input;
	thrust::cuda::vector<int>	d_result( num_elements);

	std::size_t					num_trials = 5;
	CachedAllocator				alloc;
	for (std::size_t i = 0; i < num_trials; ++i)
	{
		d_result = d_input;

		// Pass alloc through cuda::par as the first parameter to sort  to cause allocations to be handled by alloc during sort.
		thrust::sort( thrust::cuda::par(alloc), d_result.begin(), d_result.end());

		// Ensure the result is sorted.
		assert(thrust::is_sorted(d_result.begin(), d_result.end()));
	}

return;
}

//_____________________________________________________________________________________________________________________________

void	Thursters::Fire( void)
{ 
	int		major = THRUST_MAJOR_VERSION;
	int		minor = THRUST_MINOR_VERSION;

	std::cout << "Thrust v" << major << "." << minor << "\n";
	 
	AllocTest();
	//AsyncTest();
	//SaxpyTest();
	//WeldTest();
	//XFormOutTest();
	//ClampTest();
    return; 
}

//_____________________________________________________________________________________________________________________________

/* 
thrust::find(begin, end, value);

thrust::find_if(begin, end, Predicate);

thrust::copy(src_begin, src_end, dst_begin);

thrust::copy_if(src_begin, src_end, dst_begin, Predicate);

thrust::count(begin, end, value);

thrust::count_if(begin, end, Predicate);

thrust::equal(src1_begin, src1_end, src2_begin);

thrust::min_element(begin, end, [Cmp])

thrust::max_element(begin, end, [Cmp])

thrust::merge(src1_begin, src1_end, src2_begin, src2_end, dst_begin);

thrust::sort(begin, end, [Cmp])

// Map/Reduce   (transform === map)

thrust::transform(src1_begin, src1_end, src2_begin, dst_begin, BinaryOp);

thrust::reduce(begin, end, init, BinaryOp);

*/


//_____________________________________________________________________________________________________________________________
