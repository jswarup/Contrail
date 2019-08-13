 // thursters.cu ___________________________________________________________________________________________________________________

#include	"thursters.h"
#include	"thrusters/hive/th_utils.h"

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
    const float x = thrust::get<0>(tuple);
    const float y = thrust::get<1>(tuple);
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
	auto							zUVIt = thrust::make_zip_iterator(thrust::make_tuple(U.begin(), V.begin()));

	thrust::device_vector<int>		IDX(idx, idx + 3);
	thrust::device_vector<float>	W(w, w + 3);
	auto							outIt = thrust::make_transform_output_iterator(W.begin(), Functor());
	
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
    thrust::device_vector<PointF2>					vertices = input;
    thrust::device_vector<unsigned int>				indices( input.size());

    thrust::sort( vertices.begin(), vertices.end());									// sort vertices to bring duplicates together
	std::cout << "Sorted: " << Th_Utils::IterOut( vertices.begin(), vertices.end(), "  ") << " \n"; 
    
    vertices.erase( thrust::unique(vertices.begin(), vertices.end()), vertices.end());	// find unique vertices and erase redundancies
	std::cout << "Unique: " << Th_Utils::IterOut( vertices.begin(), vertices.end(), "  ") << " \n";

    // find index of each input vertex in the list of unique vertices
    thrust::lower_bound(vertices.begin(), vertices.end(), input.begin(), input.end(), indices.begin());
	std::cout << "Index : " << Th_Utils::IterOut( indices.begin(), indices.end(), "  ") << " \n";

    // print output mesh representation
    std::cout << "Output Representation" << std::endl;
    for( size_t i = 0; i < vertices.size(); i++)
    {
        PointF2 v = vertices[i];
        std::cout << " vertices[" << i << "] = (" << thrust::get<0>(v) << "," << thrust::get<1>(v) << ")" << std::endl;
    }
    for(size_t i = 0; i < indices.size(); i++)
    {
        std::cout << " indices[" << i << "] = " << indices[i] << std::endl;
    }

    return;
}

//_____________________________________________________________________________________________________________________________

void	Thursters::Fire( void)
{ 
	int		major = THRUST_MAJOR_VERSION;
	int		minor = THRUST_MINOR_VERSION;

	std::cout << "Thrust v" << major << "." << minor << "\n";
	 
	WeldTest();
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
