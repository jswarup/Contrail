 // thursters.cu ___________________________________________________________________________________________________________________

#include "thursters.h"

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

template <typename Iterator>
void print_range(const std::string& name, Iterator first, Iterator last)
{
    typedef typename std::iterator_traits<Iterator>::value_type T;

    std::cout << name << ": ";
    thrust::copy(first, last, std::ostream_iterator<T>(std::cout, " "));  
    std::cout << "\n";
} 

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
    
    print_range("values         ", values.begin(), values.end());

    // define some more types

    // create a transform_iterator that applies clamp() to the values array
    auto		cv_begin = thrust::make_transform_iterator(values.begin(), Clamp<int>(lo, hi));
    auto		cv_end   = cv_begin + values.size();
    
    // now [clamped_begin, clamped_end) defines a sequence of clamped values
    print_range("clamped values ", cv_begin, cv_end); 

    // compute the sum of the clamped sequence with reduce()
    std::cout << "sum of clamped values : " << thrust::reduce(cv_begin, cv_end) << "\n";

	thrust::counting_iterator<int> count_begin(0);
    thrust::counting_iterator<int> count_end(10);
    
    print_range("sequence         ", count_begin, count_end);

    auto	cs_begin = thrust::make_transform_iterator(count_begin, Clamp<int>(lo, hi));
    auto	cs_end   = thrust::make_transform_iterator(count_end,   Clamp<int>(lo, hi));

    print_range("clamped sequence ", cs_begin, cs_end);
    
    auto	ncs_begin = thrust::make_transform_iterator(cs_begin, thrust::negate<int>());
    auto	ncs_end   = thrust::make_transform_iterator(cs_end,   thrust::negate<int>());

    print_range("negated sequence ", ncs_begin, ncs_end);
	 

    return;
}

//_____________________________________________________________________________________________________________________________

void	Thursters::Fire( void)
{ 
	int		major = THRUST_MAJOR_VERSION;
	int		minor = THRUST_MINOR_VERSION;

	std::cout << "Thrust v" << major << "." << minor << std::endl;

    // create an STL list with 4 values
    std::list<int> stl_list;

    stl_list.push_back(10);
    stl_list.push_back(20);
    stl_list.push_back(30);
    stl_list.push_back(40); 
	ClampTest();
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
