// random generator includes
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/xor_combine_engine.h>
#include <thrust/random.h>
#include <curand_kernel.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/for_each.h>


#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <map>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <math.h>
#include <string>
#include <boost/math/tools/roots.hpp>
#include <thrust/tuple.h>

#include "STE_DataStructures_double.cuh"

#include <vector>

// to write 6-vector to screen
// __host__ std::ostream& operator<< (std::ostream& os, const float6& p)
// {
// 	os << std::setw(15) << "x" << std::setw(15) << "y" << std::setw(15) << "z" << std::endl;
// 	os << std::setw(15) << p.x << std::setw(15) << p.px << std::setw(15) << p.y << std::endl;
// 	os << std::setw(15) << p.py << std::setw(15) << p.t <<std::setw(15) << p.delta << std::endl;;
// 	return os;
// };

// // to write 2-vector to screen
// __host__ std::ostream& operator<< (std::ostream& os, const float2& p)
// {
// 	os << std::setw(21) << p.x << std::setw(21) << p.y;
// 	// os << printf("%.16f",p.x) << "\t" << printf("%.16f",p.y) << std::endl;
// 	// os << printf("%.16f \t %.16f\n",p.x,p.y);
// 	return os;
// };
