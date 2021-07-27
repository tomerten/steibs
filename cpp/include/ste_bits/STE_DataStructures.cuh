//include guard
#ifndef DATASTRUCTURES_H_INCLUDED
#define DATASTRUCTURES_H_INCLUDED

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

#include <vector>

#include "STE_ReadBunchFiles.cuh"

struct float6 {
	float x;
	float px;
	float y;
	float py;
	float t;
	float delta;
};


struct addFloat6 
{
  __host__ __device__
  float6 operator()(const float6& a, const float6& b) const {  

  		float6 result;

		result.x = a.x + b.x;
		result.px = a.px + b.px;
		result.y = a.y + b.y;
		result.py = a.py + b.py;
		result.t = a.t + b.t;
		result.delta = a.delta + b.delta;

		return result;
	}
};

template <typename T>
struct squareFunctor
{
	__host__ __device__ 
	T operator() (const float6& vec) const 
	{	float6 out;
		out.x     = vec.x  * vec.x;
		out.px    = vec.px * vec.px;
		out.y     = vec.y  * vec.y;
		out.py    = vec.py * vec.py;
		out.t     = vec.t  * vec.t;
		out.delta = vec.delta * vec.delta;
		return out;
	};
};



// to write 6-vector to screen
// __host__ std::ostream& operator<< (std::ostream& os, const float6& p)
// {
// 	os << std::setw(15) << p.x << std::setw(15) << p.px << std::setw(15) << p.y 
// 	<< std::setw(15) << p.py << std::setw(15) << p.t <<std::setw(15) << p.delta << std::endl;;
// 	return os;
// };

#endif

#ifndef CUDA_PI_F 
#define CUDA_PI_F 3.141592654f
#endif

#ifndef CUDA_EULER_F 
#define CUDA_EULER_F 0.577215f
#endif

#ifndef CUDA_C_F
#define CUDA_C_F 299792458.0f
#endif

#ifndef CUDA_C_R_ELECTRON
#define CUDA_C_R_ELECTRON 2.8179403227E-15f
#endif

#ifndef CUDA_HBAR_F
#define CUDA_HBAR_F 6.582119514e-16f // in eV s
#endif

#ifndef CUDA_ELECTRON_REST_E_F
#define CUDA_ELECTRON_REST_E_F 0.5109989461e6f // in eV 
#endif

#ifndef CUDA_ELEMENTARY_CHARGE
#define CUDA_ELEMENTARY_CHARGE 1.6021766208e-19 // in eV 
#endif
