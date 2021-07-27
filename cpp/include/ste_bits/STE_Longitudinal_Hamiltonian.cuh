// include guard
#ifndef HAMILTONIAN_H_INCLUDED
#define HAMILTONIAN_H_INCLUDED

#include <cstdlib>
#include <cmath>
#include <math.h>
#include <string>
#include <vector>
#include <set>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <boost/lexical_cast.hpp>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>

#include <iterator>
#include <sstream>
#include <fstream>
#include <algorithm>

// load float 6 datastructure
#include "STE_DataStructures.cuh"

// load tfsTableData datastructure
#include "STE_TFS.cuh"

// structure for giving to different functions, makes passing of variable easier
struct hamiltonianParameters
{
	float eta;
	float angularFrequency;
	float p0;
	float betar;
	float particleCharge;
	float delta;
	float t;
	double phis;


	float3 voltages;
	float3 harmonicNumbers;
};


class STE_Longitudinal_Hamiltonian
{
public:
	STE_Longitudinal_Hamiltonian( hamiltonianParameters& );
	// ~STE_Longitudinal_Hamiltonian();

	__host__ __device__ float tcoeff( hamiltonianParameters& );
	__host__ __device__ float pcoeff( hamiltonianParameters& , float );
	__host__ __device__ float HamiltonianTripleRf( hamiltonianParameters& );
	// __host__ __device__ float VoltageTripleRf( hamiltonianParameters& );

}; 

#endif