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

// load double 6 datastructure
#include "STE_DataStructures_double.cuh"

// load tfsTableData datastructure
#include "STE_TFS_double.cuh"

// structure for giving to different functions, makes passing of variable easier
struct hamiltonianParameters
{
	double eta;
	double angularFrequency;
	double p0;
	double betar;
	double particleCharge;
	double delta;
	double t;
	double phis;


	double3 voltages;
	double3 harmonicNumbers;
};


class STE_Longitudinal_Hamiltonian
{
public:
	STE_Longitudinal_Hamiltonian( hamiltonianParameters& );
	// ~STE_Longitudinal_Hamiltonian();

	__host__ __device__ double tcoeff( hamiltonianParameters& );
	__host__ __device__ double pcoeff( hamiltonianParameters& , double );
	__host__ __device__ double HamiltonianTripleRf( hamiltonianParameters& );
	// __host__ __device__ double VoltageTripleRf( hamiltonianParameters& );

}; 

#endif