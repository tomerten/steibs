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

// load Hamiltonian functions
#include "STE_Longitudinal_Hamiltonian.cuh"

// constructor
STE_Longitudinal_Hamiltonian::STE_Longitudinal_Hamiltonian( hamiltonianParameters& params )
{
	
}

__host__ __device__ float STE_Longitudinal_Hamiltonian::tcoeff( hamiltonianParameters& params )
{
	return  (params.angularFrequency * params.eta * params.harmonicNumbers.x);
}

__host__ __device__ float STE_Longitudinal_Hamiltonian::pcoeff( hamiltonianParameters& params, float voltage ){
	return  (params.angularFrequency * voltage * params.particleCharge) / (2 * CUDA_PI_F * params.p0 * params.betar);
};

__host__ __device__ float STE_Longitudinal_Hamiltonian::HamiltonianTripleRf( hamiltonianParameters& params ){

	float kinetic, potential1, potential2, potential3;

	kinetic = 0.5 * tcoeff( params ) * pow(params.delta,2);

	float pcoeff1 = pcoeff( params , params.voltages.x );
	float pcoeff2 = pcoeff( params , params.voltages.y );
	float pcoeff3 = pcoeff( params , params.voltages.z );

	float phi1 = params.harmonicNumbers.x * params.angularFrequency * params.t;
	float phi2 = params.harmonicNumbers.y * params.angularFrequency * params.t;
	float phi3 = params.harmonicNumbers.z * params.angularFrequency * params.t;

	float h0oh1 = params.harmonicNumbers.x / params.harmonicNumbers.y;
	float h0oh2 = params.harmonicNumbers.x / params.harmonicNumbers.z;

	float h1oh0 = params.harmonicNumbers.y / params.harmonicNumbers.x;
	float h2oh0 = params.harmonicNumbers.z / params.harmonicNumbers.x;

	potential1 = pcoeff1 * (cos(phi1) - cos(params.phis) + (phi1 - params.phis) * sin(params.phis));
	potential2 = pcoeff2 * h0oh1 * (cos(phi2) - cos(h1oh0 * params.phis ) + (phi2 - h1oh0 * params.phis ) * sin(h1oh0 * params.phis));
	potential3 = pcoeff3 * h0oh2 * (cos(phi3) - cos(h2oh0 * params.phis) + (phi3 - h2oh0 * params.phis) * sin(h2oh0 * params.phis));

	return kinetic + potential1 + potential2 + potential3;
};