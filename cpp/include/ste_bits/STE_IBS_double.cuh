// include guard
#ifndef IBS_H_INCLUDED
#define IBS_H_INCLUDED

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

// load double 6 datastructure
#include "STE_DataStructures_double.cuh"

// load tfsTableData datastructure
#include "STE_TFS_double.cuh"


struct ibsparameters {
	double acceleratorLength;
	double gammaTransition;
	double betx;
	double bety;
	double emitx;
	double emity;
	double dponp;
	double ParticleRadius;
	double betar;
	double gamma0;
	double numberRealParticles;
	double initnumberMacroParticles;
	double sigs;
	double tauhat;
	double ibsCoupling;
	double fracibstot; // artificially increase or decrease ibs with a scaling factor
	double timeratio;
	double numberMacroParticles;
	double trev;
	double nbins;
	double fmohlNumPoints;
	double seed;
	double coulomblog;
	int methodIBS;
	double phiSynchronous;
	//thrust::device_vector<tfsTableData> tfsdata;
};


struct adddouble3 
{
  __host__ __device__
  double3 operator()(const double3& a, const double3& b) const {  

  		double3 result;

		result.x = a.x + b.x;
		result.y = a.y + b.y;
		result.z = a.z + b.z;

		return result;
	}
};

struct sqrtdoubleFunctor
{
	__device__ __host__ void operator() (double& v)
	{
		v =  sqrt(v);
	}
};


// gets the particle times and uses the phase acceptance and number of bins
// to generate an integer which is than binned/histogram by ParticleTimesToHistogram
struct ParticleTimesToInteger
{
	/*
	x -> tauahat = half the phase acceptance
	y -> nbins = numbers of bins to use in binning
	z -> t synchronous to shift the center of the distribution to zero for the binning
	*/
	__host__ __device__
	int operator()(double6& data, double3& params) const 
	{
		double out;
		double dtsamp2 = 2*params.x / params.y ;
		out = (data.t - params.z + params.x ) / dtsamp2;
		out = (int)(out + 0.5f);
		return out;
	}
};


class STE_IBS
{
public:
	STE_IBS( ibsparameters&, thrust::device_vector<double6> , thrust::device_vector<tfsTableData> , double);
	// ~STE_IBS();

	
	template <typename Vector1, typename Vector2>
	void dense_histogram( const Vector1& , Vector2& , int , double  );

	double3 CalculateIBSGrowthRates( ibsparameters& , int ,  double , thrust::device_vector<tfsTableData> );
	double3 getIBSLifeTimes( double3 );
	double3 getIBSGrowthRates();

	__host__ double4 CalcIbsCoeff(ibsparameters& , int ,  double , double3 );
	
	thrust::device_vector<int> ParticleTimesToHistogram( thrust::device_vector<double6> , int , double , double );
	thrust::device_vector<double> HistogramToSQRTofCumul( thrust::device_vector<int> , double );


	void update( ibsparameters&  , thrust::device_vector<double6> , thrust::device_vector<tfsTableData> , double );

	double4 getIBScoeff();
	thrust::device_vector<int> getTimeHistogram();
	thrust::device_vector<double> getSqrtHistogram();

private:
	double6 CalcRMS( thrust::device_vector<double6> , int );

	double3 ibsGrowthRates;
	double4 ibscoeff;
	thrust::device_vector<int> histogramTime;
	thrust::device_vector<double> sqrthistogram; //denlon2k array in original code

};
#endif
