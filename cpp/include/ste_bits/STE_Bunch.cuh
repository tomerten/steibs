// include guard
#ifndef BUNCH_H_INCLUDED
#define BUNCH_H_INCLUDED

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
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <boost/math/tools/roots.hpp>
#include <boost/function.hpp>
#include <thrust/tuple.h>


#include <iterator>
#include <sstream>
#include <fstream>
#include <algorithm>

// load float 6 datastructure
#include "STE_DataStructures.cuh"

// load tfsTableData datastructure
#include "STE_TFS.cuh"

#include "STE_Radiation.cuh"
#include "STE_Longitudinal_Hamiltonian.cuh"
#include "STE_Synchrotron.cuh"
#include "STE_Random.cuh"

// load ibs
#include "STE_IBS.cuh"

struct bunchparameters {
	int bucket;
	float tunex;
	float tuney;
	float ksix;
	float ksiy;
	float trev;
	float timeratio;
	std::string particleType;
	int particleAtomNumber;
	int particleCharge;
	int methodRadiationIntegrals;
	std::string tfsfile;
	int numberMacroParticles;
	int methodLongitudinalDist;
	float realNumberOfParticles;
	float k2l;
	float k2sl;
	float betatroncoupling;
	int conversion;

	radiationIntegralsParameters radparams;
	hamiltonianParameters hamparams;
	synchrotronParameters synparams;
	longitudinalParameters longparams;
	ibsparameters ibsparams;

 };


struct RadiationDampingRoutineFunctor{
	RadiationDampingRoutineFunctor(float6 radiationDampParams, float trev, float timeratio, int seed) : radiationDampParams(radiationDampParams), trev(trev) ,timeratio(timeratio), seed(seed) {}
	__host__ __device__ float6 operator()(float6 particle)
	{
		/*
		 * .x     -> t_emitx
		 * .px    -> t_emity
		 * .y     -> t_sigs
		 * .py    -> emit equilib x
		 * .t     -> emit equilib y
		 * .delta -> sigs equilib
		*/
		float6 result;
		unsigned int N = 1000;
		thrust::default_random_engine rng;
		rng.seed((int)seed);
		rng.discard(N);
		thrust::uniform_real_distribution<float> u01(0,1);

		// timeratio is real machine turns over per simulation turn 
		float coeffdecaylong  = 1 - (trev / radiationDampParams.y) * timeratio;

		// excitation uses a uniform deviate on [-1:1]
		float coeffexcitelong = radiationDampParams.delta * CUDA_C_F * sqrt(3.) * sqrt(2*(trev / radiationDampParams.y) * timeratio);

		// the damping time is for EMITTANCE, therefore need to multiply by 2
	    float coeffdecayx     = 1 - ((trev /(2 * radiationDampParams.x)) * timeratio);
	    float coeffdecayy     = 1 - ((trev /(2 * radiationDampParams.px)) * timeratio);
	    
	    // exact     coeffgrow= sigperp*sqrt(3.)*sqrt(1-coeffdecay**2)
	    
	    float coeffgrowx       = radiationDampParams.py * sqrt(3.) * sqrt(1 - pow(coeffdecayx,2));
	    float coeffgrowy       = radiationDampParams.t  * sqrt(3.) * sqrt(1 - pow(coeffdecayy,2));

	    if ((radiationDampParams.x < 0.0) || (radiationDampParams.px) < 0.0) 
	   		return particle;
	    else
		   {
		    	result.x     = coeffdecayx * particle.x     + coeffgrowx * (2*u01(rng)-1);
		    	result.px    = coeffdecayx * particle.px    + coeffgrowx * (2*u01(rng)-1);
		    	result.y     = coeffdecayy * particle.y     + coeffgrowy * (2*u01(rng)-1);
		    	result.py    = coeffdecayy * particle.py    + coeffgrowy * (2*u01(rng)-1);
		    	result.t     = particle.t;
		    	result.delta = coeffdecaylong * particle.delta    + coeffexcitelong * (2*u01(rng)-1);
		    	return result;
		    }
	}	
	private:
		float trev,seed;
		float6 radiationDampParams;
		float timeratio;
 };


struct BetatronRoutine{
	BetatronRoutine(float qx, float qy, float ksix, float ksiy, float coupling, float K2L, float K2SL) : qx(qx), qy(qy), ksix(ksix), ksiy(ksiy), coupling(coupling), K2L(K2L), K2SL(K2SL) {}
	__host__ __device__ float6 operator()(float6 particle)
	{
		float6 result;

		float psix = 2 * CUDA_PI_F * qx;
    	float psiy = 2 * CUDA_PI_F * qy;

		// rotation in x-px plane
		float psi1 = psix + particle.delta*ksix;
	    float a11  = cos(psi1);
	    float a12  = sin(psi1);
	   
	    result.x   = particle.x  * a11 + particle.px * a12;
	    result.px  = particle.px * a11 - particle.x  * a12;

	    // rotation in y-py plane
		float psi2 = psiy + particle.delta*ksiy;
	    a11  = cos(psi2);
	    a12  = sin(psi2);

	    result.y   = particle.y  * a11 + particle.py * a12;
	    result.py  = particle.py * a11 - particle.y  * a12;

	    // now have dqmin part - coupling between x and y
        result.px +=  coupling * result.y;
        result.py +=  coupling * result.x;

		// thin sextupole kick 
        result.px +=  0.5 * K2L  * (pow(result.x,2) - pow(result.y,2)) - K2SL * (result.x * result.y);
        result.py +=  0.5 * K2SL * (pow(result.x,2) - pow(result.y,2)) + K2L * (result.x * result.y); 

        return result;
	}
 	private:
		float qx,qy,ksix,ksiy,coupling,K2L, K2SL;
 };

struct denlon2knk{

	thrust::device_vector<float> denlon;
	float tauhat;
	int nbins;

	denlon2knk(thrust::device_vector<float> denlon, float tauhat, int nbins) : denlon(denlon), tauhat(tauhat), nbins(nbins) {}
	float operator()( float6 particle )
	{
		// bin the particle longitudinally
		float bin;
		int nk;
		float dtsamp2 = 2*tauhat/ nbins;
		bin = (particle.t + tauhat) / dtsamp2;
		nk = (int)(bin + 0.5f);
		return denlon[nk];
	}
 };

struct multiplyFloat6Float{

	__host__ __device__
  	float6 operator()(const float6& a, const float& b) const {  
  		float6 result;

  		result.x     = a.x * b;
  		result.px    = a.px * b;
  		result.y     = a.y * b;
  		result.py    = a.py * b;
  		result.t     = a.t * b;
  		result.delta = a.delta * b;

  		return result;
  	}
 };

struct multiplyFloat6Float6 {
	__host__ __device__
  	float6 operator()(const float6& a, const float6& b) const {  
  		float6 result;

  		result.x     = a.x * b.x;
  		result.px    = a.px * b.px;
  		result.y     = a.y * b.y;
  		result.py    = a.py * b.py;
  		result.t     = a.t * b.t;
  		result.delta = a.delta * b.delta;

  		return result;
  	}
 };

struct generatefloat6triplegauss{	

	__host__ __device__
	float6 operator()(const float2& a, const float2& b) const{
		float6 result;

		result.x = 0.0;
		result.px = a.x;
		result.y = 0.0;
		result.py = a.y;
		result.t = 0.0;
		result.delta = b.x;

		return result;
	}

 };

struct ibsRoutine{

   float4 ibscoeff;
   float* denlon2k;
   ibsparameters ibsparms;

   ibsRoutine(float4 ibscoeff, float* denlon2k, ibsparameters& ibsparms): ibscoeff(ibscoeff), denlon2k(denlon2k), ibsparms(ibsparms) {}
   
   __host__ __device__
   float6 operator()(float6 particle) const
   {
   		unsigned int N = 1000;
   		float6 out;
   		float r1,r2,amp,facc,grv1, grv2;
   		thrust::default_random_engine rng;
		rng.seed((int)ibsparms.seed);
		rng.discard(N);
		thrust::uniform_real_distribution<float> u01(0,1);
		// boost::function<int (float6& data, float2 params)> f;


		do 
		{
			r1 = 2*u01(rng)-1;
		    r2 = 2*u01(rng)-1;
		    amp = r1*r1+r2*r2;
		    facc = sqrt(-2.*log(amp)/amp);
		    grv1 = r1*facc/1.73205;
         	grv2 = r2*facc/1.73205;
		}
		while ((amp >= 1.0) || (amp <= 1.0e-8));

		// bin the particle longitudinally
		float bin;
		int nk;
		float dtsamp2 = 2*ibsparms.tauhat/ ibsparms.nbins;
		bin = (particle.t + ibsparms.tauhat) / dtsamp2;
		nk = (int)(bin + 0.5f);
		// thrust::device_vector<float>  linedens(denlon2k);
		// thrust::host_vector<float6> dparticle(1);
		// thrust::fill(dparticle.begin(),dparticle.end(), particle);

		// thrust::host_vector<int> timecomponent(1);
		// thrust::transform(dparticle.begin(),dparticle.end(),thrust::make_constant_iterator(make_float2(ibsparms.tauhat,ibsparms.nbins)),
		// 	timecomponent.begin(),ParticleTimesToInteger());

		// thrust::host_vector<int>::iterator begin = timecomponent.begin();
		// int nk = *begin;

		// get the corresponding element from denlon2k
		//thrust::host_vector<float>::iterator itdenlon2k; // = denlon2k.begin() + nk;
		// (*itdenlon2k)
		// thrust::device_vector<float> data(denlon2k);
		float point = denlon2k[nk];
		float dy =  point * ibscoeff.y * grv1;
        float dp = point * ibscoeff.z * grv2;


        do 
		{
			r1 = 2*u01(rng)-1;
		    r2 = 2*u01(rng)-1;
		    amp = r1*r1+r2*r2;
		    facc = sqrt(-2.*log(amp)/amp);
		    grv1 = r1*facc/1.73205;
         	grv2 = r2*facc/1.73205;
		}
		while ((amp >= 1.0) || (amp <= 1.0e-8));

		float dx = point * ibscoeff.x * grv1;

		out.x = particle.x;
		out.y = particle.y;
		out.t = particle.t;

		out.px = particle.px + dx;
		out.py = particle.py + dy;
		out.delta = particle.delta + dp; 	


		return out;

   	};


 };



// function to remove particles outside of rf bucket after rf update
struct isInLong{
	isInLong(float tauhat, float synctime): tauhat(tauhat), synctime(synctime) {}
	__host__ __device__ 
	bool  operator()(const float6 particle)
	{
		return !((abs(particle.t - synctime) < tauhat ) || (tauhat == 0.0));
	}
	private:
		float tauhat;
		float synctime;
 };

// to write 6-vector to screen
// __host__ std::ostream& operator<< (std::ostream& os, const float6& p)
// {
// 	os << std::setw(15) << p.x << std::setw(15) << p.px << std::setw(15) << p.y 
// 	<< std::setw(15) << p.py << std::setw(15) << p.t <<std::setw(15) << p.delta << std::endl;;
// 	return os;
// };

class STE_Bunch
{
public:
	STE_TFS* tfs;
	STE_Radiation* rad;
	STE_Longitudinal_Hamiltonian* ham;
	STE_Synchrotron* syn;	
	STE_IBS* ibs;


	/* ************************************************************ */
	/* 																*/
	/*  					constructor          					*/
	/* 																*/
	/* ************************************************************ */
	STE_Bunch( bunchparameters& );
	// ~STE_Bunch();

	/* ************************************************************ */
	/* 																*/
	/*  					helper functions      					*/
	/* 																*/
	/* ************************************************************ */

	std::string get_date();
	float6 calculateEmittance( thrust::device_vector<float6>, bunchparameters&  );
	
	/* ************************************************************ */
	/* 																*/
	/*  					set functions      		     			*/
	/* 																*/
	/* ************************************************************ */

	void setEmittance( bunchparameters& );
	void resetDebunchLosses();
	void setIntensity( bunchparameters&  , int  );

	/* ************************************************************ */
	/* 																*/
	/*  					get functions      		     			*/
	/* 																*/
	/* ************************************************************ */

	bunchparameters getparams();

	int getBucketNumber();

	float3 getIBSGrowthRates();
	float6 getEmittance();

	std::map<int, float3> getIBSLifeTimes();

	/* ************************************************************ */
	/* 																*/
	/*  					physcis routines    					*/
	/* 																*/
	/* ************************************************************ */
	
	void RadiationRoutine( bunchparameters& );
	void BetaRoutine( bunchparameters& );
	void IBSRoutine( bunchparameters& );
	void RFRoutine( bunchparameters&  , float );

	/* ************************************************************ */
	/* 																*/
	/*  					updating functions  					*/
	/* 																*/
	/* ************************************************************ */
	
	void updateIBSLifeTimes( int  , float3 );
	void updatedIntensity( bunchparameters& , int );
	void updateEmittance( int );
	void updateRadiation();
	void updateBetaTron();
	void updateIBS( int );
	void updateRF( int , int );

	/* ******************************************* */
	/* main routine to be called from main program */
	void updateBunch( int  , int  , int  , int  , int , int, int );
	/* ******************************************* */

	/* ************************************************************ */
	/* 																*/
	/*  					printing to screen functions     		*/
	/* 																*/
	/* ************************************************************ */

	void printBunchParams();
	void printDistribution();
	void printEmittance( int );
	void printHistogramTime();
	void printSqrtHistogram();


	void writeDistributionToFile( int );
	void writeEmittanceToFile();
	void writeIBSLifeTimes();

	/* ************************************************************ */
	/* 																*/
	/*  					printing to file functions     		    */
	/* 																*/
	/* ************************************************************ */

	

	

	

private:

	void Init( std::string );

	void ReadBunchFile( std::string );
	float CalculatedEnergySpread();
	
	int bucketNumber;
	int debunchlosses;

	thrust::device_vector<float6> distribution;
	float synchronousPhase;
	float synchronousPhaseNext;
	float synchrotronTune;
	float hamMax;
	float energyLostPerTurn;
	float tauhat;
	float initEmitx;
	float initEmity;
	float initSigs;
	float initSigE;
	float current; 
	float6 emittance;

	bunchparameters bparams;

	//simulation output - write file at end of simulation
	std::map<int, float3> IBSLifeTimes;
	std::map<int, float6> mapEmittances;
	std::map<int, float6> mapIntensity; 
	
};

#endif