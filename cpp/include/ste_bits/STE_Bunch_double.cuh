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
#include <thrust/iterator/zip_iterator.h>

#include <iterator>
#include <sstream>
#include <fstream>
#include <algorithm>

// load double 6 datastructure
#include "STE_DataStructures_double.cuh"

// load tfsTableData datastructure
#include "STE_TFS_double.cuh"

#include "STE_Radiation_double.cuh"
#include "STE_Longitudinal_Hamiltonian_double.cuh"
#include "STE_Synchrotron_double.cuh"
#include "STE_Random_double.cuh"

// load ibs
#include "STE_IBS_double.cuh"

// struct double6 {
// 	double x;
// 	double px;
// 	double y;
// 	double py;
// 	double t;
// 	double delta;
// };

struct bunchparameters {
	int bucket;
	double tunex;
	double tuney;
	double ksix;
	double ksiy;
	double trev;
	double timeratio;
	std::string particleType;
	int particleAtomNumber;
	int particleCharge;
	int methodRadiationIntegrals;
	std::string tfsfile;
	int numberMacroParticles;
	int methodLongitudinalDist;
	double realNumberOfParticles;
	double k2l;
	double k2sl;
	double betatroncoupling;
	double conversion;

	radiationIntegralsParameters radparams;
	hamiltonianParameters hamparams;
	synchrotronParameters synparams;
	longitudinalParameters longparams;
	ibsparameters ibsparams;

 };


struct randomNumber{
	__host__ __device__ double operator()(int idum)
	{
		#define MBIG 1000000000
		#define MSEED 161803398
		#define MZ 0
		#define FAC (1.0/MBIG)

		// double ran3(int idum)
		//int *idum;
		// {
			static int inext,inextp;
			static long ma[56];
			static int iff=0;
			long mj,mk;
			int i,ii,k;

			if (idum < 0 || iff == 0) {
				iff=1;
				mj=MSEED-(idum < 0 ? -idum : idum);
				mj %= MBIG;
				ma[55]=mj;
				mk=1;
				for (i=1;i<=54;i++) {
					ii=(21*i) % 55;
					ma[ii]=mk;
					mk=mj-mk;
					if (mk < MZ) mk += MBIG;
					mj=ma[ii];
				}
				for (k=1;k<=4;k++)
					for (i=1;i<=55;i++) {
						ma[i] -= ma[1+(i+30) % 55];
						if (ma[i] < MZ) ma[i] += MBIG;
					}
				inext=0;
				inextp=31;
				idum=1;
			}
			if (++inext == 56) inext=1;
			if (++inextp == 56) inextp=1;
			mj=ma[inext]-ma[inextp];
			if (mj < MZ) mj += MBIG;
			ma[inext]=mj;
			return mj*FAC;
		// };

		#undef MBIG
		#undef MSEED
		#undef MZ
		#undef FAC
	}
};




struct GenRand
{
	int seed;
	GenRand(int seed) : seed(seed) {}
    __device__
    double6 operator () (int idx)
    {
    	double6 result;
		unsigned int N = 100;
		thrust::default_random_engine rng;
		rng.seed((int)seed);
		rng.discard(N * idx);
		thrust::uniform_real_distribution<double> u01(0,1);
		// randomNumber rn = ran;
		result.x      = (2*u01(rng)-1);
		result.px     = (2*u01(rng)-1);
		result.y      = (2*u01(rng)-1);
		result.py     = (2*u01(rng)-1);
		result.t      = (2*u01(rng)-1);
		result.delta  = (2*u01(rng)-1);

        return result;
    }
};

struct ibsRoutineNew{
   double4 ibscoeff;
   double* denlon2k;
   ibsparameters ibsparms;
   double tsynchro;

   ibsRoutineNew( double4 ibscoeff, double* denlon2k, ibsparameters& ibsparms , double tsynchro): ibscoeff(ibscoeff), 
   denlon2k(denlon2k), ibsparms(ibsparms), tsynchro(tsynchro) {}
   
   __host__ __device__
   double6 operator()(double6& particle, double6& R) 
   {
   		double6 out;
		double bin;
		int nk;

		double dtsamp2 = 2*ibsparms.tauhat/ ibsparms.nbins;

		bin = (particle.t - tsynchro + ibsparms.tauhat) / dtsamp2;
		nk  = (int)(bin + 0.5f);

		double point = denlon2k[nk];
		double dy    = point * ibscoeff.y * R.px;
	    double dp    = point * ibscoeff.z * R.py;
	    double dx    = point * ibscoeff.x * R.delta;

		out.x     = particle.x;
		out.y     = particle.y;
		out.t     = particle.t;

		out.px    = particle.px + dx;
		out.py    = particle.py + dy;
		out.delta = particle.delta + dp; 	


		return out;
 };
};


struct GenRandIBS{
	int seed;
	GenRandIBS(int seed) : seed(seed) {}

    __device__
    double6 operator () (int idx)
    {
    	double6 result;
    	double r1,r2,amp,facc,grv1,grv2;

		unsigned int N = 100;
		thrust::default_random_engine rng;
		rng.seed((int)seed);
		rng.discard(N * idx);
		thrust::uniform_real_distribution<double> u01(0,1);
		
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

		result.x      = 0.0;
		result.px     = grv1;
		result.y      = 0.0;
		result.py     = grv2;

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

		result.t      = 0.0;
		result.delta  = grv1;

        return result;
    }
 };


struct PDER{
	__device__ double6 operator()(double6& P, double6& D , double6& E, double6& R) const {
		double6 out;

		out.x     = P.x * D.x + E.x * R.x;
		out.px    = P.px * D.px + E.px * R.px;
		out.y     = P.y * D.y + E.y * R.y;
		out.py    = P.py * D.py + E.py * R.py;
		out.t     = P.t * D.t + E.t * R.t;
		out.delta = P.delta * D.delta + E.delta * R.delta;

		return out;
		// thrust::get<0>(t).x     = thrust::get<0>(t).x * thrust::get<1>(t).x + thrust::get<2>(t).x * thrust::get<4>(t).x;
		// thrust::get<0>(t).px    = thrust::get<0>(t).px * thrust::get<1>(t).px + thrust::get<2>(t).px * thrust::get<4>(t).px;
		// thrust::get<0>(t).y     = thrust::get<0>(t).y * thrust::get<1>(t).y + thrust::get<2>(t).y * thrust::get<4>(t).y;
		// thrust::get<0>(t).py    = thrust::get<0>(t).py * thrust::get<1>(t).py + thrust::get<2>(t).py * thrust::get<4>(t).py;
		// thrust::get<0>(t).t     = thrust::get<0>(t).t * thrust::get<1>(t).t + thrust::get<2>(t).t * thrust::get<4>(t).t;
		// thrust::get<0>(t).delta = thrust::get<0>(t).delta * thrust::get<1>(t).delta + thrust::get<2>(t).delta * thrust::get<4>(t).delta;

	}
 };

struct blowupfunctor{
	double6 multi;
	blowupfunctor( double6 multi): multi(multi) {}
	__host__ __device__ double6 operator()(double6 particle)
	{
		double6 result;
		result.x     = particle.x     * multi.x;
		result.px    = particle.px    * multi.px;
		result.y     = particle.y     * multi.y;
		result.py    = particle.py    * multi.py;
		result.t     = particle.t     * multi.t;
		result.delta = particle.delta * multi.delta;
		return result;
	}
};

struct RadiationDampingRoutineFunctor{
	RadiationDampingRoutineFunctor(double6 radiationDampParams, double trev, double timeratio, int seed) : radiationDampParams(radiationDampParams), 
	trev(trev) ,timeratio(timeratio), seed(seed) {}
	__host__ __device__ double6 operator()(double6 particle)
	{
		/*
		 * .x     -> t_emitx
		 * .px    -> t_emity
		 * .y     -> t_sigs
		 * .py    -> emit equilib x
		 * .t     -> emit equilib y
		 * .delta -> sigs equilib
		*/
		double6 result;
		unsigned int N = 1000;
		thrust::default_random_engine rng;
		rng.seed((int)seed);
		rng.discard(N);
		thrust::uniform_real_distribution<double> u01(0,1);
		// randomNumber rn = ran;

		// timeratio is real machine turns over per simulation turn 
		double coeffdecaylong  = 1 - (trev / radiationDampParams.y) * timeratio;

		// excitation uses a uniform deviate on [-1:1]
		double coeffexcitelong = radiationDampParams.delta * CUDA_C_F * sqrt(3.) * sqrt(2*(trev / radiationDampParams.y) * timeratio);

		// the damping time is for EMITTANCE, therefore need to multiply by 2
	    double coeffdecayx     = 1 - ((trev /(2 * radiationDampParams.x)) * timeratio);
	    double coeffdecayy     = 1 - ((trev /(2 * radiationDampParams.px)) * timeratio);
	    
	    // exact     coeffgrow= sigperp*sqrt(3.)*sqrt(1-coeffdecay**2)
	    
	    double coeffgrowx       = radiationDampParams.py * sqrt(3.) * sqrt(1 - pow(coeffdecayx,2));
	    double coeffgrowy       = radiationDampParams.t  * sqrt(3.) * sqrt(1 - pow(coeffdecayy,2));

	    if ((radiationDampParams.x < 0.0) || (radiationDampParams.px) < 0.0) 
	   		return particle;
	    else
		   {
		    	result.x     = coeffdecayx * particle.x     + coeffgrowx * ran.x; //(2*rn((int)seed)-1); // (2*u01(rng)-1);
		    	result.px    = coeffdecayx * particle.px    + coeffgrowx * ran.px; //(2*rn((int)seed)-1); //(2*u01(rng)-1);
		    	result.y     = coeffdecayy * particle.y     + coeffgrowy * ran.y; //(2*rn((int)seed)-1); //(2*u01(rng)-1);
		    	result.py    = coeffdecayy * particle.py    + coeffgrowy * ran.py; //(2*rn((int)seed)-1); //(2*u01(rng)-1);
		    	result.t     = particle.t;//
		    	result.delta = coeffdecaylong * particle.delta   + coeffexcitelong * ran.delta;
		    	return result;
		    }
	}	
	private:
		double trev,seed;
		double6 radiationDampParams;
		double timeratio;
		double6 ran;
 };


struct BetatronRoutine{
	BetatronRoutine(double qx, double qy, double ksix, double ksiy, double coupling, double K2L, double K2SL) : qx(qx), qy(qy), ksix(ksix), ksiy(ksiy), coupling(coupling), K2L(K2L), K2SL(K2SL) {}
	__host__ __device__ double6 operator()(double6 particle)
	{
		double6 result;

		double psix = 2 * CUDA_PI_F * qx;
    	double psiy = 2 * CUDA_PI_F * qy;

		// rotation in x-px plane
		double psi1 = psix ;//+ particle.delta*ksix;
	    double a11  = cos(psi1);
	    double a12  = sin(psi1);

	   
	    result.x   = particle.x  * a11 + particle.px * a12;
	    result.px  = particle.px * a11 - particle.x  * a12;

	    // rotation in y-py plane
		double psi2 = psiy; //+ particle.delta*ksiy;
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

        result.t = particle.t;
        result.delta = particle.delta;
        return result;
	}
 	private:
		double qx,qy,ksix,ksiy,coupling,K2L, K2SL;
 };

struct denlon2knk{

	thrust::device_vector<double> denlon;
	double tauhat;
	int nbins;

	denlon2knk(thrust::device_vector<double> denlon, double tauhat, int nbins) : denlon(denlon), tauhat(tauhat), nbins(nbins) {}
	double operator()( double6 particle )
	{
		// bin the particle longitudinally
		double bin;
		int nk;
		double dtsamp2 = 2*tauhat/ nbins;
		bin = (particle.t + tauhat) / dtsamp2;
		nk = (int)(bin + 0.5f);
		return denlon[nk];
	}
 };

struct multiplydouble6double{

	__host__ __device__
  	double6 operator()(const double6& a, const double& b) const {  
  		double6 result;

  		result.x     = a.x * b;
  		result.px    = a.px * b;
  		result.y     = a.y * b;
  		result.py    = a.py * b;
  		result.t     = a.t * b;
  		result.delta = a.delta * b;

  		return result;
  	}
 };

struct multiplydouble6double6 {
	__host__ __device__
  	double6 operator()(const double6& a, const double6& b) const {  
  		double6 result;

  		result.x     = a.x * b.x;
  		result.px    = a.px * b.px;
  		result.y     = a.y * b.y;
  		result.py    = a.py * b.py;
  		result.t     = a.t * b.t;
  		result.delta = a.delta * b.delta;

  		return result;
  	}
 };

struct generatedouble6triplegauss{	

	__host__ __device__
	double6 operator()(const double2& a, const double2& b) const{
		double6 result;

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

   double4 ibscoeff;
   double* denlon2k;
   ibsparameters ibsparms;
  

   ibsRoutine(double4 ibscoeff, double* denlon2k, ibsparameters& ibsparms): 
   ibscoeff(ibscoeff), denlon2k(denlon2k), ibsparms(ibsparms) {}
   
   __host__ __device__
   double6 operator()(double6 particle) const
   {
   		unsigned int N = 1000;
   		double6 out;
   		double r1,r2,amp,facc,grv1, grv2;
   		thrust::default_random_engine rng;
		rng.seed((int)ibsparms.seed);
		rng.discard(N);
		thrust::uniform_real_distribution<double> u01(0,1);
		boost::function<int (double6& data, double2 params)> f;

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
		double bin;
		int nk;
		double dtsamp2 = 2*ibsparms.tauhat/ ibsparms.nbins;
		bin = (particle.t + ibsparms.tauhat) / dtsamp2;
		nk = (int)(bin + 0.5f);
		// thrust::device_vector<double>  linedens(denlon2k);
		// thrust::host_vector<double6> dparticle(1);
		// thrust::fill(dparticle.begin(),dparticle.end(), particle);

		// thrust::host_vector<int> timecomponent(1);
		// thrust::transform(dparticle.begin(),dparticle.end(),thrust::make_constant_iterator(make_double2(ibsparms.tauhat,ibsparms.nbins)),
		// 	timecomponent.begin(),ParticleTimesToInteger());

		// thrust::host_vector<int>::iterator begin = timecomponent.begin();
		// int nk = *begin;

		// get the corresponding element from denlon2k
		//thrust::host_vector<double>::iterator itdenlon2k; // = denlon2k.begin() + nk;
		// (*itdenlon2k)
		// thrust::device_vector<double> data(denlon2k);
		double point = denlon2k[nk];
		double dy =  point * ibscoeff.y * grv1;
        double dp = point * ibscoeff.z * grv2;


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

		double dx = point * ibscoeff.x * grv1;

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
	isInLong(double tauhat, double synctime): tauhat(tauhat), synctime(synctime) {}
	__host__ __device__ 
	bool  operator()(const double6 particle)
	{
		return !((abs(particle.t - synctime) < tauhat ) || (tauhat == 0.0));
	}
	private:
		double tauhat;
		double synctime;
 };

// to write 6-vector to screen
// __host__ std::ostream& operator<< (std::ostream& os, const double6& p)
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
	STE_Bunch( bunchparameters& , int );
	// ~STE_Bunch();

	/* ************************************************************ */
	/* 																*/
	/*  					helper functions      					*/
	/* 																*/
	/* ************************************************************ */

	std::string get_date();
	double6 calculateEmittance( thrust::device_vector<double6>, bunchparameters&  );
	
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

	double3 getIBSGrowthRates();
	double6 getEmittance();

	std::map<int, double3> getIBSLifeTimes();

	/* ************************************************************ */
	/* 																*/
	/*  					physcis routines    					*/
	/* 																*/
	/* ************************************************************ */
	
	void BlowUp( thrust::device_vector<double6>& , double6 ,  int );
	void RadiationRoutine( bunchparameters& , int );
	void BetaRoutine( bunchparameters& );
	void IBSRoutine( bunchparameters& , int );
	void RFRoutine( bunchparameters&  , double );
	void Radiation( thrust::device_vector<double6>& , double6 ,  int );

	/* ************************************************************ */
	/* 																*/
	/*  					updating functions  					*/
	/* 																*/
	/* ************************************************************ */
	
	void updateIBSLifeTimes( int  , double3 );
	void updatedIntensity( bunchparameters& , int );
	void updateEmittance( int );
	void updateRadiation( int );
	void updateBetaTron();
	void updateIBS( int );
	void updateRF( int , double );

	/* ******************************************* */
	/* main routine to be called from main program */
	void updateBunch( int  , int  , int  , int  , int , double, int , double , double );
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
	void writeIntensity();

	/* ************************************************************ */
	/* 																*/
	/*  					printing to file functions     		    */
	/* 																*/
	/* ************************************************************ */

	

	

	

private:

	void Init( std::string );

	void ReadBunchFile( std::string );
	double CalculatedEnergySpread();
	
	int bucketNumber;
	int debunchlosses;
	int lastTurn;

	thrust::device_vector<double6> distribution;
	double synchronousPhase;
	double synchronousPhaseNext;
	double synchrotronTune;
	double hamMax;
	double energyLostPerTurn;
	double tauhat;
	double initEmitx;
	double initEmity;
	double initSigs;
	double initSigE;
	double current; 
	double6 emittance;
	double6 RadDecayExitationCoeff;
	randomNumber rn;
	bunchparameters bparams;
	double minharmonic;

	//simulation output - write file at end of simulation
	std::map<int, double3> IBSLifeTimes;
	std::map<int, double6> mapEmittances;
	std::map<int, double6> mapIntensity; 
	
};

#endif