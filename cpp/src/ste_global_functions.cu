// #include "bunch.h"
#include "simParameters.cuh"

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

#include "read_tfs.cu"

#include <vector>
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

#ifndef CUDA_ELECTRON_REST_E_F
#define CUDA_ELECTRON_REST_E_F 0.5109989461e6f // in eV 
#endif

using namespace std;
// using thrust::placeholders;

typedef struct float6 {
	float x;
	float px;
	float y;
	float py;
	float t;
	float delta;
} float6;

typedef struct radiationIntegralsParameters
{
	float betxRingAverage;
	float betyRingAverage;
	float acceleratorLength;
	float gammaTransition;
	float DipoleBendingRadius;
	float ParticleRadius; // nucleon or electron 
	float ParticleEnergy;
	float cq;
	float gammar;
	float eta;
	float omegas;
	float p0;

} radiationIntegralsParameters;



typedef struct radiationIntegrals {
	float I2;
	float I3;
	float I4x;
	float I4y;
	float I5x;
	float I5y;
} radiationIntegrals;

typedef struct longitudinalParameters {
	float seed;
	float betx;
	float bety;
	float emitx;
	float emity;
	float tauhat;
	float sigs;
	float sige;
	float omega0;
	float v0;
	float v1;
	float v2;
	float h0;
	float h1;
	float h2;
	float phiSynchronous;
	float p0;
	float betar;
	float eta;
	float hammax;
	float charge;
} longitudinalParameters;

typedef struct ibsparameters {
	float acceleratorLength;
	float gammaTransition;
	float betx;
	float bety;
	float emitx;
	float emity;
	float dponp;
	float ParticleRadius;
	float betar;
	float gamma0;
	float numberRealParticles;
	float sigs;
	float tauhat;
	float ibsCoupling;
	float fracibstot; // artificially increase or decrease ibs with a scaling factor
	float timeratio;
	float numberMacroParticles;
	float trev;
	float nbins;
	float fmohlNumPoints;
	float seed;
	float coulomblog;
} ibsparameters;

/*
 * RANDOM GENERATOR 
 * -----------------
 * ran3(int seed) -  from numerical recipes book 
 * added host and device option so it is also callable as function in functors 
 * executing either on host or device.
 */

#define MBIG 1000000000
#define MSEED 161803398
#define MZ 0
#define FAC (1.0/MBIG)

__host__ __device__ float ran3(int idum)
{
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
}

#undef MBIG
#undef MSEED
#undef MZ
#undef FAC


// to write radiation integrals to screen
__host__ std::ostream& operator<< (std::ostream& os, const radiationIntegrals& p)
{
	os << "I2  : " << std::setw(15) << p.I2  << endl;
	os << "I3  : " << std::setw(15) << p.I3  << endl;
	os << "I4x : " << std::setw(15) << p.I4x << endl;
	os << "I4y : " << std::setw(15) << p.I4y << endl;
	os << "I5x : " << std::setw(15) << p.I5x << endl;
	os << "I5y : " << std::setw(15) << p.I5y << endl;
	return os;
}


// write tfsdata required
__host__ std::ostream& operator<< (std::ostream& os, const tfsTableData& p)
{
	os << std::setw(12) << p.s << std::setw(12) << p.angle << std::setw(10) << p.l << std::setw(12) << p.k1l << std::setw(10) << p.dx <<std::setw(10) << endl;
	return os;
}

// to write 6-vector to screen
__host__ std::ostream& operator<< (std::ostream& os, const float6& p)
{
	os << std::setw(15) << p.x << std::setw(15) << p.px << std::setw(15) << p.y << std::setw(15) << p.py << std::setw(15) << p.t <<std::setw(15) << p.delta;
	return os;
}

// to write 4-vector to screen
__host__ std::ostream& operator<< (std::ostream& os, const float4& p)
{
	os << "["<< p.x << "," << p.y << "," << p.z <<"," << p.w <<"]";
	return os;
}

// to write 2-vector to screen
__host__ std::ostream& operator<< (std::ostream& os, const float2& p)
{
	os << std::setw(21) << p.x << std::setw(21) << p.y;
	// os << printf("%.16f",p.x) << "\t" << printf("%.16f",p.y) << std::endl;
	// os << printf("%.16f \t %.16f\n",p.x,p.y);
	return os;
}


/*
 * PARALLEL RANDOM GENERATOR 
 * -----------------
 * randxor_functor(int seed) 
*/
template <typename T>
struct randxor_functor
{
	int seed;
	randxor_functor(int _seed): seed(_seed) {}

	__host__ __device__
	T operator()(unsigned int thread_id) const
	{
		unsigned int N = 1000; // samples per stream
		thrust::default_random_engine rng;
		rng.seed(seed);
		rng.discard(N*thread_id);
		thrust::uniform_real_distribution<float> u01(0,1);
		// thrust::xor_combine_engine<thrust::minstd_rand,0,thrust::minstd_rand0,0> rng;
		//float x = (float)u01(rng);
		return u01(rng);
		
		// return (float)rng()/(float)rng.max;
	}
};

/*
 * PARALLEL RANDOM GENERATOR BI-GAUSSIAN IN 2D 
 * -------------------------------------------
 * rand_2d_gauss(float3 in)  
 * generates bi-gaussian in 2D 
*/
template <typename T>
struct rand_2d_gauss
{
	float3 in;
	rand_2d_gauss(float3 in): in(in)  {}

	__host__ __device__
	float2 operator()(unsigned int thread_id) const
	{
		unsigned int N = 1000; // samples per stream
		float ampx, amp, r1, r2, facc; // variables for generating bi-gaussian

		thrust::default_random_engine rng;
		rng.seed((int)in.x);
		rng.discard(N*thread_id);
		thrust::uniform_real_distribution<float> u01(0,1);

		ampx = sqrt(in.y*in.z);
		do 
    	{
        	r1 = 2*u01(rng)-1;
	        r2 = 2*u01(rng)-1;
	        amp = r1*r1+r2*r2;
	    } 
    	while ((amp >=1) || (amp<=3.e-9));

		facc = sqrt(-2*log(amp)/amp);

		
		// thrust::xor_combine_engine<thrust::minstd_rand,0,thrust::minstd_rand0,0> rng;
		//float x = (float)u01(rng);
		return make_float2(ampx*r1*facc,ampx*r2*facc);
		
		// return (float)rng()/(float)rng.max;
	}
};

__host__ __device__ float tcoeff(float eta, float angularFreq, float h){
	return  (angularFreq * eta *h);
};


template <typename T>
struct rand_6d_gauss
{
	longitudinalParameters in;
	rand_6d_gauss(longitudinalParameters in): in(in) {}

	__host__ __device__
	float6 operator()(unsigned int thread_id) const
	{
		unsigned int N = 1000;
		float ampx, ampy, amp, r1,r2, facc;
		float ampt,ham, tc, pc;
		float kinetic, potential1, potential2, potential3,ts;
		float6 out;

		thrust::default_random_engine rng;
		rng.seed((int)in.seed);
		rng.discard(N*thread_id);
		thrust::uniform_real_distribution<float> u01(0,1);

		// *************** x-px phase-space ********************
		// sig = sqrt(emit * beta)
		ampx = sqrt(in.betx*in.emitx);
		do 
    	{
        	r1 = 2*u01(rng)-1;
	        r2 = 2*u01(rng)-1;
	        amp = r1*r1+r2*r2;
	    } 
    	while ((amp >=1) || (amp<=3.e-9));

		facc = sqrt(-2*log(amp)/amp);

		out.x = ampx*r1*facc;
		out.px = ampx*r2*facc;


		// *************** y-py phase-space ***************************
		// sig = sqrt(emit * beta)
		ampy = sqrt(in.bety*in.emity);
		do 
    	{
        	r1 = 2*u01(rng)-1;
	        r2 = 2*u01(rng)-1;
	        amp = r1*r1+r2*r2;
	    } 
    	while ((amp >=1) || (amp<=3.e-9));

		facc = sqrt(-2*log(amp)/amp);

		out.y = ampy*r1*facc;
		out.py = ampy*r2*facc;

		ampt = in.sigs / CUDA_C_F;
		ts = in.phiSynchronous / (in.h0 * in.omega0); // distribution has to be around phis

		do 
    	{
        	r1 = 2*u01(rng)-1;
	        r2 = 2*u01(rng)-1;
	        amp = r1*r1+r2*r2;
	        if (amp >=1) continue;

	        facc = sqrt(-2*log(amp)/amp);
	        out.t = ts + ampt * r1 * facc;
	        if  (abs(out.t-ts)>=in.tauhat) continue;

	        out.delta = in.sige * r2 * facc;
	        tc = (in.omega0 * in.eta * in.h0);
	        pc = (in.omega0 * in.charge) / (2 * CUDA_PI_F * in.p0 * in.betar);

			kinetic = 0.5 * tc * pow(out.delta,2);

			potential1 = pc * in.v0 * (cos(in.h0 * in.omega0 * out.t) - cos(in.phiSynchronous) + (in.h0 * in.omega0 * out.t - in.phiSynchronous) * sin(in.phiSynchronous));
 			potential2 = pc * in.v1 * (in.h0 / in.h1) * (cos(in.h1 * in.omega0 * out.t) - cos(in.h1 * in.phiSynchronous / in.h0) + 
 				(in.h1 * in.omega0 * out.t - in.h1 * in.phiSynchronous / in.h0) * sin(in.h1 * in.phiSynchronous / in.h0));
			potential3 = pc * in.v2 * (in.h0 / in.h2) * (cos(in.h2 * in.omega0 * out.t) - cos(in.h2 * in.phiSynchronous / in.h0) + 
				(in.h2 * in.omega0 * out.t - in.h2 * in.phiSynchronous / in.h0) * sin(in.h2 * in.phiSynchronous / in.h0));
			ham = kinetic + potential1 + potential2 + potential3;
	    } 
    	while ((amp >=1)|| (ham>in.hammax) || (abs(out.t-ts)>=in.tauhat)); // || (ham>in.hammax) || (abs(out.t)>=in.tauhat));

		// phidelta = make_float2(0.0,0.0);
		// thrust::default_random_engine rng;
		// rng.seed((int)in.seed);
		// rng.discard(N*thread_id);
		// thrust::uniform_real_distribution<float> u01(0,1);
		// out.phi = 0.0;
		// out.delta = 0.0;

		return out;
	}
};


/*
 * PCOEFF is the coefficient in front of the goniometric functions (potential) in the Hamiltonian
 * REF : Lee (Third Ed.) page 233 eq. 3.13
 *
*/
__host__ __device__ float pcoeff(float voltage, float angularFreq, float p0, float betaRelativistic, float charge){
	return  (angularFreq * voltage* charge) / (2 * CUDA_PI_F * p0 * betaRelativistic);
};

/*
 * Triple RF Hamiltonian
 *
 * H = 0.5 * tcoeff * delta**2 + pcoeff[v_i,omega0,p0,betarelativistic](cos(h_i omega0 t)- cos(h_i/h_0 phis) + (h_i omega0 t- h_i/h_0 phis) sin(h_i/h_0 phis))
 *
 * REF : Lee (Third Ed.) page 233 eq. 3.13
 *
*/
__host__ __device__ float HamiltonianTripleRf(float tcoeff, float3 voltages, float3 harmonicNumbers, float phiSynchronous, float t, 
	float delta , float omega0, float p0, float betaRelativistic, float charge){
	float kinetic, potential1, potential2, potential3;
	kinetic = 0.5 * tcoeff * pow(delta,2);

	potential1 = pcoeff(voltages.x, omega0, p0, betaRelativistic, charge) * (cos(harmonicNumbers.x * omega0 * t) - cos(phiSynchronous) + 
		(harmonicNumbers.x * omega0 * t - phiSynchronous) * sin(phiSynchronous));

	potential2 = pcoeff(voltages.y, omega0, p0, betaRelativistic, charge) * (harmonicNumbers.x / harmonicNumbers.y) * (cos(harmonicNumbers.y * omega0 * t) - 
		cos(harmonicNumbers.y * phiSynchronous / harmonicNumbers.x) + 
		(harmonicNumbers.y * omega0 * t - harmonicNumbers.y * phiSynchronous / harmonicNumbers.x) * sin(harmonicNumbers.y * phiSynchronous / harmonicNumbers.x));

	potential3 = pcoeff(voltages.z, omega0, p0, betaRelativistic, charge) * (harmonicNumbers.x / harmonicNumbers.z) * (cos(harmonicNumbers.z * omega0 * t) - 
		cos(harmonicNumbers.z * phiSynchronous / harmonicNumbers.x) + 
		(harmonicNumbers.z * omega0 * t - harmonicNumbers.z * phiSynchronous / harmonicNumbers.x) * sin(harmonicNumbers.z * phiSynchronous / harmonicNumbers.x));

	return kinetic + potential1 + potential2 + potential3;
};


__host__ __device__ float VoltageTripleRf(float phi, float3 voltages, float3 harmonicNumbers){
	float volt1, volt2, volt3;
	volt1 =  voltages.x * sin(phi);
	volt2 =  voltages.y * sin((harmonicNumbers.y / harmonicNumbers.x) * phi);
	volt3 =  voltages.z * sin((harmonicNumbers.z / harmonicNumbers.x) * phi);
	return volt1 + volt2 + volt3;
}; 
 
/*
 * Functor returning Triple RF voltage in function of phase phi and it's derivative
 * This is needed to find the synchronous phases using Newton-Raphson root finding algorithm
 * from the boost::math::tools library
*/
template <class T>
struct synchronousPhaseFunctor
{
   synchronousPhaseFunctor(T const& target,float3 voltages, float3 harmonicNumbers, float charge) : U0(target),volts(voltages), hs(harmonicNumbers), ch(charge) {}
   __host__ __device__ thrust::tuple<T, T> operator()(T const& phi)
   {
   	T volt1, volt2, volt3;
   	T dvolt1, dvolt2, dvolt3;
	volt1 =  ch * volts.x * sin(phi);
	volt2 =  ch * volts.y * sin((hs.y / hs.x) * phi);
	volt3 =  ch * volts.z * sin((hs.z / hs.x) * phi);

	dvolt1 =  ch * volts.x * cos(phi);
	dvolt2 =  ch * volts.y * (hs.y / hs.x) * cos((hs.y / hs.x) * phi);
	dvolt3 =  ch * volts.z * (hs.z / hs.x) * cos((hs.z / hs.x) * phi);
    return thrust::make_tuple(volt1 + volt2 + volt3  - U0, dvolt1 + dvolt2 + dvolt3);
   }
private:
   T U0;
   float3 volts;
   float3 hs;
   float ch;
};

/*
 * Root finding using Newton-Rahpson from boost::math::tools library 
 * The root corresponds to the synchronous phase given the energy lost per turn per particle (due to radiation damping, hence synchronous phase
 * differs from voltage zero crossing => accelerating phase)
 *
 * usage :
 * -------
 * double r = cbrt_deriv(energylost,make_float3(-1.4e6,-20.0e6,-17.14e6),make_float3(400,1200,1400),0.0,-1.0,1.0);
 * 
 * Note for using floats
 * ---------------------
 * float energylost = 170000;
 * float r = cbrt_deriv(energylost,make_float3(-1.4e6,-20.0e6,-17.14e6),make_float3(400,1200,1400),(float)0.0,(float)-1.0,(float)1.0);
*/
template <class T>
T synchronousPhaseFunctorDeriv(T x,float3 voltages, float3 harmnumbers, float charge, T guess, T min, T max)
{ 
  // return cube root of x using 1st derivative and Newton_Raphson.
  using namespace boost::math::tools;
 
  const int digits = std::numeric_limits<T>::digits;  // Maximum possible binary digits accuracy for type T.
  int get_digits = static_cast<int>(digits * 0.6);    // Accuracy doubles with each step, so stop when we have
                                                      // just over half the digits correct.
  const boost::uintmax_t maxit = 20;
  boost::uintmax_t it = maxit;
  T result = newton_raphson_iterate(synchronousPhaseFunctor<T>(x,voltages,harmnumbers,charge), guess, min, max, get_digits, it);
  return result;
}


template <typename T>
struct rand6DTransverseBiGaussLongitudinalMatched
{
	longitudinalParameters in;
	rand6DTransverseBiGaussLongitudinalMatched(longitudinalParameters in): in(in) {}
	__host__ __device__
	float6 operator()(unsigned int thread_id) const
	{
		unsigned int N = 1000;
		float ampx, ampy, amp, r1,r2, facc,ts;
		float ampt,ham,hammin, tc, pc;
		float kinetic, potential1, potential2, potential3;
		float6 out;

		thrust::default_random_engine rng;
		rng.seed((int)in.seed);
		rng.discard(N*thread_id);
		thrust::uniform_real_distribution<float> u01(0,1);

		// *************** x-px phase-space ********************
		// sig = sqrt(emit * beta)
		ampx = sqrt(in.betx*in.emitx);
		do 
    	{
        	r1 = 2*u01(rng)-1;
	        r2 = 2*u01(rng)-1;
	        amp = r1*r1+r2*r2;
	    } 
    	while ((amp >=1) || (amp<=3.e-9));

		facc = sqrt(-2*log(amp)/amp);

		out.x = ampx*r1*facc;
		out.px = ampx*r2*facc;


		// *************** y-py phase-space ***************************
		// sig = sqrt(emit * beta)
		ampy = sqrt(in.bety*in.emity);
		do 
    	{
        	r1 = 2*u01(rng)-1;
	        r2 = 2*u01(rng)-1;
	        amp = r1*r1+r2*r2;
	    } 
    	while ((amp >=1) || (amp<=3.e-9));

    	facc = sqrt(-2*log(amp)/amp);

		out.y = ampy*r1*facc;
		out.py = ampy*r2*facc;

    	// t ~ N(t_s,sigt)

    	ampt = in.sigs / CUDA_C_F;
    	ts = in.phiSynchronous / (in.h0 * in.omega0); // distribution has to be around phis
    	tc = (in.omega0 * in.eta * in.h0);
	    pc = (in.omega0 * in.charge) / (2 * CUDA_PI_F * in.p0 * in.betar);

    	do
    	{	
    		r1 = 2*u01(rng)-1;
	        r2 = 2*u01(rng)-1;
	        amp = r1*r1+r2*r2;
        	facc = sqrt(-2*log(amp)/amp);
        	out.t = ts + ampt * r1 * facc;
        	if (abs(out.t)>=in.tauhat) continue;

        	// calculate the hamiltonian for this t (delta =0)
        	// the accepted interval for delta is then only where hammin<=ham(t,delta)<=hammax
        	potential1 = pc * in.v0 * (cos(in.h0 * in.omega0 * out.t) - cos(in.phiSynchronous) + (in.h0 * in.omega0 * out.t - in.phiSynchronous) * sin(in.phiSynchronous));

			potential2 = pc * in.v1 * (in.h0 / in.h1) * (cos(in.h1 * in.omega0 * out.t) - cos(in.h1 * in.phiSynchronous / in.h0) + 
				(in.h1 * in.omega0 * out.t - in.h1 * in.phiSynchronous / in.h0) * sin(in.h1 * in.phiSynchronous / in.h0));

			potential3 = pc * in.v2 * (in.h0 / in.h2) * (cos(in.h2 * in.omega0 * out.t) - cos(in.h2 * in.phiSynchronous / in.h0) + 
				(in.h2 * in.omega0 * out.t - in.h2 * in.phiSynchronous / in.h0) * sin(in.h2 * in.phiSynchronous / in.h0));

			hammin = potential1 + potential2 + potential3;

		}
		while ((hammin>in.hammax)|| (abs(out.t-ts)>=in.tauhat) || (isnan(out.t)));

		do 
		{
			r1 = 2*u01(rng)-1;
	        r2 = 2*u01(rng)-1;
	        amp = r1*r1+r2*r2;
        	facc = sqrt(-2*log(amp)/amp);
        	out.delta = in.sige * r2 * facc;

			kinetic = 0.5 * tc * pow(out.delta,2);

			potential1 = pc * in.v0 * (cos(in.h0 * in.omega0 * out.t) - cos(in.phiSynchronous) + (in.h0 * in.omega0 * out.t - in.phiSynchronous) * sin(in.phiSynchronous));
 			potential2 = pc * in.v1 * (in.h0 / in.h1) * (cos(in.h1 * in.omega0 * out.t) - cos(in.h1 * in.phiSynchronous / in.h0) + 
 				(in.h1 * in.omega0 * out.t - in.h1 * in.phiSynchronous / in.h0) * sin(in.h1 * in.phiSynchronous / in.h0));
			potential3 = pc * in.v2 * (in.h0 / in.h2) * (cos(in.h2 * in.omega0 * out.t) - cos(in.h2 * in.phiSynchronous / in.h0) + 
				(in.h2 * in.omega0 * out.t - in.h2 * in.phiSynchronous / in.h0) * sin(in.h2 * in.phiSynchronous / in.h0));
			ham = kinetic + potential1 + potential2 + potential3;
		}
		while ((ham < hammin) || (ham > in.hammax) || (isnan(out.delta))); 

		return out;

	};	
};

__host__ __device__ float synchrotronTune(longitudinalParameters in)
{	
	float Omega2 = (
		in.h0 * in.eta * in.charge)/(2 * CUDA_PI_F * in.p0) * (in.v0 * cos(in.phiSynchronous) 
		+ in.v1 * (in.h1/in.h0) * cos((in.h1/in.h0) * in.phiSynchronous)
		+ in.v2 * (in.h2/in.h0) * cos((in.h2/in.h0) * in.phiSynchronous)
		);
	return sqrt(abs(Omega2));
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


struct addFloat3 
{
  __host__ __device__
  float3 operator()(const float3& a, const float3& b) const {  

  		float3 result;

		result.x = a.x + b.x;
		result.y = a.y + b.y;
		result.z = a.z + b.z;

		return result;
	}
};

struct addRadiationIntegralsElements
{
	__host__ __device__
	radiationIntegrals operator()(const radiationIntegrals& radiationIntegralsElement1, const radiationIntegrals& radiationIntegralsElement2) const {

		radiationIntegrals result;

		result.I2  = radiationIntegralsElement1.I2  + radiationIntegralsElement2.I2;
		result.I3  = radiationIntegralsElement1.I3  + radiationIntegralsElement2.I3;
		result.I4x = radiationIntegralsElement1.I4x + radiationIntegralsElement2.I4x;
		result.I4y = radiationIntegralsElement1.I4y + radiationIntegralsElement2.I4y;
		result.I5x = radiationIntegralsElement1.I5x + radiationIntegralsElement2.I5x;
		result.I5y = radiationIntegralsElement1.I5y + radiationIntegralsElement2.I5y;

		return result;
	}
};
// template <class T>
struct subtractAverage
{

	__host__ __device__
	float6 operator()(const float6& in, const float6& avg) const
	{
		float6 r;
		r.x = in.x - avg.x;
		r.px = in.px - avg.px;
		r.y = in.y - avg.y;
		r.py = in.py - avg.py;
		r.t = in.t - avg.t;
		r.delta = in.delta - avg.delta;
		return r;
	};

};



float6 caluculateEmittance(thrust::device_vector<float6> distribution, float2 tunes,float circumference, int n, float gamma0)
{	
	float betax, betay;

	betax = circumference / (2 * CUDA_PI_F * tunes.x);
	betay = circumference / (2 * CUDA_PI_F * tunes.y);
    
    // float6 sig, np;
    // np = length(distribution);
	float6 sum;
	sum.x= 0.0;
	sum.px=0.0;
	sum.y=0.0;
	sum.py=0.0;
	sum.t=0.0;
	sum.delta=0.0;
	float6 average = thrust::reduce(distribution.begin(),distribution.end(),sum,addFloat6());

	average.x     = - average.x / n;
	average.px    = - average.px / n;
	average.y     = - average.y / n;
	average.py    = - average.py / n;
	average.t     = - average.t / n;
	average.delta = - average.delta / n;

	thrust::transform(distribution.begin(), distribution.end(),thrust::make_constant_iterator(average),distribution.begin(),addFloat6());	


	thrust::transform(distribution.begin(),distribution.end(),distribution.begin(),squareFunctor<float6> ());
	float6 outsum = thrust::reduce(distribution.begin(),distribution.end(),sum,addFloat6());
	
	outsum.x = outsum.x / (n * betax);
	outsum.y = outsum.y / (n * betay);
	
	outsum.t = sqrt(outsum.t/n);
	outsum.delta = sqrt(outsum.delta/n);

    
    return outsum;

}

__host__ __device__ 
radiationIntegrals CalculateRadiationIntegralsApprox(radiationIntegralsParameters radiationIntParameters)
{
	radiationIntegrals outputIntegralsApprox;
	// growth rates
	float alphax = 0.0;
	float alphay = 0.0;

	float gammax = (1.0 +  pow(alphax,2)) / radiationIntParameters.betxRingAverage;
	float gammay = (1.0 +  pow(alphay,2)) / radiationIntParameters.betyRingAverage;

	float Dx = radiationIntParameters.acceleratorLength / (2 * CUDA_PI_F * radiationIntParameters.gammaTransition);
	float Dy = 0.0;

	float Dxp = 0.1; // should find an approximation formula. However not very important
	float Dyp = 0.0;

	float Hx = (radiationIntParameters.betxRingAverage * Dxp + 2 * alphax * Dx * Dxp + gammax * Dx);
    float Hy = (radiationIntParameters.betyRingAverage * Dyp + 2 * alphay * Dy * Dyp + gammay * Dy);

    //  define smooth approximation of radiation integrals
    outputIntegralsApprox.I2 = 2 * CUDA_PI_F / radiationIntParameters.DipoleBendingRadius;
    outputIntegralsApprox.I3 = 2 * CUDA_PI_F / pow(radiationIntParameters.DipoleBendingRadius,2);
    outputIntegralsApprox.I4x = 0.0;
    outputIntegralsApprox.I4y = 0.0;
    outputIntegralsApprox.I5x = Hx * 2 * CUDA_PI_F / pow(radiationIntParameters.DipoleBendingRadius,2);
    outputIntegralsApprox.I5y = Hy * 2 * CUDA_PI_F / pow(radiationIntParameters.DipoleBendingRadius,2);

    
    return outputIntegralsApprox;

};


struct CalculateRadiationIntegralsLatticeElement
{


	__host__ __device__
	radiationIntegrals operator()(tfsTableData& tfsAcceleratorElement, radiationIntegralsParameters& radiationIntParameters) const 
	{
		radiationIntegrals outputIntegralsLattice;

		float angle  = tfsAcceleratorElement.angle;
		float l      = tfsAcceleratorElement.l;
		float k1l    = tfsAcceleratorElement.k1l;
		float dy     = tfsAcceleratorElement.dy;
		float k1s    = tfsAcceleratorElement.k1sl;
		float alphax = tfsAcceleratorElement.alfx;
		float alphay = tfsAcceleratorElement.alfy;
		float betx   = tfsAcceleratorElement.betx;
		float bety   = tfsAcceleratorElement.bety;
		float dx     = tfsAcceleratorElement.dx;
		float dpx    = tfsAcceleratorElement.dpx;
		float dpy    = tfsAcceleratorElement.dpy;

		float rhoi = ( angle > 0.0) ? l /angle : 0.0;
		float ki = (l > 0.0) ? k1l / l : 0.0 ;

		outputIntegralsLattice.I2 = (rhoi > 0.0) ? l / pow(rhoi,2) : 0.0 ;
		outputIntegralsLattice.I3 = (rhoi > 0.0) ? l / pow(rhoi,3) : 0.0 ;
    
    	//  corrected to equations in accelerator handbook  Chao second edition p 220
    	outputIntegralsLattice.I4x = (rhoi > 0.0) ? ((dx / pow(rhoi,3)) + 2 * (ki * dx + (k1s / l) * dy) / rhoi) *l : 0.0 ;
    	outputIntegralsLattice.I4y = 0.0;

    	float gammax = (1.0 + pow(alphax,2)) / betx;
    	float gammay = (1.0 + pow(alphay,2)) / bety;

    	float Hx =  betx * pow(dpx,2) + 2. * alphax * dx * dpx + gammax * pow(dx,2);
    	float Hy =  bety * pow(dpy,2) + 2. * alphay * dy * dpy + gammay * pow(dy,2);

    	outputIntegralsLattice.I5x = Hx * 2 * CUDA_PI_F / pow(radiationIntParameters.DipoleBendingRadius,2) * l;
    	outputIntegralsLattice.I5y = Hy * 2 * CUDA_PI_F / pow(radiationIntParameters.DipoleBendingRadius,2) * l;

		return outputIntegralsLattice;
	}
};


radiationIntegrals CalculateRadiationIntegralsLatticeRing(thrust::device_vector<tfsTableData> tfsData, radiationIntegralsParameters params)
{
	int n = tfsData.size();
	thrust::device_vector<radiationIntegrals> radiationIntegralsPerElement(n);

	thrust::transform(tfsData.begin(),tfsData.end(),thrust::make_constant_iterator(params),radiationIntegralsPerElement.begin(),CalculateRadiationIntegralsLatticeElement());

	radiationIntegrals initsum;
	initsum.I2  = 0.0;
	initsum.I3  = 0.0;
	initsum.I4x = 0.0;
	initsum.I4y = 0.0;
	initsum.I5x = 0.0;
	initsum.I5y = 0.0;

	radiationIntegrals total = thrust::reduce(radiationIntegralsPerElement.begin(),radiationIntegralsPerElement.end(),initsum,addRadiationIntegralsElements());

	return total;
}

float6 CalculateRadiationDampingTimesAndEquilib(radiationIntegralsParameters params, radiationIntegrals integrals)
{
	float6 result;

	// Chao handbook second edition page 221 eq. 11
	// float CalphaEC   = params.ParticleRadius * CUDA_C_F / (3 * params.acceleratorLength);
	float CalphaEC   = params.ParticleRadius * CUDA_C_F / (3 * pow(CUDA_ELECTRON_REST_E_F,3)) * (pow(params.p0,3)/params.acceleratorLength);
	
	// extra factor 2 to get growth rates for emittances and not amplitudes (sigmas)
	float alphax = 2.0f * CalphaEC * integrals.I2 * (1.0f - integrals.I4x / integrals.I2);
	float alphay = 2.0f * CalphaEC * integrals.I2 * (1.0f - integrals.I4y / integrals.I2);
	float alphas = 2.0f * CalphaEC * integrals.I2 * (2.0f + (integrals.I4x + integrals.I4y) / integrals.I2);

	// longitudinal equilibrium
	// Chao handbook second edition page 221 eq. 19
	float sigEoE02 = params.cq * pow(params.gammar,2) * integrals.I3 / (2 * integrals.I2 + integrals.I4x + integrals.I4y);
	float sigsEquilib = (CUDA_C_F * abs(params.eta) / params.omegas) * sqrt(sigEoE02);

	// Chao handbook second edition page 221 eq. 12
	float Jx = 1. - integrals.I4x / integrals.I2;
   float Jy = 1. - integrals.I4y / integrals.I2;

   // transverse equilibrium
   float EmitEquilibx = params.cq * pow(params.gammar,2) * integrals.I5x / (Jx * integrals.I2);
   float EmitEquiliby = params.cq * pow(params.gammar,2) * integrals.I5y / (Jy * integrals.I2);

   if (EmitEquiliby == 0.0)
   	EmitEquiliby = params.cq * params.betyRingAverage * integrals.I3 / (2 * Jy * integrals.I2);


   result.x     = 1.0 / alphax; // damping time returned in seconds
   result.px    = 1.0 / alphay;
   result.y     = 1.0 / alphas; 
   result.py    = EmitEquilibx;
   result.t     = EmitEquiliby;
   result.delta = sigsEquilib/CUDA_C_F; // sigs returned in seconds 
   return result;

}

struct RadiationDampingRoutineFunctor
{
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
		float coeffexcitelong = radiationDampParams.delta * CUDA_C_F * sqrt(3.) * sqrt(1 - pow(coeffdecaylong,2));

		// the damping time is for EMITTANCE, therefore need to multiply by 2
	    float coeffdecayx     = 1 - ((trev /(2 * radiationDampParams.x)) * timeratio);
	    float coeffdecayy     = 1 - ((trev /(2 * radiationDampParams.px)) * timeratio);
	    
	    // exact     coeffgrow= sigperp*sqrt(3.)*sqrt(1-coeffdecay**2)
	    // but trev << tradperp so
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

struct RfUpdateRoutineFunctor
{
	RfUpdateRoutineFunctor(float fmix, longitudinalParameters params): fmix(fmix), params(params) {}
	__host__ __device__ float6 operator()(float6 particle)
	{
		float6 result;

		// using parameters to generate parameters for functions 
		float3 voltages = make_float3(params.v0, params.v1, params.v2);
		float3 harmonicNumbers = make_float3(params.h0, params.h1, params.h2);
		float tcoeffValue = tcoeff(params.eta,params.omega0,params.h0) / (params.h0 * params.omega0);

		// Lee Third edition page 233 eqs 3.6 3.16 3.13
		
		// the phase  = h0 omega0 t
		float phi = params.h0 * params.omega0 * particle.t ;

		// Delta delta 
		float pcoeffValue = pcoeff(VoltageTripleRf(phi, voltages, harmonicNumbers) - VoltageTripleRf(params.phiSynchronous, voltages, harmonicNumbers), params.omega0, params.p0, params.betar, params.charge);

		result.x     = particle.x;
		result.px    = particle.px;
		result.y     = particle.y;
		result.py    = particle.py;
		result.t     = particle.t + fmix * 2 * CUDA_PI_F * params.eta * particle.delta / params.omega0;
		float voltdiff = VoltageTripleRf(phi, voltages, harmonicNumbers) - VoltageTripleRf(params.phiSynchronous, voltages, harmonicNumbers);
		result.delta = particle.delta + fmix * voltdiff * params.charge / (2 * CUDA_PI_F * params.betar * params.p0);

		return result;

	}
private:
	longitudinalParameters params;
	float fmix;
};

// function to remove particles outside of rf bucket after rf update
struct isInLong
{
	isInLong(float tauhat): tauhat(tauhat) {}
	__host__ __device__ 
	bool  operator()(const float6 particle)
	{
		return !((abs(particle.t) < tauhat ) || (tauhat == 0.0));
	}
private:
	float tauhat;
};

struct BetatronRoutine
{
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


__host__ __device__ float fmohl(float a, float b, float q, int fmohlNumPoints) 
{
      float result;
      float sum = 0;
      float du = 1.0/fmohlNumPoints;
      float u, cp, cq, dsum;

      for(int k=0;k<fmohlNumPoints;k++)
      	{
      		u = k*du;
         	cp = sqrt(a * a + (1 - a * a) * u * u);
         	cq = sqrt(b * b + (1 - b * b) * u * u);
         	dsum = 2*log(q*(1/cp+1/cq)/2) - CUDA_EULER_F;
         	dsum *= (1-3*u*u)/(cp*cq);
         	if (k==0)
         		dsum = dsum / 2;
         	if (k==fmohlNumPoints)
         		dsum = dsum / 2;
         	sum += dsum;
         }
      result = 8 * CUDA_PI_F * du * sum;

      return result;
}

__host__ __device__ float3 ibsPiwinskiSmooth(ibsparameters params)
	{
		float d;
		float xdisp = params.acceleratorLength / (2 * CUDA_PI_F * pow(params.gammaTransition,2));
		float rmsx  = sqrt(params.emitx * params.betx);
		float rmsy  = sqrt(params.emity * params.bety);

		float sigh2inv =  1.0 / pow(params.dponp,2)  + pow((xdisp / rmsx),2);
		float sigh = 1.0 / sqrt(sigh2inv);

		float atop = pow(params.ParticleRadius,2) * CUDA_C_F * params.numberRealParticles;
		float coeff = params.emitx * params.emity * params.sigs * params.dponp;
      	float abot = 64.0 * pow(CUDA_PI_F,2) * pow(params.betar,3) * pow(params.gamma0,4) * coeff;

      	// ca is Piwinski's A
      	float ca = atop/abot;
      	float a  = sigh * params.betx / (params.gamma0 * rmsx);
      	float b  = sigh * params.bety / (params.gamma0 * rmsy);

      	if (rmsx <=rmsy)
         	d = rmsx;
      	else
        	d = rmsy;
      
      	float q = sigh * params.betar * sqrt(2 * d / params.ParticleRadius);

      	float fmohlp = fmohl(a,b,q,params.fmohlNumPoints);
      	float fmohlx = fmohl(1/a,b/a,q/a,params.fmohlNumPoints);
      	float fmohly = fmohl(1/b,a/b,q/b,params.fmohlNumPoints);
      	float alfap0 = ca * fmohlp * pow((sigh/params.dponp),2);
      	float alfax0 = ca * (fmohlx + fmohlp * pow((xdisp * sigh / rmsx),2));
      	float alfay0 = ca * fmohly;

      	return make_float3(alfax0, alfay0, alfap0);

 	};

 struct ibsPiwinskiLattice
 {
 	ibsparameters params;
 	ibsPiwinskiLattice(ibsparameters params) : params(params) {}

 // 	__host__ __device__
	// radiationIntegrals operator()(tfsTableData& tfsAcceleratorElement, radiationIntegralsParameters& radiationIntParameters) const 
 	__host__ __device__
    float3 operator()(tfsTableData& tfsrow) const
    {
    	float d;

    	float atop = pow(params.ParticleRadius,2) * CUDA_C_F * params.numberRealParticles;
		float coeff = params.emitx * params.emity * params.sigs * params.dponp;
      	float abot = 64.0 * pow(CUDA_PI_F,2) * pow(params.betar,3) * pow(params.gamma0,4) * coeff;

      	// ca is Piwinski's A
      	float ca = atop/abot;

      	float rmsx = sqrt(params.emitx * tfsrow.betx);
        float rmsy = sqrt(params.emity * tfsrow.bety);

        if (rmsx <= rmsy)
        	d = rmsx;
        else
        	d = rmsy;

        float sigh2inv =  1.0 / pow(params.dponp,2)  + pow((tfsrow.dx / rmsx),2);
		float sigh = 1.0 / sqrt(sigh2inv);

		float a = sigh * tfsrow.betx / (params.gamma0 * rmsx);
		float b = sigh * tfsrow.bety / (params.gamma0 * rmsy);
		float q = sigh * params.betar * sqrt(2 * d / params.ParticleRadius);

		float fmohlp = fmohl(a,b,q,params.fmohlNumPoints);
      	float fmohlx = fmohl(1/a,b/a,q/a,params.fmohlNumPoints);
      	float fmohly = fmohl(1/b,a/b,q/b,params.fmohlNumPoints);
      	float alfap0 = ca * fmohlp * pow((sigh/params.dponp),2) * tfsrow.l / params.acceleratorLength;
      	float alfax0 = ca * (fmohlx + fmohlp * pow((tfsrow.dx * sigh / rmsx),2)) * tfsrow.l / params.acceleratorLength;
      	float alfay0 = ca * fmohly * tfsrow.l / params.acceleratorLength;

      	return make_float3(alfax0, alfay0, alfap0);
    };
 };


struct ibsmodPiwinskiLattice
{
	ibsparameters params;
 	ibsmodPiwinskiLattice(ibsparameters params) : params(params) {}

 // 	__host__ __device__
	// radiationIntegrals operator()(tfsTableData& tfsAcceleratorElement, radiationIntegralsParameters& radiationIntParameters) const 
 	__host__ __device__
    float3 operator()(tfsTableData& tfsrow) const
    {
    	float d;

    	float atop = pow(params.ParticleRadius,2) * CUDA_C_F * params.numberRealParticles;
		float coeff = params.emitx * params.emity * params.sigs * params.dponp;
      	float abot = 64.0 * pow(CUDA_PI_F,2) * pow(params.betar,3) * pow(params.gamma0,4) * coeff;

      	// ca is Piwinski's A
      	float ca = atop/abot;

      	float H = (pow(tfsrow.dx,2)+ pow((tfsrow.betx * tfsrow.dpx - 0.5 * tfsrow.dx * (-2 * tfsrow.alfx)),2)) / tfsrow.betx;

      	float rmsx = sqrt(params.emitx * tfsrow.betx);
        float rmsy = sqrt(params.emity * tfsrow.bety);

        if (rmsx <= rmsy)
        	d = rmsx;
        else
        	d = rmsy;

        float sigh2inv =  1.0 / pow(params.dponp,2)  + (H / params.emitx);
		float sigh = 1.0 / sqrt(sigh2inv);

		float a = sigh * tfsrow.betx / (params.gamma0 * rmsx);
		float b = sigh * tfsrow.bety / (params.gamma0 * rmsy);
		float q = sigh * params.betar * sqrt(2 * d / params.ParticleRadius);

		float fmohlp = fmohl(a,b,q,params.fmohlNumPoints);
      	float fmohlx = fmohl(1/a,b/a,q/a,params.fmohlNumPoints);
      	float fmohly = fmohl(1/b,a/b,q/b,params.fmohlNumPoints);
      	float alfap0 = ca * fmohlp * pow((sigh/params.dponp),2) * tfsrow.l / params.acceleratorLength;
      	float alfax0 = ca * (fmohlx + fmohlp * pow((tfsrow.dx * sigh / rmsx),2)) * tfsrow.l / params.acceleratorLength;
      	float alfay0 = ca * fmohly * tfsrow.l / params.acceleratorLength;

      	return make_float3(alfax0, alfay0, alfap0);
    };
};

// numerical recipes
// !Computes Carlson elliptic integral of the second kind, RD(x, y, z). x and y must be
// !nonnegative, and at most one can be zero. z must be positive. TINY must be at least twice
// !the negative 2/3 power of the machine overflow limit. BIG must be at most 0.1×ERRTOL
// !times the negative 2/3 power of the machine underflow limit.
__host__ __device__ float rd_s(float3 in) 
{
	float ERRTOL = 0.05;
	// float TINY   = 1.0e-25;
	// float BIG    = 4.5e21;
	float C1 = 3.0/14.0;
	float C2 = 1.0/6.0;
	float C3 = 9.0/22.0;
	float C4 = 3.0/26.0;
	float C5 = 0.25 * C3;
	float C6 = 1.5*C4;

	float xt  = in.x;
	float yt  = in.y;
	float zt  = in.z;
	float sum = 0.0;
	float fac = 1.0;
	int iter  = 0;

	float sqrtx, sqrty, sqrtz, alamb, ave, delx, dely, delz, maxi ;
	do
	{
		iter++;
		sqrtx = sqrt(xt);
	    sqrty = sqrt(yt);
	    sqrtz = sqrt(zt);
	    alamb = sqrtx * (sqrty + sqrtz) + sqrty * sqrtz;
   		sum   = sum + fac / (sqrtz * (zt + alamb));
   		fac   = 0.25 * fac;
   		xt    = 0.25 * (xt + alamb);
   		yt    = 0.25 * (yt + alamb);
   		zt    = 0.25 * (zt + alamb);
   		ave   = 0.2  * (xt + yt + 3.0 * zt);
   		delx  = (ave - xt) / ave;
  		dely  = (ave - yt) / ave;
  		delz  = (ave - zt) / ave;
  		maxi = abs(delx);
  		if (abs(dely) > maxi) 
  			maxi = abs(dely);
  		if (abs(delz) > maxi) 
  			maxi = abs(delz);
	}
	while (maxi > ERRTOL);

	float ea = delx * dely;
	float eb = delz * delz;
	float ec = ea - eb;
	float ed = ea - 6.0 * eb;
	float ee = ed + ec + ec;
	float rd_s = 3.0 * sum + fac * (1.0 + ed *(-C1+C5*ed-C6*delz*ee)+delz*(C2*ee+delz*(-C3*ec+delz*C4*ea)))/(ave*sqrt(ave));
	return rd_s;
};

struct ibsnagaitsev
{
	ibsparameters params;
 	ibsnagaitsev(ibsparameters params) : params(params) {}
 	__host__ __device__
    float3 operator()(tfsTableData& tfsrow) const
    {
    	float phi = tfsrow.dpx + (tfsrow.alfx * (tfsrow.dx/tfsrow.betx));
    	float axx = tfsrow.betx / params.emitx; 
        float ayy = tfsrow.bety / params.emity;

        float sigmax = sqrt( pow(tfsrow.dx,2) * pow(params.dponp,2) + params.emitx * tfsrow.betx);
        float sigmay = sqrt(params.emity * tfsrow.bety);
        float as     = axx * (pow(tfsrow.dx,2)/pow(tfsrow.betx,2) + pow(phi,2)) + (1/(pow(params.dponp,2)));

        float a1 = 0.5 * (axx + pow(params.gamma0,2) * as);
        float a2 = 0.5 * (axx - pow(params.gamma0,2) * as);

        float b1 = sqrt(pow(a2,2) + pow(params.gamma0,2) * pow(axx,2) * pow(phi,2));

        float lambda1 = ayy;
        float lambda2 = a1 + b1;
        float lambda3 = a1 - b1;

        float R1 = (1/lambda1) * rd_s(make_float3(1./lambda2,1./lambda3,1./lambda1));
        float R2 = (1/lambda2) * rd_s(make_float3(1./lambda3,1./lambda1,1./lambda2));
        float R3 = 3*sqrt((lambda1*lambda2)/lambda3)-(lambda1/lambda3)*R1-(lambda2/lambda3)*R2;

        float sp = (pow(params.gamma0,2)/2.0) * ( 2.0*R1 - R2*( 1.0 - 3.0*a2/b1 ) - R3*( 1.0 + 3.0*a2/b1 ));
		float sx = 0.50 * (2.0*R1 - R2*(1.0 + 3.0*a2/b1) -R3*(1.0 - 3.0*a2/b1));
		float sxp=(3.0 * pow(params.gamma0,2)* pow(phi,2)*axx)/b1*(R3-R2);

        float alfapp = sp/(sigmax*sigmay);
        float alfaxx = (tfsrow.betx/(sigmax*sigmay)) * (sx+sxp+sp*(pow(tfsrow.dx,2)/pow(tfsrow.betx,2) + pow(phi,2)));
		float alfayy = (tfsrow.bety/(sigmax*sigmay)) * (-2.0*R1+R2+R3);

		float alfap0 = alfapp * tfsrow.l / params.acceleratorLength;
        float alfax0 = alfaxx * tfsrow.l / params.acceleratorLength;
        float alfay0 = alfayy * tfsrow.l / params.acceleratorLength;

        return make_float3(alfax0, alfay0, alfap0);
    };
};
// algortihm taken from https://github.com/thrust/thrust/blob/master/examples/histogram.cu#L171 and adapted to simulation code requirements
// histogram creation
template <typename Vector1, 
          typename Vector2>
void dense_histogram(const Vector1& input,
                           Vector2& histogram,
                           int nbins,
                           float tauhat)
{
  typedef typename Vector1::value_type ValueType; // input value type
  typedef typename Vector2::value_type IndexType; // histogram index type

  // copy input data (could be skipped if input is allowed to be modified)
  thrust::device_vector<ValueType> data(input);
    
  // print the initial data
  // print_vector("initial data", data);

  // sort data to bring equal elements together
  thrust::sort(data.begin(), data.end());
  
  // print the sorted data
  // print_vector("sorted data", data);

  // number of histogram bins is equal to the maximum value plus one
  IndexType num_bins = nbins+1;

  // resize histogram storage
  histogram.resize(num_bins);
  
  // find the end of each bin of values
  thrust::counting_iterator<IndexType> search_begin(0);

  // thrust::device_vector<float> gc_d(nbins+1);
  // thrust::fill(gc_d.begin(),gc_d.end(), tauhat / nbins);

  // thrust::upper_bound(data.begin(), data.end(),
  //                     gc_d.begin(), gc_d.begin() + num_bins,
  //                     histogram.begin());
  

  thrust::upper_bound(data.begin(), data.end(),
                      search_begin, search_begin + num_bins,
                      histogram.begin());
  
  // print the cumulative histogram
  // print_vector("cumulative histogram", histogram);

  // compute the histogram by taking differences of the cumulative histogram
  thrust::adjacent_difference(histogram.begin(), histogram.end(),
                              histogram.begin());

  // print the histogram
  // print_vector("histogram", histogram);
}


// gets the particle times and uses the phase acceptance and number of bins
// to generate an integer which is than binned/histogram by ParticleTimesToHistogram
struct ParticleTimesToInteger
{

	__host__ __device__
	int operator()(float6& data, float2& params) const 
	{
		float out;
		float dtsamp2 = 2*params.x / params.y ;
		out = (data.t + params.x ) / dtsamp2;
		out = (int)(out + 0.5f);
		return out;
	}
};


// takes the integers produced by ParticleTimes to Integer to produce a histogram or binned version
// necessary for collision and intra beam scattering routines

thrust::device_vector<int> ParticleTimesToHistogram(thrust::device_vector<float6> data, int nbins, float tauhat)
{
	int n = data.size();
	thrust::device_vector<int> timecomponent(n);
	thrust::device_vector<int> histogram;

	// load the integer time components in a device vector
	thrust::transform(data.begin(),data.end(),thrust::make_constant_iterator(make_float2(tauhat,nbins)),timecomponent.begin(),ParticleTimesToInteger());
 	
 	// binning the times
	dense_histogram(timecomponent,histogram,nbins,tauhat);

	return histogram;
};

float6 CalcRMS(thrust::device_vector<float6> distribution, int numberMacroParticles)
{
	float6 sum;

	sum.x= 0.0;
	sum.px=0.0;
	sum.y=0.0;
	sum.py=0.0;
	sum.t=0.0;
	sum.delta=0.0;
	float6 average = thrust::reduce(distribution.begin(),distribution.end(),sum,addFloat6());

	average.x     = - average.x / numberMacroParticles;
	average.px    = - average.px / numberMacroParticles;
	average.y     = - average.y / numberMacroParticles;
	average.py    = - average.py / numberMacroParticles;
	average.t     = - average.t / numberMacroParticles;
	average.delta = - average.delta / numberMacroParticles;

	// subtract shift -> t_synchronous
	thrust::transform(distribution.begin(), distribution.end(),thrust::make_constant_iterator(average),distribution.begin(),addFloat6());

	// square 
	thrust::transform(distribution.begin(),distribution.end(),distribution.begin(),squareFunctor<float6> ());
	
	sum.x= 0.0;
	sum.px=0.0;
	sum.y=0.0;
	sum.py=0.0;
	sum.t=0.0;
	sum.delta=0.0;

	// sum squares
	float6 MS = thrust::reduce(distribution.begin(),distribution.end(),sum,addFloat6());

	MS.x     /= numberMacroParticles;
	MS.px    /= numberMacroParticles;
	MS.y     /= numberMacroParticles;
	MS.py    /= numberMacroParticles;
	MS.t     /= numberMacroParticles;
	MS.delta /= numberMacroParticles;

	MS.x     = sqrt(MS.x);
	MS.px    = sqrt(MS.px);
	MS.y     = sqrt(MS.y);
	MS.py    = sqrt(MS.py);
	MS.t     = sqrt(MS.t);
	MS.delta = sqrt(MS.delta);

	return MS;
};
	

__host__ float4 CalcIbsCoeff(ibsparameters params, int method,  float tsynchro, thrust::device_vector<tfsTableData> tfsdata)
{
	float3 ibsgrowthrates;
	float coeffs,coeffx, coeffy;
	float alphaAverage;
	float coeffMulT;

	int m = tfsdata.size();
	thrust::device_vector<float3> ibsgrowthratestfs(m);
	

	float dtsamp2 = 2 * params.tauhat / params.nbins;
	float rmsdelta  = params.dponp; 
	float sigs  = params.sigs;

	float rmsx = sqrt(params.emitx * params.betx);
	float rmsy = sqrt(params.emity * params.bety);

	// get growth rates according to selected ibs method
  	switch(method)
  	{
  		case 0: {
  			ibsgrowthrates = ibsPiwinskiSmooth(params);
  			// debugging
  			// cout << ibsgrowthrates.z << endl;
  			break;
  		}
  		case 1:{
  			thrust::transform(tfsdata.begin(),tfsdata.end(),ibsgrowthratestfs.begin(),ibsPiwinskiLattice(params));
  			ibsgrowthrates = thrust::reduce(ibsgrowthratestfs.begin(),ibsgrowthratestfs.end(),make_float3(0.0,0.0,0.0),addFloat3());
  			// debugging
  			// cout << ibsgrowthrates.z << endl;
  			break;
  		}
  		case 2:{
  			thrust::transform(tfsdata.begin(),tfsdata.end(),ibsgrowthratestfs.begin(),ibsmodPiwinskiLattice(params));
  			ibsgrowthrates = thrust::reduce(ibsgrowthratestfs.begin(),ibsgrowthratestfs.end(),make_float3(0.0,0.0,0.0),addFloat3());
  			// debugging
  			// cout << ibsgrowthrates.z << endl;
  			break;
  		}
  		case 3: {
  	// 		int m = tfsdata.size();
			// thrust::device_vector<float3> ibsgrowthratestfs(m);
  			thrust::transform(tfsdata.begin(),tfsdata.end(),ibsgrowthratestfs.begin(),ibsnagaitsev(params));
  			ibsgrowthrates = thrust::reduce(ibsgrowthratestfs.begin(),ibsgrowthratestfs.end(),make_float3(0.0,0.0,0.0),addFloat3());
  			float nom = 0.5 * (params.numberRealParticles * pow(params.ParticleRadius,2) * CUDA_C_F * params.coulomblog);
  			float denom = (12.0 * CUDA_PI_F * pow(params.betar,3) * pow(params.gamma0,5) * params.sigs);
  			ibsgrowthrates.x = ibsgrowthrates.x / params.emitx *  nom / denom;
  			ibsgrowthrates.y = ibsgrowthrates.y / params.emity *  nom / denom;
  			ibsgrowthrates.z = ibsgrowthrates.z / pow(params.dponp,2) *  nom / denom;
  			// debugging
  			// cout << ibsgrowthrates.z << endl;
  			break;
  		}
  		

  	};

  	// uncomment for debugging
  	// cout << "fracibstot " << params.fracibstot << endl;
  	// cout << "real part " << params.numberRealParticles << endl;
  	// cout << "timeRatio " << params.timeratio << endl;
  	// cout << "growth rate z " << ibsgrowthrates.z << endl;
  	// cout << "numberMacroParticles " << params.numberMacroParticles << endl;

  	float alfap = 2 * params.fracibstot * params.numberRealParticles * params.timeratio * ibsgrowthrates.z / params.numberMacroParticles;
  	float alfax = 2 * params.fracibstot * params.numberRealParticles * params.timeratio * ibsgrowthrates.x / params.numberMacroParticles;
  	float alfay = 2 * params.fracibstot * params.numberRealParticles * params.timeratio * ibsgrowthrates.y / params.numberMacroParticles;

  	// debugging
  	// cout << "alfap " << alfap << endl << endl;
  	if (alfap > 0.0f)
    	coeffs = sqrt(6 * alfap * params.trev) * rmsdelta;
  	else
      	coeffs = 0.0f;
  
  	// coupling
    if (params.ibsCoupling == 0.0)
    {
    	if (alfay > 0.0)
    		coeffy = sqrt(6 * alfay * params.trev) * rmsy;
    	else
    		coeffy = 0.0f;

    	if (alfax > 0.0)
    		coeffx = sqrt(6 * alfax * params.trev) * rmsx;
    	else
    		coeffx = 0.0f;
    }
    else
    {	
    	// alphaAverage
    	alphaAverage = 0.5 * (alfax + alfay);
    	if (alphaAverage > 0.0)
    	{
    		coeffx = sqrt(6 * alphaAverage * params.trev) * rmsx;
        	coeffy = sqrt(6 * alphaAverage * params.trev) * rmsy;
    	}
    	else
    	{
    		coeffx = 0.0f;
        	coeffy = 0.0f;
    	}
    	// end if alphaAverage
    }
    // end if ibs coupling

    coeffMulT = sigs* 2* sqrt(CUDA_PI_F)/(params.numberRealParticles * dtsamp2 * CUDA_C_F);



	return make_float4(coeffx,coeffy,coeffs, coeffMulT);
};

struct sqrtFloatFunctor
{
	__device__ __host__ void operator() (float& v)
	{
		v =  sqrt(v);
	}
};


thrust::device_vector<float> HistogramToSQRTofCumul(thrust::device_vector<int> inputHistogram, float coeff)
{
	int n = inputHistogram.size();
	thrust::device_vector<float> vcoeff(n);
	// thrust::device_vector<float> vcoeff2;
	thrust::device_vector<float> cumul(n);
	// thrust::device_vector<int> outputhistogram;

	// fill constant vector
	thrust::fill(vcoeff.begin(),vcoeff.end(),coeff);

	// multiply with constant
	thrust::transform(inputHistogram.begin(),inputHistogram.end(),vcoeff.begin(),vcoeff.begin(),thrust::multiplies<float>());

	// cumulative sum
	thrust::inclusive_scan(vcoeff.begin(),vcoeff.end(),cumul.begin());

	// take sqrt
	thrust::for_each(cumul.begin(),cumul.end(),sqrtFloatFunctor());

	return cumul;
};


struct ibsRoutine
{
   float4 ibscoeff;
   float denlon2k;
   ibsparameters ibsparms;

   ibsRoutine(float4 ibscoeff, float denlon2k, ibsparameters ibsparms): ibscoeff(ibscoeff), denlon2k(denlon2k), ibsparms(ibsparms) {}
   
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

		float dy = denlon2k * ibscoeff.y * grv1;
        float dp = denlon2k * ibscoeff.z * grv2;


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

		float dx = denlon2k * ibscoeff.x * grv1;

		out.x = particle.x;
		out.y = particle.y;
		out.t = particle.t;

		out.px = particle.px + dx;
		out.py = particle.py + dy;
		out.delta = particle.delta + dp;

		return out;

   };
};
// struct ibsRoutineFunctor(const Vector1& input,
//                              Vector2& histogram,
//                              int method,
//                            	 float4 ibscoeff)
// {
//   typedef typename Vector1::value_type ValueType; // input value type
//   typedef typename Vector2::value_type IndexType; // histogram index type

//   // copy input data (could be skipped if input is allowed to be modified)
//   thrust::device_vector<ValueType> data(input);

//   // getting denlon2k from old code
//   thrust::device_vector<float> sqrtcumul = HistogramToSQRTofCumul(histogram, ibscoeff.w);  
//   thrust::device_vector<float>::iterator end = sqrttest.end()-1;  // .end() iterator returns an iterator on past the last element
//   float denlon2k = *end;


// struct getparticletime
// {
// 	__host__ __device__ float operator()(float6& particle)
// 	{
// 	return particle.t;
// 	}
// };

// __host__ __device__ binLong(thrust::device_vector<float6> distribution, int numbins)
// {
// 	thrust
// 	thrust::sort(distribution.begin(),distribution.end());
// 	thrust::device_vector<float6> histogram(numbins);
// 	thrust::counting_iterator<int> search_begin(0);
//   	thrust::upper_bound(distribution.begin(), distribution.end(),search_begin, search_begin + numbins, histogram.begin());
//   	thrust::adjacent_difference(histogram.begin(), histogram.end(), histogram.begin());
// };
// routine for avg t around 0 - correct by subtracting t synchronous needs to be added again !!!
// struct ibsRoutine
// {
// 	ibsRoutine(int nbins, ibsparameters params, int method, float sigt, float sigdelta, float tsynchro): {} // TODO use map to get method string to int
// 	__host__ __device__ float3 operator()(float6 particle)
// 	{
// 		float dtsamp2 = 2* params.tauhat / nBins;

// 		// binning
// 		float tk = (particle.t + params.tauhat) / dtsamp2;
// 		int nk = (int)(sqrt(tk)+0.5f);

// 		if (nk == 0)
// 			nk = 1;
// 		if (nk > nbins)
// 			nk = nbins;

// 		float dponp = sigdelta;
// 		float rmsg  = pow(params.betar,2) * params.gamma0 * sigdelta; 
// 		float sigs  = sigt * CUDA_C_F * params.betar;

// 		float rmsx = sqrt(params.emitx * params.betx);
//       	float rmsy = sqrt(params.emity * params.bety);

//       	// get growth rates according to selected ibs method
//       	switch(method)
//       	{
//       		case 0: {
//       			float3 ibsgrowthrates = ibsPiwinskiSmooth(params);
//       			break;
//       		}

//       	};
      	
//       	float alfap = 2 * params.fracibstot * params.numberRealParticles * params.timeRatio * ibsgrowthrates.z / params.numberMacroParticles;
//       	float alfay = 2 * params.fracibstot * params.numberRealParticles * params.timeRatio * ibsgrowthrates.y / params.numberMacroParticles;
//       	float alfay = 2 * params.fracibstot * params.numberRealParticles * params.timeRatio * ibsgrowthrates.x / params.numberMacroParticles;

//       	if (alfap > 0.0f)
//         	float coeffs = sqrt(6 * alfap * trev) * rmsg;
//       	else
//           	float coeffs = 0.0f;
      
//       	// coupling
//         if (params.ibsCoupling == 0.0)
//         {
//         	if (alfy > 0.0)
//         		float coeffy = sqrt(6 * alfay * trev) * rmsy;
//         	else
//         		float coeffy = 0..0f;

//         	if (alfx > 0.0)
//         		float coeffx = sqrt(6 * alfax * trev) * rmsx;
//         	else
//         		float coeffx = 0..0f;
//         }
//         else
//         {
//         	float alphaAverage = 0.5 * (alfax + alfay);
//         	if (alphaAverage > 0.0)
//         	{
//         		float coeffx = sqrt(6 * alphaAverage * trev) * rmsx;
//             	float coeffy = sqrt(6 * alphaAverage * trev) * rmsy;
//         	}
//         	else
//         	{
//         		float coeffx = 0.0f;
//             	float coeffy = 0.0f;
//         	}
//         	// end if alphaAverage
//         }
//         // end if ibs coupling

//         float coeffmult = sigs* 2* sqrt(CUDA_PI_F)/(params.numberRealParticles * dtsamp2 * CUDA_C_F);


// 	}
// };