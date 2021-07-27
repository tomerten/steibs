// include guard
#ifndef RANDOM_H_INCLUDED
#define RANDOM_H_INCLUDED

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
#include <boost/math/tools/roots.hpp>
#include <thrust/tuple.h>

// random generator includes
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/xor_combine_engine.h>
#include <thrust/random.h>
#include <curand_kernel.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/for_each.h>

#include <iterator>
#include <sstream>
#include <fstream>
#include <algorithm>

// load float 6 datastructure
#include "STE_DataStructures.cuh"


struct longitudinalParameters {
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
};





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


		return out;
	}
};

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

#endif