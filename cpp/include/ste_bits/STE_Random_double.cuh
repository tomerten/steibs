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

// load double 6 datastructure
#include "STE_DataStructures.cuh"


struct longitudinalParameters {
	double seed;
	double betx;
	double bety;
	double emitx;
	double emity;
	double tauhat;
	double sigs;
	double sige;
	double omega0;
	double v0;
	double v1;
	double v2;
	double h0;
	double h1;
	double h2;
	double phiSynchronous;
	double p0;
	double betar;
	double eta;
	double hammax;
	double charge;
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
		thrust::uniform_real_distribution<double> u01(0,1);
		// thrust::xor_combine_engine<thrust::minstd_rand,0,thrust::minstd_rand0,0> rng;
		//double x = (double)u01(rng);
		return u01(rng);
		
		// return (double)rng()/(double)rng.max;
	}
};

/*
 * PARALLEL RANDOM GENERATOR BI-GAUSSIAN IN 2D 
 * -------------------------------------------
 * rand_2d_gauss(double3 in)  
 * generates bi-gaussian in 2D 
*/
template <typename T>
struct rand_2d_gauss
{
	double3 in;
	rand_2d_gauss(double3 in): in(in)  {}

	__host__ __device__
	double2 operator()(unsigned int thread_id) const
	{
		unsigned int N = 1000; // samples per stream
		double ampx, amp, r1, r2, facc; // variables for generating bi-gaussian

		thrust::default_random_engine rng;
		rng.seed((int)in.x);
		rng.discard(N*thread_id);
		thrust::uniform_real_distribution<double> u01(0,1);

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
		//double x = (double)u01(rng);
		return make_double2(ampx*r1*facc,ampx*r2*facc);
		
		// return (double)rng()/(double)rng.max;
	}
};


template <typename T>
struct rand_6d_gauss
{
	longitudinalParameters in;
	rand_6d_gauss(longitudinalParameters in): in(in) {}

	__host__ __device__
	double6 operator()(unsigned int thread_id) const
	{
		unsigned int N = 1000;
		double ampx, ampy, amp, r1,r2, facc;
		double ampt,ham, tc, pc;
		double kinetic, potential1, potential2, potential3,ts;
		double6 out;

		thrust::default_random_engine rng;
		rng.seed((int)in.seed);
		rng.discard(N*thread_id);
		thrust::uniform_real_distribution<double> u01(0,1);

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
	double6 operator()(unsigned int thread_id) const
	{
		unsigned int N = 1000;
		double ampx, ampy, amp, r1,r2, facc,ts;
		double ampt,ham,hammin, tc, pc;
		double kinetic, potential1, potential2, potential3;
		double6 out;

		thrust::default_random_engine rng;
		rng.seed((int)in.seed);
		rng.discard(N*thread_id);
		thrust::uniform_real_distribution<double> u01(0,1);

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