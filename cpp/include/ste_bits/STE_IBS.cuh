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

// load float 6 datastructure
#include "STE_DataStructures.cuh"

// load tfsTableData datastructure
#include "STE_TFS.cuh"


struct ibsparameters {
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
	int methodIBS;
	float phiSynchronous;
	//thrust::device_vector<tfsTableData> tfsdata;
};

// numerical recipes
// !Computes Carlson elliptic integral of the second kind, RD(x, y, z). x and y must be
// !nonnegative, and at most one can be zero. z must be positive. TINY must be at least twice
// !the negative 2/3 power of the machine overflow limit. BIG must be at most 0.1×ERRTOL
// !times the negative 2/3 power of the machine underflow limit.
// __host__ __device__ float rd_s(float3 in) 
// {
// 	float ERRTOL = 0.05;
// 	// float TINY   = 1.0e-25;
// 	// float BIG    = 4.5e21;
// 	float C1 = 3.0/14.0;
// 	float C2 = 1.0/6.0;
// 	float C3 = 9.0/22.0;
// 	float C4 = 3.0/26.0;
// 	float C5 = 0.25 * C3;
// 	float C6 = 1.5*C4;

// 	float xt  = in.x;
// 	float yt  = in.y;
// 	float zt  = in.z;
// 	float sum = 0.0;
// 	float fac = 1.0;
// 	int iter  = 0;

// 	float sqrtx, sqrty, sqrtz, alamb, ave, delx, dely, delz, maxi ;
// 	do
// 	{
// 		iter++;
// 		sqrtx = sqrt(xt);
// 	    sqrty = sqrt(yt);
// 	    sqrtz = sqrt(zt);
// 	    alamb = sqrtx * (sqrty + sqrtz) + sqrty * sqrtz;
//    		sum   = sum + fac / (sqrtz * (zt + alamb));
//    		fac   = 0.25 * fac;
//    		xt    = 0.25 * (xt + alamb);
//    		yt    = 0.25 * (yt + alamb);
//    		zt    = 0.25 * (zt + alamb);
//    		ave   = 0.2  * (xt + yt + 3.0 * zt);
//    		delx  = (ave - xt) / ave;
//   		dely  = (ave - yt) / ave;
//   		delz  = (ave - zt) / ave;
//   		maxi = abs(delx);
//   		if (abs(dely) > maxi) 
//   			maxi = abs(dely);
//   		if (abs(delz) > maxi) 
//   			maxi = abs(delz);
// 	}
// 	while (maxi > ERRTOL);

// 	float ea = delx * dely;
// 	float eb = delz * delz;
// 	float ec = ea - eb;
// 	float ed = ea - 6.0 * eb;
// 	float ee = ed + ec + ec;
// 	float rd_s = 3.0 * sum + fac * (1.0 + ed *(-C1+C5*ed-C6*delz*ee)+delz*(C2*ee+delz*(-C3*ec+delz*C4*ea)))/(ave*sqrt(ave));
// 	return rd_s;
// };

// __host__ __device__ float fmohl(float a, float b, float q, int fmohlNumPoints) 
// {
//       float result;
//       float sum = 0;
//       float du = 1.0/fmohlNumPoints;
//       float u, cp, cq, dsum;

//       for(int k=0;k<fmohlNumPoints;k++)
//       	{
//       		u = k*du;
//          	cp = sqrt(a * a + (1 - a * a) * u * u);
//          	cq = sqrt(b * b + (1 - b * b) * u * u);
//          	dsum = 2*log(q*(1/cp+1/cq)/2) - CUDA_EULER_F;
//          	dsum *= (1-3*u*u)/(cp*cq);
//          	if (k==0)
//          		dsum = dsum / 2;
//          	if (k==fmohlNumPoints)
//          		dsum = dsum / 2;
//          	sum += dsum;
//          }
//       result = 8 * CUDA_PI_F * du * sum;

//       return result;
// }


// template <typename T>
// struct squareFunctor
// {
// 	__host__ __device__ 
// 	T operator() (const float6& vec) const 
// 	{	float6 out;
// 		out.x     = vec.x  * vec.x;
// 		out.px    = vec.px * vec.px;
// 		out.y     = vec.y  * vec.y;
// 		out.py    = vec.py * vec.py;
// 		out.t     = vec.t  * vec.t;
// 		out.delta = vec.delta * vec.delta;
// 		return out;
// 	};
// };

// struct addFloat6 
// {
//   __host__ __device__
//   float6 operator()(const float6& a, const float6& b) const {  

//   		float6 result;

// 		result.x = a.x + b.x;
// 		result.px = a.px + b.px;
// 		result.y = a.y + b.y;
// 		result.py = a.py + b.py;
// 		result.t = a.t + b.t;
// 		result.delta = a.delta + b.delta;

// 		return result;
// 	}
// };

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

struct sqrtFloatFunctor
{
	__device__ __host__ void operator() (float& v)
	{
		v =  sqrt(v);
	}
};


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

// __host__ __device__ float3 ibsPiwinskiSmooth(ibsparameters& params){
// 	float d;
// 	float xdisp = params.acceleratorLength / (2 * CUDA_PI_F * pow(params.gammaTransition,2));
// 	float rmsx  = sqrt(params.emitx * params.betx);
// 	float rmsy  = sqrt(params.emity * params.bety);

// 	float sigh2inv =  1.0 / pow(params.dponp,2)  + pow((xdisp / rmsx),2);
// 	float sigh = 1.0 / sqrt(sigh2inv);

// 	float atop = pow(params.ParticleRadius,2) * CUDA_C_F * params.numberRealParticles;
// 	float coeff = params.emitx * params.emity * params.sigs * params.dponp;
//   	float abot = 64.0 * pow(CUDA_PI_F,2) * pow(params.betar,3) * pow(params.gamma0,4) * coeff;

//   	// ca is Piwinski's A
//   	float ca = atop/abot;
//   	float a  = sigh * params.betx / (params.gamma0 * rmsx);
//   	float b  = sigh * params.bety / (params.gamma0 * rmsy);

//   	if (rmsx <=rmsy)
//      	d = rmsx;
//   	else
//     	d = rmsy;
  
//   	float q = sigh * params.betar * sqrt(2 * d / params.ParticleRadius);

//   	float fmohlp = fmohl(a,b,q,params.fmohlNumPoints);
//   	float fmohlx = fmohl(1/a,b/a,q/a,params.fmohlNumPoints);
//   	float fmohly = fmohl(1/b,a/b,q/b,params.fmohlNumPoints);
//   	float alfap0 = ca * fmohlp * pow((sigh/params.dponp),2);
//   	float alfax0 = ca * (fmohlx + fmohlp * pow((xdisp * sigh / rmsx),2));
//   	float alfay0 = ca * fmohly;

//   	return make_float3(alfax0, alfay0, alfap0);
//  };


// struct ibsPiwinskiLattice{
//  	ibsparameters params;
//  	ibsPiwinskiLattice(ibsparameters& params) : params(params) {}

//  // 	__host__ __device__
// 	// radiationIntegrals operator()(tfsTableData& tfsAcceleratorElement, radiationIntegralsParameters& radiationIntParameters) const 
//  	__host__ __device__
//     float3 operator()(tfsTableData& tfsrow) const
//     {
//     	float d;

//     	float atop = pow(params.ParticleRadius,2) * CUDA_C_F * params.numberRealParticles;
// 		float coeff = params.emitx * params.emity * params.sigs * params.dponp;
//       	float abot = 64.0 * pow(CUDA_PI_F,2) * pow(params.betar,3) * pow(params.gamma0,4) * coeff;

//       	// ca is Piwinski's A
//       	float ca = atop/abot;

//       	float rmsx = sqrt(params.emitx * tfsrow.betx);
//         float rmsy = sqrt(params.emity * tfsrow.bety);

//         if (rmsx <= rmsy)
//         	d = rmsx;
//         else
//         	d = rmsy;

//         float sigh2inv =  1.0 / pow(params.dponp,2)  + pow((tfsrow.dx / rmsx),2);
// 		float sigh = 1.0 / sqrt(sigh2inv);

// 		float a = sigh * tfsrow.betx / (params.gamma0 * rmsx);
// 		float b = sigh * tfsrow.bety / (params.gamma0 * rmsy);
// 		float q = sigh * params.betar * sqrt(2 * d / params.ParticleRadius);

// 		float fmohlp = fmohl(a,b,q,params.fmohlNumPoints);
//       	float fmohlx = fmohl(1/a,b/a,q/a,params.fmohlNumPoints);
//       	float fmohly = fmohl(1/b,a/b,q/b,params.fmohlNumPoints);
//       	float alfap0 = ca * fmohlp * pow((sigh/params.dponp),2) * tfsrow.l / params.acceleratorLength;
//       	float alfax0 = ca * (fmohlx + fmohlp * pow((tfsrow.dx * sigh / rmsx),2)) * tfsrow.l / params.acceleratorLength;
//       	float alfay0 = ca * fmohly * tfsrow.l / params.acceleratorLength;

//       	return make_float3(alfax0, alfay0, alfap0);
//     };
//  };


// struct ibsmodPiwinskiLattice{
// 	ibsparameters params;
//  	ibsmodPiwinskiLattice(ibsparameters& params) : params(params) {}

//  // 	__host__ __device__
// 	// radiationIntegrals operator()(tfsTableData& tfsAcceleratorElement, radiationIntegralsParameters& radiationIntParameters) const 
//  	__host__ __device__
//     float3 operator()(tfsTableData& tfsrow) const
//     {
//     	float d;

//     	float atop = pow(params.ParticleRadius,2) * CUDA_C_F * params.numberRealParticles;
// 		float coeff = params.emitx * params.emity * params.sigs * params.dponp;
//       	float abot = 64.0 * pow(CUDA_PI_F,2) * pow(params.betar,3) * pow(params.gamma0,4) * coeff;

//       	// ca is Piwinski's A
//       	float ca = atop/abot;

//       	float H = (pow(tfsrow.dx,2)+ pow((tfsrow.betx * tfsrow.dpx - 0.5 * tfsrow.dx * (-2 * tfsrow.alfx)),2)) / tfsrow.betx;

//       	float rmsx = sqrt(params.emitx * tfsrow.betx);
//         float rmsy = sqrt(params.emity * tfsrow.bety);

//         if (rmsx <= rmsy)
//         	d = rmsx;
//         else
//         	d = rmsy;

//         float sigh2inv =  1.0 / pow(params.dponp,2)  + (H / params.emitx);
// 		float sigh = 1.0 / sqrt(sigh2inv);

// 		float a = sigh * tfsrow.betx / (params.gamma0 * rmsx);
// 		float b = sigh * tfsrow.bety / (params.gamma0 * rmsy);
// 		float q = sigh * params.betar * sqrt(2 * d / params.ParticleRadius);

// 		float fmohlp = fmohl(a,b,q,params.fmohlNumPoints);
//       	float fmohlx = fmohl(1/a,b/a,q/a,params.fmohlNumPoints);
//       	float fmohly = fmohl(1/b,a/b,q/b,params.fmohlNumPoints);
//       	float alfap0 = ca * fmohlp * pow((sigh/params.dponp),2) * tfsrow.l / params.acceleratorLength;
//       	float alfax0 = ca * (fmohlx + fmohlp * pow((tfsrow.dx * sigh / rmsx),2)) * tfsrow.l / params.acceleratorLength;
//       	float alfay0 = ca * fmohly * tfsrow.l / params.acceleratorLength;

//       	return make_float3(alfax0, alfay0, alfap0);
//     };
// 	};


// gets the particle times and uses the phase acceptance and number of bins
// to generate an integer which is than binned/histogram by ParticleTimesToHistogram

// struct ibsnagaitsev{
// 	ibsparameters params;
//  	ibsnagaitsev(ibsparameters& params) : params(params) {}
//  	__host__ __device__
//     float3 operator()(tfsTableData& tfsrow) const
//     {
//     	float phi = tfsrow.dpx + (tfsrow.alfx * (tfsrow.dx/tfsrow.betx));
//     	float axx = tfsrow.betx / params.emitx; 
//         float ayy = tfsrow.bety / params.emity;

//         float sigmax = sqrt( pow(tfsrow.dx,2) * pow(params.dponp,2) + params.emitx * tfsrow.betx);
//         float sigmay = sqrt(params.emity * tfsrow.bety);
//         float as     = axx * (pow(tfsrow.dx,2)/pow(tfsrow.betx,2) + pow(phi,2)) + (1/(pow(params.dponp,2)));

//         float a1 = 0.5 * (axx + pow(params.gamma0,2) * as);
//         float a2 = 0.5 * (axx - pow(params.gamma0,2) * as);

//         float b1 = sqrt(pow(a2,2) + pow(params.gamma0,2) * pow(axx,2) * pow(phi,2));

//         float lambda1 = ayy;
//         float lambda2 = a1 + b1;
//         float lambda3 = a1 - b1;

//         float R1 = (1/lambda1) * rd_s(make_float3(1./lambda2,1./lambda3,1./lambda1));
//         float R2 = (1/lambda2) * rd_s(make_float3(1./lambda3,1./lambda1,1./lambda2));
//         float R3 = 3*sqrt((lambda1*lambda2)/lambda3)-(lambda1/lambda3)*R1-(lambda2/lambda3)*R2;

//         float sp = (pow(params.gamma0,2)/2.0) * ( 2.0*R1 - R2*( 1.0 - 3.0*a2/b1 ) - R3*( 1.0 + 3.0*a2/b1 ));
// 		float sx = 0.50 * (2.0*R1 - R2*(1.0 + 3.0*a2/b1) -R3*(1.0 - 3.0*a2/b1));
// 		float sxp=(3.0 * pow(params.gamma0,2)* pow(phi,2)*axx)/b1*(R3-R2);

//         float alfapp = sp/(sigmax*sigmay);
//         float alfaxx = (tfsrow.betx/(sigmax*sigmay)) * (sx+sxp+sp*(pow(tfsrow.dx,2)/pow(tfsrow.betx,2) + pow(phi,2)));
// 		float alfayy = (tfsrow.bety/(sigmax*sigmay)) * (-2.0*R1+R2+R3);

// 		float alfap0 = alfapp * tfsrow.l / params.acceleratorLength;
//         float alfax0 = alfaxx * tfsrow.l / params.acceleratorLength;
//         float alfay0 = alfayy * tfsrow.l / params.acceleratorLength;

//         return make_float3(alfax0, alfay0, alfap0);
//     };
//  };


class STE_IBS
{
public:
	STE_IBS( ibsparameters&, thrust::device_vector<float6> , thrust::device_vector<tfsTableData>);
	// ~STE_IBS();

	
	template <typename Vector1, typename Vector2>
	void dense_histogram( const Vector1&, Vector2&, int ,float );

	float3 CalculateIBSGrowthRates( ibsparameters& , int ,  float , thrust::device_vector<tfsTableData> );
	float3 getIBSLifeTimes( float3 );
	float3 getIBSGrowthRates();

	__host__ float4 CalcIbsCoeff(ibsparameters& , int ,  float , float3 );
	
	thrust::device_vector<int> ParticleTimesToHistogram( thrust::device_vector<float6> , int , float );
	thrust::device_vector<float> HistogramToSQRTofCumul( thrust::device_vector<int> , float );

	void update( ibsparameters&  , thrust::device_vector<float6> , thrust::device_vector<tfsTableData> );

	float4 getIBScoeff();
	thrust::device_vector<int> getTimeHistogram();
	thrust::device_vector<float> getSqrtHistogram();

private:
	float6 CalcRMS( thrust::device_vector<float6> , int );

	float3 ibsGrowthRates;
	float4 ibscoeff;
	thrust::device_vector<int> histogramTime;
	thrust::device_vector<float> sqrthistogram; //denlon2k array in original code

};
#endif
