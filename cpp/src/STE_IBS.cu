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
#include <boost/math/tools/roots.hpp>
#include <thrust/tuple.h>
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
// #include "STE_DataStructures.cuh"

// load tfsTableData datastructure
// #include "STE_TFS.cuh"

// load ibs header
#include "STE_IBS.cuh"

// numerical recipes
// !Computes Carlson elliptic integral of the second kind, RD(x, y, z). x and y must be
// !nonnegative, and at most one can be zero. z must be positive. TINY must be at least twice
// !the negative 2/3 power of the machine overflow limit. BIG must be at most 0.1×ERRTOL
// !times the negative 2/3 power of the machine underflow limit.
__host__ __device__ float rd_s(float3 in) {
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

__host__ __device__ float fmohl(float a, float b, float q, int fmohlNumPoints) {

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

     };



__host__ __device__ float3 ibsPiwinskiSmooth(ibsparameters& params){
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

struct ibsPiwinskiLattice{
 	ibsparameters params;
 	ibsPiwinskiLattice(ibsparameters& params) : params(params) {}

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

struct ibsmodPiwinskiLattice{
	ibsparameters params;
 	ibsmodPiwinskiLattice(ibsparameters& params) : params(params) {}

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

struct ibsnagaitsev{
	ibsparameters params;
 	ibsnagaitsev(ibsparameters& params) : params(params) {}
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



/* ******************** */
/* class implementation */
/* ******************** */

STE_IBS::STE_IBS( ibsparameters&  params, thrust::device_vector<float6> bunch , thrust::device_vector<tfsTableData> tfsdata){
	// update inpup parameters
	params.dponp = (CalcRMS( bunch, params.numberMacroParticles )).delta;

	// make a histogram for the distribution longitudinally
	histogramTime      = ParticleTimesToHistogram(bunch,params.nbins,params.tauhat);

	// calculate the ibs growth coefficients used in the ibs routine
	ibsGrowthRates = CalculateIBSGrowthRates( params , params.methodIBS , params.phiSynchronous , tfsdata );

	ibscoeff           = CalcIbsCoeff( params , params.methodIBS , params.phiSynchronous, ibsGrowthRates );

	// combined the histogram and coefficients in a vector to use in ibs calculations
	// sqrt ( longitudinal coeff * cumul histogram ) - quantity representing the particle line density impact on momentum changes
	sqrthistogram = HistogramToSQRTofCumul(histogramTime,ibscoeff.w);
 }



template <typename Vector1, typename Vector2>
void STE_IBS::dense_histogram( const Vector1& input , Vector2& histogram , int nbins , float tauhat ){
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
 };

// takes the integers produced by ParticleTimes to Integer to produce a histogram or binned version
// necessary for collision and intra beam scattering routines

thrust::device_vector<int> STE_IBS::ParticleTimesToHistogram(thrust::device_vector<float6> data, int nbins, float tauhat){
	int n = data.size();
	thrust::device_vector<int> timecomponent(n);
	thrust::device_vector<int> histogram;

	// load the integer time components in a device vector
	thrust::transform(data.begin(),data.end(),thrust::make_constant_iterator(make_float2(tauhat,nbins)),timecomponent.begin(),ParticleTimesToInteger());
 	
 	// binning the times
	dense_histogram(timecomponent,histogram,nbins,tauhat);

	return histogram;
 };



float6 STE_IBS::CalcRMS(thrust::device_vector<float6> distribution, int numberMacroParticles){
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

float3 STE_IBS::CalculateIBSGrowthRates(ibsparameters& params, int method,  float tsynchro, thrust::device_vector<tfsTableData> tfsdata){
	float3 ibsgrowthrates;
	

	int m = tfsdata.size();
	thrust::device_vector<float3> ibsgrowthratestfs(m);
	

	// std::cout << "method = " << method << std::endl;
	// float rmsx = sqrt(params.emitx * params.betx);
	// float rmsy = sqrt(params.emity * params.bety);

	// get growth rates according to selected ibs method
  	switch(method)
  	{
  		case 0: {
  			ibsgrowthrates = ibsPiwinskiSmooth(params);
  			break;
  		}
  		case 1:{
        std::cout << "method ok " << std::endl;
  			thrust::transform(tfsdata.begin(),tfsdata.end(),ibsgrowthratestfs.begin(),ibsPiwinskiLattice(params));
  			ibsgrowthrates = thrust::reduce(ibsgrowthratestfs.begin(),ibsgrowthratestfs.end(),make_float3(0.0,0.0,0.0),addFloat3());
  			break;
  		}
  		case 2:{
  			thrust::transform(tfsdata.begin(),tfsdata.end(),ibsgrowthratestfs.begin(),ibsmodPiwinskiLattice(params));
  			ibsgrowthrates = thrust::reduce(ibsgrowthratestfs.begin(),ibsgrowthratestfs.end(),make_float3(0.0,0.0,0.0),addFloat3());
  			break;
  		}
  		case 3: {
  			thrust::transform(tfsdata.begin(),tfsdata.end(),ibsgrowthratestfs.begin(),ibsnagaitsev(params));
  			ibsgrowthrates = thrust::reduce(ibsgrowthratestfs.begin(),ibsgrowthratestfs.end(),make_float3(0.0,0.0,0.0),addFloat3());
  			float nom = 0.5 * (params.numberRealParticles * pow(params.ParticleRadius,2) * CUDA_C_F * params.coulomblog);
  			float denom = (12.0 * CUDA_PI_F * pow(params.betar,3) * pow(params.gamma0,5) * params.sigs);
  			ibsgrowthrates.x = ibsgrowthrates.x / params.emitx *  nom / denom;
  			ibsgrowthrates.y = ibsgrowthrates.y / params.emity *  nom / denom;
  			ibsgrowthrates.z = ibsgrowthrates.z / pow(params.dponp,2) *  nom / denom;
  			
  			break;
  		}
  		

  	};

  	// uncomment for debugging
  	// std::cout << "fracibstot " << params.fracibstot << std::endl;
  	// std::cout << "real part " << params.numberRealParticles << std::endl;
  	// std::cout << "timeRatio " << params.timeratio << std::endl;
  	// std::cout << "growth rate z " << ibsgrowthrates.z << std::endl;
  	// std::cout << "numberMacroParticles " << params.numberMacroParticles << std::endl;

  	float alfap = 2 * params.fracibstot * params.numberRealParticles * params.timeratio * ibsgrowthrates.z / params.numberMacroParticles;
  	float alfax = 2 * params.fracibstot * params.numberRealParticles * params.timeratio * ibsgrowthrates.x / params.numberMacroParticles;
  	float alfay = 2 * params.fracibstot * params.numberRealParticles * params.timeratio * ibsgrowthrates.y / params.numberMacroParticles;

  	return make_float3(alfax,alfay,alfap);

 };

float3 STE_IBS::getIBSLifeTimes( float3 ibsGrowthRates )
{
	return make_float3( ibsGrowthRates.x / (2 * 1) , ibsGrowthRates.y / (2 * 1) , ibsGrowthRates.z / (2 * 1) );
};

__host__ float4 STE_IBS::CalcIbsCoeff(ibsparameters& params, int method,  float tsynchro, float3 ibsGrowthRates )
{
	float coeffs,coeffx, coeffy;
	float alphaAverage;
	float coeffMulT;

	float alfax = ibsGrowthRates.x;
	float alfay = ibsGrowthRates.y;
	float alfap = ibsGrowthRates.z;

	float dtsamp2 = 2 * params.tauhat / params.nbins;
	float rmsdelta  = params.dponp; 
	float sigs  = params.sigs;


	float rmsx = sqrt(params.emitx * params.betx);
	float rmsy = sqrt(params.emity * params.bety);

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

thrust::device_vector<float> STE_IBS::HistogramToSQRTofCumul(thrust::device_vector<int> inputHistogram, float coeff)
{
	int n = inputHistogram.size();
	thrust::device_vector<float> vcoeff(n);
	// thrust::device_vector<float> vcoeff2;
	//thrust::device_vector<float> cumul(n);
	// thrust::device_vector<int> outputhistogram;

	// fill constant vector
	thrust::fill(vcoeff.begin(),vcoeff.end(),coeff);

	// multiply with constant
	thrust::transform(inputHistogram.begin(),inputHistogram.end(),vcoeff.begin(),vcoeff.begin(),thrust::multiplies<float>());

	// this was wrong, no cumul taken in original code
	// cumulative sum
	//thrust::inclusive_scan(vcoeff.begin(),vcoeff.end(),cumul.begin());

	// take sqrt
	//thrust::for_each(cumul.begin(),cumul.end(),sqrtFloatFunctor());
	thrust::for_each(vcoeff.begin(),vcoeff.end(),sqrtFloatFunctor());

	return vcoeff;
};

void STE_IBS::update( ibsparameters&  params, thrust::device_vector<float6> bunch ,  thrust::device_vector<tfsTableData> tfsdata)
{
	histogramTime  = ParticleTimesToHistogram( bunch , params.nbins , params.tauhat );
  // std::cout << "before : "<< ibsGrowthRates.x << " " << ibsGrowthRates.y << " " << ibsGrowthRates.z << std::endl;
	ibsGrowthRates = CalculateIBSGrowthRates( params , params.methodIBS , params.phiSynchronous , tfsdata );
  // std::cout << "after : " << ibsGrowthRates.x << " " << ibsGrowthRates.y << " " << ibsGrowthRates.z  << std::endl;
	ibscoeff       = CalcIbsCoeff( params , params.methodIBS , params.phiSynchronous , ibsGrowthRates );
	sqrthistogram  = HistogramToSQRTofCumul(histogramTime,ibscoeff.w);
}

float4 STE_IBS::getIBScoeff()
{
	return ibscoeff;
}


thrust::device_vector<int> STE_IBS::getTimeHistogram()
{
	return histogramTime;
}


thrust::device_vector<float> STE_IBS::getSqrtHistogram()
{
	return sqrthistogram;
}

float3 STE_IBS::getIBSGrowthRates(){
	return ibsGrowthRates;
}