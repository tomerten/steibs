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

// load double 6 datastructure
// #include "STE_DataStructures.cuh"

// load tfsTableData datastructure
// #include "STE_TFS.cuh"

// load ibs header
#include "STE_IBS_double.cuh"

/* ************************************************************ */
/*                                                              */
/*            Numerical functions for IBS                       */
/*                                                              */
/* ************************************************************ */

// numerical recipes
// !Computes Carlson elliptic integral of the second kind, RD(x, y, z). x and y must be
// !nonnegative, and at most one can be zero. z must be positive. TINY must be at least twice
// !the negative 2/3 power of the machine overflow limit. BIG must be at most 0.1×ERRTOL
// !times the negative 2/3 power of the machine underflow limit.
__host__ __device__ double rd_s(double3 in) {
	double ERRTOL = 0.005;
	// double TINY   = 1.0e-25;
	// double BIG    = 4.5e21;
	double C1 = 3.0/14.0;
	double C2 = 1.0/6.0;
	double C3 = 9.0/22.0;
	double C4 = 3.0/26.0;
	double C5 = 0.25 * C3;
	double C6 = 1.5*C4;

	double xt  = in.x;
	double yt  = in.y;
	double zt  = in.z;
	double sum = 0.0;
	double fac = 1.0;
	int iter  = 0;

	double sqrtx, sqrty, sqrtz, alamb, ave, delx, dely, delz, maxi ;
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

	double ea = delx * dely;
	double eb = delz * delz;
	double ec = ea - eb;
	double ed = ea - 6.0 * eb;
	double ee = ed + ec + ec;
	double rd_s = 3.0 * sum + fac * (1.0 + ed *(-C1+C5*ed-C6*delz*ee)+delz*(C2*ee+delz*(-C3*ec+delz*C4*ea)))/(ave*sqrt(ave));
	return rd_s;
 };

__host__ __device__ double fmohl(double a, double b, double q, int fmohlNumPoints) {

      double result;
      double sum = 0;
      double du = 1.0/fmohlNumPoints;
      double u, cp, cq, dsum;

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


/* ************************************************************ */
/*                                                              */
/*            IBS calculation methods                           */
/*                                                              */
/* ************************************************************ */
__host__ __device__ double3 ibsPiwinskiSmooth(ibsparameters& params){
	double d;
	double xdisp = params.acceleratorLength / (2 * CUDA_PI_F * pow(params.gammaTransition,2));
	double rmsx  = sqrt(params.emitx * params.betx);
	double rmsy  = sqrt(params.emity * params.bety);

	double sigh2inv =  1.0 / pow(params.dponp,2)  + pow((xdisp / rmsx),2);
	double sigh = 1.0 / sqrt(sigh2inv);

	double atop = pow(params.ParticleRadius,2) * CUDA_C_F * params.numberRealParticles;
	double coeff = params.emitx * params.emity * params.sigs * params.dponp;
  	double abot = 64.0 * pow(CUDA_PI_F,2) * pow(params.betar,3) * pow(params.gamma0,4) * coeff;

  	// ca is Piwinski's A
  	double ca = atop/abot;
  	double a  = sigh * params.betx / (params.gamma0 * rmsx);
  	double b  = sigh * params.bety / (params.gamma0 * rmsy);

  	if (rmsx <=rmsy)
     	d = rmsx;
  	else
    	d = rmsy;
  
  	double q = sigh * params.betar * sqrt(2 * d / params.ParticleRadius);

  	double fmohlp = fmohl(a,b,q,params.fmohlNumPoints);
  	double fmohlx = fmohl(1/a,b/a,q/a,params.fmohlNumPoints);
  	double fmohly = fmohl(1/b,a/b,q/b,params.fmohlNumPoints);
  	double alfap0 = ca * fmohlp * pow((sigh/params.dponp),2);
  	double alfax0 = ca * (fmohlx + fmohlp * pow((xdisp * sigh / rmsx),2));
  	double alfay0 = ca * fmohly;

  	return make_double3(alfax0, alfay0, alfap0);
 };

struct ibsPiwinskiLattice{
 	ibsparameters params;
 	ibsPiwinskiLattice(ibsparameters& params) : params(params) {}

 // 	__host__ __device__
	// radiationIntegrals operator()(tfsTableData& tfsAcceleratorElement, radiationIntegralsParameters& radiationIntParameters) const 
 	__host__ __device__
    double3 operator()(tfsTableData& tfsrow) const
    {
    	double d;

    	double atop = pow(params.ParticleRadius,2) * CUDA_C_F * params.numberRealParticles;
		double coeff = params.emitx * params.emity * params.sigs * params.dponp;
      	double abot = 64.0 * pow(CUDA_PI_F,2) * pow(params.betar,3) * pow(params.gamma0,4) * coeff;

      	// ca is Piwinski's A
      	double ca = atop/abot;

      	double rmsx = sqrt(params.emitx * tfsrow.betx);
        double rmsy = sqrt(params.emity * tfsrow.bety);

        if (rmsx <= rmsy)
        	d = rmsx;
        else
        	d = rmsy;

        double sigh2inv =  1.0 / pow(params.dponp,2)  + pow((tfsrow.dx / rmsx),2);
		double sigh = 1.0 / sqrt(sigh2inv);

		double a = sigh * tfsrow.betx / (params.gamma0 * rmsx);
		double b = sigh * tfsrow.bety / (params.gamma0 * rmsy);
		double q = sigh * params.betar * sqrt(2 * d / params.ParticleRadius);

		double fmohlp = fmohl(a,b,q,params.fmohlNumPoints);
      	double fmohlx = fmohl(1/a,b/a,q/a,params.fmohlNumPoints);
      	double fmohly = fmohl(1/b,a/b,q/b,params.fmohlNumPoints);
      	double alfap0 = ca * fmohlp * pow((sigh/params.dponp),2) * tfsrow.l / params.acceleratorLength;
      	double alfax0 = ca * (fmohlx + fmohlp * pow((tfsrow.dx * sigh / rmsx),2)) * tfsrow.l / params.acceleratorLength;
      	double alfay0 = ca * fmohly * tfsrow.l / params.acceleratorLength;

      	return make_double3(alfax0, alfay0, alfap0);
    };
 };

struct ibsmodPiwinskiLattice{
	ibsparameters params;
 	ibsmodPiwinskiLattice(ibsparameters& params) : params(params) {}

 // 	__host__ __device__
	// radiationIntegrals operator()(tfsTableData& tfsAcceleratorElement, radiationIntegralsParameters& radiationIntParameters) const 
 	__host__ __device__
    double3 operator()(tfsTableData& tfsrow) const
    {
    	double d;

    	double atop = pow(params.ParticleRadius,2) * CUDA_C_F * params.numberRealParticles;
		double coeff = params.emitx * params.emity * params.sigs * params.dponp;
      	double abot = 64.0 * pow(CUDA_PI_F,2) * pow(params.betar,3) * pow(params.gamma0,4) * coeff;

      	// ca is Piwinski's A
      	double ca = atop/abot;

      	double H = (pow(tfsrow.dx,2)+ pow((tfsrow.betx * tfsrow.dpx - 0.5 * tfsrow.dx * (-2 * tfsrow.alfx)),2)) / tfsrow.betx;

      	double rmsx = sqrt(params.emitx * tfsrow.betx);
        double rmsy = sqrt(params.emity * tfsrow.bety);

        if (rmsx <= rmsy)
        	d = rmsx;
        else
        	d = rmsy;

        double sigh2inv =  1.0 / pow(params.dponp,2)  + (H / params.emitx);
		double sigh = 1.0 / sqrt(sigh2inv);

		double a = sigh * tfsrow.betx / (params.gamma0 * rmsx);
		double b = sigh * tfsrow.bety / (params.gamma0 * rmsy);
		double q = sigh * params.betar * sqrt(2 * d / params.ParticleRadius);

		double fmohlp = fmohl(a,b,q,params.fmohlNumPoints);
      	double fmohlx = fmohl(1/a,b/a,q/a,params.fmohlNumPoints);
      	double fmohly = fmohl(1/b,a/b,q/b,params.fmohlNumPoints);
      	double alfap0 = ca * fmohlp * pow((sigh/params.dponp),2) * tfsrow.l / params.acceleratorLength;
      	double alfax0 = ca * (fmohlx + fmohlp * pow((tfsrow.dx * sigh / rmsx),2)) * tfsrow.l / params.acceleratorLength;
      	double alfay0 = ca * fmohly * tfsrow.l / params.acceleratorLength;

      	return make_double3(alfax0, alfay0, alfap0);
    };
	};

struct ibsnagaitsev{
	ibsparameters params;
 	ibsnagaitsev(ibsparameters& params) : params(params) {}
 	__host__ __device__
    double3 operator()(tfsTableData& tfsrow) const
    {
    	double phi = tfsrow.dpx + (tfsrow.alfx * (tfsrow.dx/tfsrow.betx));
    	double axx = tfsrow.betx / params.emitx; 
        double ayy = tfsrow.bety / params.emity;

        double sigmax = sqrt( pow(tfsrow.dx,2) * pow(params.dponp,2) + params.emitx * tfsrow.betx);
        double sigmay = sqrt(params.emity * tfsrow.bety);
        double as     = axx * (pow(tfsrow.dx,2)/pow(tfsrow.betx,2) + pow(phi,2)) + (1/(pow(params.dponp,2)));

        double a1 = 0.5 * (axx + pow(params.gamma0,2) * as);
        double a2 = 0.5 * (axx - pow(params.gamma0,2) * as);

        double b1 = sqrt(pow(a2,2) + pow(params.gamma0,2) * pow(axx,2) * pow(phi,2));

        double lambda1 = ayy;
        double lambda2 = a1 + b1;
        double lambda3 = a1 - b1;

        double R1 = (1/lambda1) * rd_s(make_double3(1./lambda2,1./lambda3,1./lambda1));
        double R2 = (1/lambda2) * rd_s(make_double3(1./lambda3,1./lambda1,1./lambda2));
        double R3 = 3*sqrt((lambda1*lambda2)/lambda3)-(lambda1/lambda3)*R1-(lambda2/lambda3)*R2;

        double sp = (pow(params.gamma0,2)/2.0) * ( 2.0*R1 - R2*( 1.0 - 3.0*a2/b1 ) - R3*( 1.0 + 3.0*a2/b1 ));
		double sx = 0.50 * (2.0*R1 - R2*(1.0 + 3.0*a2/b1) -R3*(1.0 - 3.0*a2/b1));
		double sxp=(3.0 * pow(params.gamma0,2)* pow(phi,2)*axx)/b1*(R3-R2);

        double alfapp = sp/(sigmax*sigmay);
        double alfaxx = (tfsrow.betx/(sigmax*sigmay)) * (sx+sxp+sp*(pow(tfsrow.dx,2)/pow(tfsrow.betx,2) + pow(phi,2)));
		double alfayy = (tfsrow.bety/(sigmax*sigmay)) * (-2.0*R1+R2+R3);

		double alfap0 = alfapp * tfsrow.l / params.acceleratorLength;
        double alfax0 = alfaxx * tfsrow.l / params.acceleratorLength;
        double alfay0 = alfayy * tfsrow.l / params.acceleratorLength;

        return make_double3(alfax0, alfay0, alfap0);
    };
 };


 /* ************************************************************ */
 /*                                                              */
 /*            class implementation                              */
 /*                                                              */
 /* ************************************************************ */


STE_IBS::STE_IBS( ibsparameters&  params, thrust::device_vector<double6> bunch , thrust::device_vector<tfsTableData> tfsdata, double tsynchro ){
	
  /* Init of the IBS object */

  // update inpup parameters
	params.dponp = (CalcRMS( bunch, params.numberMacroParticles )).delta;

	// make a histogram for the distribution longitudinally
	histogramTime = ParticleTimesToHistogram(bunch,params.nbins,params.tauhat, tsynchro );

	// calculate the ibs growth coefficients used in the ibs routine
	ibsGrowthRates = CalculateIBSGrowthRates( params , params.methodIBS , params.phiSynchronous , tfsdata );

  // calculate the coefficients for multiplying with the growth rates
	ibscoeff  = CalcIbsCoeff( params , params.methodIBS , params.phiSynchronous, ibsGrowthRates );

	// combined the histogram and coefficients in a vector to use in ibs calculations
	// sqrt ( longitudinal coeff * cumul histogram ) - quantity representing the particle line density impact on momentum changes
	sqrthistogram = HistogramToSQRTofCumul(histogramTime,ibscoeff.w);
 }


template <typename Vector1, typename Vector2>
void STE_IBS::dense_histogram( const Vector1& input , Vector2& histogram , int nbins , double tauhat ){

  /* Function for creating a longitudinal histogram of the particle distribution */
  /* This is necessary to calculate the IBS kick to particles                    */
  /* Regions of high particle density have a higher scattering probability       */

  typedef typename Vector1::value_type ValueType; // input value type
  typedef typename Vector2::value_type IndexType; // histogram index type

  // copy input data (could be skipped if input is allowed to be modified)
  thrust::device_vector<ValueType> data(input);
  
  // sort data to bring equal elements together
  thrust::sort(data.begin(), data.end());

  // number of histogram bins is equal to the maximum value plus one
  IndexType num_bins = nbins+1;

  // resize histogram storage
  histogram.resize(num_bins);
  
  // find the end of each bin of values
  thrust::counting_iterator<IndexType> search_begin(0);
  thrust::upper_bound(data.begin(), data.end(),
                      search_begin, search_begin + num_bins,
                      histogram.begin());
  
  // compute the histogram by taking differences of the cumulative histogram
  thrust::adjacent_difference(histogram.begin(), histogram.end(),
                              histogram.begin());

 };


thrust::device_vector<int> STE_IBS::ParticleTimesToHistogram(thrust::device_vector<double6> data, int nbins, double tauhat, double ts ){

  /* Takes the integers produced by ParticleTimes to Integer Vector to produce a histogram or binned version */

	int n = data.size();
	thrust::device_vector<int> timecomponent(n);
	thrust::device_vector<int> histogram;

	// load the integer time components in a device vector
	thrust::transform(data.begin(),data.end(),thrust::make_constant_iterator(make_double3(tauhat,nbins,ts)),timecomponent.begin(),ParticleTimesToInteger());
 	
 	// binning the times
	dense_histogram(timecomponent,histogram,nbins,tauhat);

	return histogram;
 };


double6 STE_IBS::CalcRMS(thrust::device_vector<double6> distribution, int numberMacroParticles){
	double6 sum;

	sum.x= 0.0;
	sum.px=0.0;
	sum.y=0.0;
	sum.py=0.0;
	sum.t=0.0;
	sum.delta=0.0;
	double6 average = thrust::reduce(distribution.begin(),distribution.end(),sum,adddouble6());

	average.x     = - average.x / numberMacroParticles;
	average.px    = - average.px / numberMacroParticles;
	average.y     = - average.y / numberMacroParticles;
	average.py    = - average.py / numberMacroParticles;
	average.t     = - average.t / numberMacroParticles;
	average.delta = - average.delta / numberMacroParticles;

	// subtract shift -> t_synchronous
	thrust::transform(distribution.begin(), distribution.end(),thrust::make_constant_iterator(average),distribution.begin(),adddouble6());

	// square 
	thrust::transform(distribution.begin(),distribution.end(),distribution.begin(),squareFunctor<double6> ());
	
	sum.x= 0.0;
	sum.px=0.0;
	sum.y=0.0;
	sum.py=0.0;
	sum.t=0.0;
	sum.delta=0.0;

	// sum squares
	double6 MS = thrust::reduce(distribution.begin(),distribution.end(),sum,adddouble6());

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

double3 STE_IBS::CalculateIBSGrowthRates(ibsparameters& params, int method,  double tsynchro, thrust::device_vector<tfsTableData> tfsdata){
	double3 ibsgrowthrates;
	

	int m = tfsdata.size();
	thrust::device_vector<double3> ibsgrowthratestfs(m);
	

	// std::cout << "method = " << method << std::endl;
	// double rmsx = sqrt(params.emitx * params.betx);
	// double rmsy = sqrt(params.emity * params.bety);

	// get growth rates according to selected ibs method
  	switch(method)
  	{
  		case 0: {
  			ibsgrowthrates = ibsPiwinskiSmooth(params);
  			break;
  		}
  		case 1:{
        // std::cout << "method ok " << std::endl;
  			thrust::transform(tfsdata.begin(),tfsdata.end(),ibsgrowthratestfs.begin(),ibsPiwinskiLattice(params));
  			ibsgrowthrates = thrust::reduce(ibsgrowthratestfs.begin(),ibsgrowthratestfs.end(),make_double3(0.0,0.0,0.0),adddouble3());
  			break;
  		}
  		case 2:{
  			thrust::transform(tfsdata.begin(),tfsdata.end(),ibsgrowthratestfs.begin(),ibsmodPiwinskiLattice(params));
  			ibsgrowthrates = thrust::reduce(ibsgrowthratestfs.begin(),ibsgrowthratestfs.end(),make_double3(0.0,0.0,0.0),adddouble3());
  			break;
  		}
  		case 3: {
  			thrust::transform(tfsdata.begin(),tfsdata.end(),ibsgrowthratestfs.begin(),ibsnagaitsev(params));
  			ibsgrowthrates = thrust::reduce(ibsgrowthratestfs.begin(),ibsgrowthratestfs.end(),make_double3(0.0,0.0,0.0),adddouble3());
  			double nom = 0.5 * (params.numberRealParticles * pow(params.ParticleRadius,2) * CUDA_C_F * params.coulomblog);
  			double denom = (12.0 * CUDA_PI_F * pow(params.betar,3) * pow(params.gamma0,5) * params.sigs);
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
   //  std::cout << "growth rate x " << ibsgrowthrates.x << " " << 1/ibsgrowthrates.x<< std::endl;
   //  std::cout << "growth rate y " << ibsgrowthrates.y << " " << 1/ibsgrowthrates.y<< std::endl;
  	// std::cout << "growth rate z " << ibsgrowthrates.z << " " << 1/ibsgrowthrates.z<< std::endl;
  	// std::cout << "numberMacroParticles " << params.numberMacroParticles << std::endl;

  	double alfap = 2 * params.fracibstot * params.numberMacroParticles * params.timeratio * ibsgrowthrates.z / params.initnumberMacroParticles;
  	double alfax = 2 * params.fracibstot * params.numberMacroParticles * params.timeratio * ibsgrowthrates.x / params.initnumberMacroParticles;
  	double alfay = 2 * params.fracibstot * params.numberMacroParticles * params.timeratio * ibsgrowthrates.y / params.initnumberMacroParticles;

  	return make_double3(alfax,alfay,alfap);

 };

double3 STE_IBS::getIBSLifeTimes( double3 ibsGrowthRates ){
	return make_double3(1/ (ibsGrowthRates.x ) , 1 / (ibsGrowthRates.y) , 1 / (ibsGrowthRates.z) );
 };

__host__ double4 STE_IBS::CalcIbsCoeff(ibsparameters& params, int method,  double tsynchro, double3 ibsGrowthRates ){
	double coeffs,coeffx, coeffy;
	double alphaAverage;
	double coeffMulT;

	double alfax = ibsGrowthRates.x;
	double alfay = ibsGrowthRates.y;
	double alfap = ibsGrowthRates.z;

	double dtsamp2 = 2 * params.tauhat / params.nbins;
	double rmsdelta  = params.dponp; 
	double sigs  = params.sigs;


	double rmsx = sqrt(params.emitx * params.betx);
	double rmsy = sqrt(params.emity * params.bety);

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

    coeffMulT = sigs* 2* sqrt(CUDA_PI_F)/(params.numberMacroParticles * dtsamp2 * CUDA_C_F);



	return make_double4(coeffx,coeffy,coeffs, coeffMulT);
 };

thrust::device_vector<double> STE_IBS::HistogramToSQRTofCumul(thrust::device_vector<int> inputHistogram, double coeff){

  /* The name is a badly chose here as it was originally used to calculate cumulated distributions which      */
  /* turned out to be not necessary - the function takes a histogram as input, multiplies it with a constant  */
  /* vector before taking the sqrt of each element. This produces a vector that is used in the IBSNew routine */
  /* to multiply with particle momenta representing the IBS contribution                                      */


	int n = inputHistogram.size();
	thrust::device_vector<double> vcoeff(n);

	// fill constant vector
	thrust::fill(vcoeff.begin(),vcoeff.end(),coeff);
  // std::cout << " **************** ibs constant factor mult histogram  ************** " << std::endl;
  // std::copy(vcoeff.begin(),vcoeff.end(),std::ostream_iterator<double>(std::cout));
  // std::cout  << std::endl;

	// multiply with constant
	thrust::transform(inputHistogram.begin(),inputHistogram.end(),vcoeff.begin(),vcoeff.begin(),thrust::multiplies<double>());

  // std::cout << " **************** coeff * hist  ************** " << std::endl;
  // std::copy(vcoeff.begin(),vcoeff.end(),std::ostream_iterator<double>(std::cout));
  // std::cout  << std::endl << std::endl;

	// *************************************************************************************
  // this was wrong, no cumul taken in original code - leaving it for future reference and not make same mistake
	// cumulative sum
	// thrust::inclusive_scan(vcoeff.begin(),vcoeff.end(),cumul.begin());
  // thrust::for_each(cumul.begin(),cumul.end(),sqrtdoubleFunctor());
  // *************************************************************************************

	// take sqrt
	thrust::for_each(vcoeff.begin(),vcoeff.end(),sqrtdoubleFunctor());
  // std::cout << " **************** sqrt coeff * hist  ************** " << std::endl;
  // std::copy(vcoeff.begin(),vcoeff.end(),std::ostream_iterator<double>(std::cout));
  // std::cout  << std::endl << std::endl;

	return vcoeff;
 };

void STE_IBS::update( ibsparameters&  params, thrust::device_vector<double6> bunch ,  
  thrust::device_vector<tfsTableData> tfsdata, double tsynchro){

  /* updated IBS input parameters for applying IBS kicks to the particles */


	histogramTime  = ParticleTimesToHistogram( bunch , params.nbins , params.tauhat , tsynchro);
  // std::cout << "before : "<< ibsGrowthRates.x << " " << ibsGrowthRates.y << " " << ibsGrowthRates.z << std::endl;

	ibsGrowthRates = CalculateIBSGrowthRates( params , params.methodIBS , params.phiSynchronous , tfsdata );
  // std::cout << "after : " << ibsGrowthRates.x << " " << ibsGrowthRates.y << " " << ibsGrowthRates.z  << std::endl;

	ibscoeff       = CalcIbsCoeff( params , params.methodIBS , params.phiSynchronous , ibsGrowthRates );
  // std::cout << "coeff : " << ibscoeff.x << " " << ibscoeff.y << " " << ibscoeff.z << " " << ibscoeff.w  << std::endl;

	sqrthistogram  = HistogramToSQRTofCumul(histogramTime,ibscoeff.w);
 }

double4 STE_IBS::getIBScoeff(){
	return ibscoeff;
 }

thrust::device_vector<int> STE_IBS::getTimeHistogram(){
	return histogramTime;
 }

thrust::device_vector<double> STE_IBS::getSqrtHistogram(){
	return sqrthistogram;
 }

double3 STE_IBS::getIBSGrowthRates(){
	return ibsGrowthRates;
 }