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

#include <map>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <math.h>
#include <boost/math/tools/roots.hpp>
#include <thrust/tuple.h>
#include <string>

#include "ste_global_functions.cu"
// #include "read_tfs.cu"

#include <vector>
#ifndef CUDA_PI_F 
#define CUDA_PI_F 3.141592654f
#endif

#ifndef CUDA_C_F
#define CUDA_C_F 299792458.0f
#endif

using namespace std;

/* Calculate radiation damping times and equilibrium emittances if not given manually
 * Reference: Chao, Tigner: Handbook of Accelerator physics and engineering, (1998) page 186
 *
 * uses radiationIntegrals struct -> see ste_global_functions.cu
*/

// __host__ __device__ 
// radiationIntegrals CalculateRadiationIntegralsApprox(radiationIntegralsParameters radiationIntParameters)
// {
// 	radiationIntegrals outputIntegralsApprox;
// 	// growth rates
// 	float alphax = 0.0;
// 	float alphay = 0.0;

// 	float gammax = (1.0 +  pow(alphax,2)) / radiationIntParameters.betxRingAverage;
// 	float gammay = (1.0 +  pow(alphay,2)) / radiationIntParameters.betyRingAverage;

// 	float Dx = radiationIntParameters.acceleratorLength / (2 * CUDA_PI_F * radiationIntParameters.gammaTransition);
// 	float Dy = 0.0;

// 	float Dxp = 0.1; // should find an approximation formula. However not very important
// 	float Dyp = 0.0;

// 	float Hx = (radiationIntParameters.betxRingAverage * Dxp + 2 * alphax * Dx * Dxp + gammax * Dx);
//     float Hy = (radiationIntParameters.betyRingAverage * Dyp + 2 * alphay * Dy * Dyp + gammay * Dy);

//     //  define smooth approximation of radiation integrals
//     outputIntegralsApprox.I2 = 2 * CUDA_PI_F / radiationIntParameters.DipoleBendingRadius;
//     outputIntegralsApprox.I3 = 2 * CUDA_PI_F / pow(radiationIntParameters.DipoleBendingRadius,2);
//     outputIntegralsApprox.I4x = 0.0;
//     outputIntegralsApprox.I4y = 0.0;
//     outputIntegralsApprox.I5x = Hx * 2 * CUDA_PI_F / pow(radiationIntParameters.DipoleBendingRadius,2);
//     outputIntegralsApprox.I5y = Hy * 2 * CUDA_PI_F / pow(radiationIntParameters.DipoleBendingRadius,2);

    
//     return outputIntegralsApprox;

// };


// struct CalculateRadiationIntegralsLatticeElement
// {


// 	__host__ __device__
// 	radiationIntegrals operator()(tfsTableData& tfsAcceleratorElement, radiationIntegralsParameters& radiationIntParameters) const 
// 	{
// 		radiationIntegrals outputIntegralsLattice;

// 		float angle  = tfsAcceleratorElement.angle;
// 		float l      = tfsAcceleratorElement.l;
// 		float k1l    = tfsAcceleratorElement.k1l;
// 		float dy     = tfsAcceleratorElement.dy;
// 		float k1s    = tfsAcceleratorElement.k1sl;
// 		float alphax = tfsAcceleratorElement.alfx;
// 		float alphay = tfsAcceleratorElement.alfy;
// 		float betx   = tfsAcceleratorElement.betx;
// 		float bety   = tfsAcceleratorElement.bety;
// 		float dx     = tfsAcceleratorElement.dx;
// 		float dpx    = tfsAcceleratorElement.dpx;
// 		float dpy    = tfsAcceleratorElement.dpy;

// 		float rhoi = ( angle > 0.0) ? l /angle : 0.0;
// 		float ki = (l > 0.0) ? k1l / l : 0.0 ;

// 		outputIntegralsLattice.I2 = (rhoi > 0.0) ? l / pow(rhoi,2) : 0.0 ;
// 		outputIntegralsLattice.I3 = (rhoi > 0.0) ? l / pow(rhoi,3) : 0.0 ;
    
//     	//  corrected to equations in accelerator handbook  Chao second edition p 220
//     	outputIntegralsLattice.I4x = (rhoi > 0.0) ? ((dx / pow(rhoi,3)) + 2 * (ki * dx + (k1s / l) * dy) / rhoi) *l : 0.0 ;
//     	outputIntegralsLattice.I4y = 0.0;

//     	float gammax = (1.0 + pow(alphax,2)) / betx;
//     	float gammay = (1.0 + pow(alphay,2)) / bety;

//     	float Hx =  betx * pow(dpx,2) + 2. * alphax * dx * dpx + gammax * pow(dx,2);
//     	float Hy =  bety * pow(dpy,2) + 2. * alphay * dy * dpy + gammay * pow(dy,2);

//     	outputIntegralsLattice.I5x = Hx * 2 * CUDA_PI_F / pow(radiationIntParameters.DipoleBendingRadius,2) * l;
//     	outputIntegralsLattice.I5y = Hy * 2 * CUDA_PI_F / pow(radiationIntParameters.DipoleBendingRadius,2) * l;

// 		return outputIntegralsLattice;
// 	}
// };


// radiationIntegrals CalculateRadiationIntegralsLatticeRing(thrust::device_vector<tfsTableData> tfsData, radiationIntegralsParameters params)
// {
// 	int n = tfsData.size();
// 	thrust::device_vector<radiationIntegrals> radiationIntegralsPerElement(n);

// 	thrust::transform(tfsData.begin(),tfsData.end(),thrust::make_constant_iterator(params),radiationIntegralsPerElement.begin(),CalculateRadiationIntegralsLatticeElement());

// 	radiationIntegrals initsum;
// 	initsum.I2  = 0.0;
// 	initsum.I3  = 0.0;
// 	initsum.I4x = 0.0;
// 	initsum.I4y = 0.0;
// 	initsum.I5x = 0.0;
// 	initsum.I5y = 0.0;

// 	radiationIntegrals total = thrust::reduce(radiationIntegralsPerElement.begin(),radiationIntegralsPerElement.end(),initsum,addRadiationIntegralsElements());

// 	return total;
// }

// float6 CalculateRadiationDampingTimesAndEquilib(radiationIntegralsParameters params, radiationIntegrals integrals)
// {
// 	float6 result;

// 	// Chao handbook second edition page 221 eq. 11
// 	// float CalphaEC   = params.ParticleRadius * CUDA_C_F / (3 * params.acceleratorLength);
// 	float CalphaEC   = params.ParticleRadius * CUDA_C_F / (3 * pow(CUDA_ELECTRON_REST_E_F,3)) * (pow(params.p0,3)/params.acceleratorLength);
	
// 	// extra factor 2 to get growth rates for emittances and not amplitudes (sigmas)
// 	float alphax = 2.0f * CalphaEC * integrals.I2 * (1.0f - integrals.I4x / integrals.I2);
// 	float alphay = 2.0f * CalphaEC * integrals.I2 * (1.0f - integrals.I4y / integrals.I2);
// 	float alphas = 2.0f * CalphaEC * integrals.I2 * (2.0f + (integrals.I4x + integrals.I4y) / integrals.I2);

// 	// longitudinal equilibrium
// 	// Chao handbook second edition page 221 eq. 19
// 	float sigEoE02 = params.cq * pow(params.gammar,2) * integrals.I3 / (2 * integrals.I2 + integrals.I4x + integrals.I4y);
// 	float sigsEquilib = (CUDA_C_F * abs(params.eta) / params.omegas) * sqrt(sigEoE02);

// 	// Chao handbook second edition page 221 eq. 12
// 	float Jx = 1. - integrals.I4x / integrals.I2;
//    float Jy = 1. - integrals.I4y / integrals.I2;

//    // transverse equilibrium
//    float EmitEquilibx = params.cq * pow(params.gammar,2) * integrals.I5x / (Jx * integrals.I2);
//    float EmitEquiliby = params.cq * pow(params.gammar,2) * integrals.I5y / (Jy * integrals.I2);

//    if (EmitEquiliby == 0.0)
//    	EmitEquiliby = params.cq * params.betyRingAverage * integrals.I3 / (2 * Jy * integrals.I2);


//    result.x     = 1.0 / alphax; // damping time returned in seconds
//    result.px    = 1.0 / alphay;
//    result.y     = 1.0 / alphas; 
//    result.py    = EmitEquilibx;
//    result.t     = EmitEquiliby;
//    result.delta = sigsEquilib/CUDA_C_F; // sigs returned in seconds 
//    return result;

// }
// need to calculate particle radius seperatly !!!


struct RadiationDampingRoutineFunctor
{
	RadiationDampingRoutineFunctor(float6 radiationDampParams, float trev, float timeratio, float seed) : radiationDampParams(radiationDampParams), trev(trev) ,timeratio(timeratio), seed(seed) {}
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


int main(int argc, char const *argv[])
{	
	double acceleratorLength = 240.00839; // length in meter
	double gammar = 3326.817037; // relativistic gamma 
	double eta = 0.0007038773471 - 1 / pow(gammar ,2); // slip factor approx alpha - 1/ gammar**2
	double betar = sqrt(1-1 / pow(gammar ,2)); // relativistic beta
	double trev = acceleratorLength / (betar * CUDA_C_F);
	double omega0 = (2 * CUDA_PI_F) / trev;
	double p0 = 1.7e9;

	radiationIntegrals radtest;
	radiationIntegralsParameters intpars;
	intpars.betxRingAverage = 10.0;
	intpars.betyRingAverage = 20.0;
	intpars.acceleratorLength = 240.00839;
	intpars.gammaTransition =  1/ sqrt(0.0007038773471);
	intpars.DipoleBendingRadius = 40.0;
	intpars.ParticleRadius = CUDA_C_R_ELECTRON;
	intpars.ParticleEnergy = 1.7e9;
	// Chao handbook second edition page 221 eq. 20
	intpars.cq = (55.0/(32.0 * sqrt(3))) * (CUDA_HBAR_F * CUDA_C_F) / (CUDA_ELECTRON_REST_E_F);
	intpars.gammar = 3326.817037;
	intpars.omegas = 0.0565621 * omega0;
	intpars.eta = eta;
	intpars.p0 = p0;

	radtest = CalculateRadiationIntegralsApprox(intpars);
	cout << radtest;

	vector<vector<string> > out;
	vector<vector<float> > fout;

	string in = "/home/tmerten/mad/2017-12-21/twiss/Long-corrected-LongPMMM-2017-12-21.tfs";
	out = ReadTfsTable(in,false);
	map<string, int> maptest = mapOfColumns(out[0],false);
	fout = TfsTableConvertStringToFloat(out,maptest);
	thrust::device_vector<tfsTableData> testdata;
	testdata = TfsTableToDevice(fout);
	std::copy(testdata.begin(),testdata.begin()+10,std::ostream_iterator<tfsTableData>(std::cout, "\n"));
	
	
	radiationIntegrals total =  CalculateRadiationIntegralsLatticeRing(testdata, intpars);
	cout << total ;

	cout << "Radiation damping times and equilibrium" << endl;
	cout << intpars.cq<< endl;
	cout << intpars.ParticleRadius * CUDA_C_F / (3 * intpars.acceleratorLength) << endl;
	cout << intpars.ParticleRadius * CUDA_C_F / (3 * pow(CUDA_ELECTRON_REST_E_F,3))  << endl;
	cout << intpars.ParticleRadius * CUDA_C_F / (3 * pow(CUDA_ELECTRON_REST_E_F,3)) * (pow(p0,3)/intpars.acceleratorLength)  << endl;
	float6 timesequilib = CalculateRadiationDampingTimesAndEquilib(intpars, total);
	cout << timesequilib<< endl;
	intpars.omegas = 0.00784862 * omega0;
	timesequilib = CalculateRadiationDampingTimesAndEquilib(intpars, total);
	cout << timesequilib<< endl;


	return 0;
}




// does radiation damping and quantum excitation once per turn

// implicit none
// integer, intent(inout) :: np
// !f2py intent(in,out) :: np
// double precision, intent(inout):: iseed
// !f2py intent(in,out) :: iseed
// integer ::k
// double precision, intent(in) :: tratio,trev,tradlong,tradperp,siglong,sigperp
// !f2py intent(in) :: tratio,trev,tradlong,siglong,sigperp
// double precision :: coeffdecaylong,coeffexcitelong,coeffgrow,coeffdecay
// double precision, intent(inout), dimension(np) ::x,px,y,py,pt
// !f2py intent(in,out) :: x,px,y,py,pt
// double precision,external :: ran
// integer, dimension(:), allocatable :: seed
// integer :: n
// ! init random generator
// call random_seed(size=n)
// allocate(seed(n))
// seed(1) = INT(iseed)
// call random_seed(put=seed)
//       coeffdecaylong  = 1 - ((trev / tradlong) * tratio)
        
//       ! excitation uses a uniform deviate on [-1:1]
//       coeffexcitelong = siglong * sqrt(3.) * sqrt(2 * (trev / tradlong) * tratio)
        
//       ! tradperp is the damping time for EMITTANCE, therefore need to multiply by 2
//       ! assume same damping in horizontal and vertical plane (I4x,I4y<<I2)
//       coeffdecay      = 1 - ((trev /(2 * tradperp)) * tratio)
        
//       ! exact     coeffgrow= sigperp*sqrt(3.)*sqrt(1-coeffdecay**2)
//       ! but trev << tradperp so
//       coeffgrow       = sigperp * sqrt(3.) * sqrt(2 * (trev /(2 * tradperp)) * tratio)


// ! skip if transverse damping time is not positive
//       if(tradperp.le.0)return

//       do k=1,np
// ! longitudinal
//          call random_number(iseed)
//          pt(k) = pt(k)*coeffdecaylong +coeffexcitelong*(2*iseed-1)
// ! transverse
//          x(k)  = coeffdecay*x(k) + (2*iseed-1)*coeffgrow
//          px(k) = coeffdecay*px(k)+(2*iseed-1)*coeffgrow
//          y(k)  = coeffdecay*y(k) + (2*iseed-1)*coeffgrow
//          py(k) = coeffdecay*py(k)+(2*iseed-1)*coeffgrow
//       enddo
//       return
// end

