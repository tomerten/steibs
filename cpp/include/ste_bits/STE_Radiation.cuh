//include guard
#ifndef RADIATION_H_INCLUDED
#define RADIATION_H_INCLUDED


// included dependencies
#include <map>
#include <iostream>
#include <iomanip>
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

#include <iterator>
#include <sstream>
#include <fstream>
#include <algorithm>

// load float 6 datastructure
#include "STE_DataStructures.cuh"

// load tfsTableData datastructure
#include "STE_TFS.cuh"

struct radiationIntegralsParameters
{
	float betxRingAverage;
	float betyRingAverage;
	float acceleratorLength;
	float gammaTransition;
	float DipoleBendingRadius;
	float ParticleRadius; // nucleon or electron 
	float ParticleEnergy;
	float ParticleRestEnergy;
	float trev;
	float cq;
	float gammar;
	float eta;
	float omegas;
	float p0;

};

struct radiationIntegrals {
	float I2;
	float I3;
	float I4x;
	float I4y;
	float I5x;
	float I5y;
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

		float rhoi = ( angle > 0.0) ? 0.5 * l / sin(angle/2) : 0.0;
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

class STE_Radiation
{
public:
	STE_Radiation( radiationIntegralsParameters, thrust::device_vector<tfsTableData>, int);

	// integer is for setting method to calculate radiation integrals
	// 1 : approx
	// 2 : lattice
	void setDampingAndEquilib( radiationIntegralsParameters, int );  
	float6 getDampingAndEquilib();

	radiationIntegrals getIntegrals( int );

	void setAverageRadiationPower( radiationIntegralsParameters, int);
	float getAverageRadiationPower();

	void printApproxIntegrals();
	void printLatticeIntegrals();
	void printDampingTimes();
	void printParameters( radiationIntegralsParameters );

	
	// ~STE_Radiation();

private:
	__host__ __device__ radiationIntegrals CalculateRadiationIntegralsApprox( radiationIntegralsParameters );
	radiationIntegrals CalculateRadiationIntegralsLatticeRing( thrust::device_vector<tfsTableData> , radiationIntegralsParameters );
	float6 CalculateRadiationDampingTimesAndEquilib( radiationIntegralsParameters , radiationIntegrals );
	
	void Init( radiationIntegralsParameters , thrust::device_vector<tfsTableData>);

	radiationIntegrals RadIntegralsApprox;
	radiationIntegrals RadIntegralsLattice;

	float6 DampingTimesAndEquilibrium;

	float AverageRadiationPower;
	int method;
};

#endif