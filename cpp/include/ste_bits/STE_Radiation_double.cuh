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

// load double 6 datastructure
#include "STE_DataStructures_double.cuh"

// load tfsTableData datastructure
#include "STE_TFS_double.cuh"

struct radiationIntegralsParameters
{
	double betxRingAverage;
	double betyRingAverage;
	double acceleratorLength;
	double gammaTransition;
	double DipoleBendingRadius;
	double ParticleRadius; // nucleon or electron 
	double ParticleEnergy;
	double ParticleRestEnergy;
	double trev;
	double cq;
	double gammar;
	double eta;
	double omegas;
	double p0;

};

struct radiationIntegrals {
	double I2;
	double I3;
	double I4x;
	double I4y;
	double I5x;
	double I5y;
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

		double angle  = tfsAcceleratorElement.angle;
		double l      = tfsAcceleratorElement.l;
		double k1l    = tfsAcceleratorElement.k1l;
		double dy     = tfsAcceleratorElement.dy;
		double k1s    = tfsAcceleratorElement.k1sl;
		double alphax = tfsAcceleratorElement.alfx;
		double alphay = tfsAcceleratorElement.alfy;
		double betx   = tfsAcceleratorElement.betx;
		double bety   = tfsAcceleratorElement.bety;
		double dx     = tfsAcceleratorElement.dx;
		double dpx    = tfsAcceleratorElement.dpx;
		double dpy    = tfsAcceleratorElement.dpy;

		double rhoi = ( angle > 0.0) ? 0.5 * l / sin(angle/2) : 0.0;
		double ki = (l > 0.0) ? k1l / l : 0.0 ;

		outputIntegralsLattice.I2 = (rhoi > 0.0) ? l / pow(rhoi,2) : 0.0 ;
		outputIntegralsLattice.I3 = (rhoi > 0.0) ? l / pow(rhoi,3) : 0.0 ;
    
    	//  corrected to equations in accelerator handbook  Chao second edition p 220
    	outputIntegralsLattice.I4x = (rhoi > 0.0) ? ((dx / pow(rhoi,3)) + 2 * (ki * dx + (k1s / l) * dy) / rhoi) *l : 0.0;
    	outputIntegralsLattice.I4y = 0.0;

    	double gammax = (1.0 + pow(alphax,2)) / betx;
    	double gammay = (1.0 + pow(alphay,2)) / bety;

    	double Hx =  betx * pow(dpx,2) + 2. * alphax * dx * dpx + gammax * pow(dx,2);
    	double Hy =  bety * pow(dpy,2) + 2. * alphay * dy * dpy + gammay * pow(dy,2);

    	outputIntegralsLattice.I5x = (rhoi > 0.0) ? Hx * l / pow(rhoi,3) : 0.0  ;//* 2 * CUDA_PI_F / pow(radiationIntParameters.DipoleBendingRadius,2) * l;
    	outputIntegralsLattice.I5y = (rhoi > 0.0) ? Hy * l / pow(rhoi,3) : 0.0  ;

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
	double6 getDampingAndEquilib();

	radiationIntegrals getIntegrals( int );

	void setAverageRadiationPower( radiationIntegralsParameters, int);
	double getAverageRadiationPower();

	void printApproxIntegrals();
	void printLatticeIntegrals();
	void printDampingTimes();
	void printParameters( radiationIntegralsParameters );

	
	// ~STE_Radiation();

private:
	__host__ __device__ radiationIntegrals CalculateRadiationIntegralsApprox( radiationIntegralsParameters );
	radiationIntegrals CalculateRadiationIntegralsLatticeRing( thrust::device_vector<tfsTableData> , radiationIntegralsParameters );
	double6 CalculateRadiationDampingTimesAndEquilib( radiationIntegralsParameters , radiationIntegrals );
	
	void Init( radiationIntegralsParameters , thrust::device_vector<tfsTableData>);

	radiationIntegrals RadIntegralsApprox;
	radiationIntegrals RadIntegralsLattice;

	double6 DampingTimesAndEquilibrium;

	double AverageRadiationPower;
	int method;
};

#endif