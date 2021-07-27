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

#include "STE_Radiation.cuh"
// #include "STE_DataStructures.cuh"
// #include "STE_DataStructures.cu"

// constructor
STE_Radiation::STE_Radiation( radiationIntegralsParameters params, thrust::device_vector<tfsTableData> tfstable , int methodint)
{
	Init(params, tfstable);
	method = methodint;
	setDampingAndEquilib(  params, methodint );
	printLatticeIntegrals();
	printDampingTimes(); // still ok
	setAverageRadiationPower( params , methodint);

}
// end constructor

// function to write radiation integrals to screen
__host__ std::ostream& operator<< (std::ostream& os, const radiationIntegrals& p)
{
	os << "I2  : " << std::setw(15) << p.I2  << std::endl;
	os << "I3  : " << std::setw(15) << p.I3  << std::endl;
	os << "I4x : " << std::setw(15) << p.I4x << std::endl;
	os << "I4y : " << std::setw(15) << p.I4y << std::endl;
	os << "I5x : " << std::setw(15) << p.I5x << std::endl;
	os << "I5y : " << std::setw(15) << p.I5y << std::endl;
	return os;
}

// function to calculate approximate radiation integrals
__host__ __device__ radiationIntegrals STE_Radiation:: CalculateRadiationIntegralsApprox( radiationIntegralsParameters radiationIntParameters )
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
}
// end function to calculate approximate radiation integrals

// function to calculate radiation integrals using lattice file
radiationIntegrals STE_Radiation::CalculateRadiationIntegralsLatticeRing(thrust::device_vector<tfsTableData> tfsData, radiationIntegralsParameters params)
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
// end function radiation integrals lattice

// function to calculate the damping times and equilibrium sizes
float6 STE_Radiation::CalculateRadiationDampingTimesAndEquilib(radiationIntegralsParameters params, radiationIntegrals integrals)
{
	float6 result;

	// Chao handbook second edition page 221 eq. 11
	// float CalphaEC   = params.ParticleRadius * CUDA_C_F / (3 * params.acceleratorLength);
	float CalphaEC   = params.ParticleRadius * CUDA_C_F / (3 * pow(params.ParticleRestEnergy,3)) * (pow(params.p0,3)/params.acceleratorLength);
	// std::cout << "CalphaEC :" << CalphaEC << std::endl;
	// std::cout << "I2 :" << integrals.I2 << std::endl;
	// std::cout << "I4x :" << integrals.I4x << std::endl;

	
	// extra factor 2 to get growth rates for emittances and not amplitudes (sigmas)
	float alphax = 2.0f * CalphaEC * integrals.I2 * (1.0f - integrals.I4x / integrals.I2);
	float alphay = 2.0f * CalphaEC * integrals.I2 * (1.0f - integrals.I4y / integrals.I2);
	float alphas = 2.0f * CalphaEC * integrals.I2 * (2.0f + (integrals.I4x + integrals.I4y) / integrals.I2);

	// longitudinal equilibrium
	// Chao handbook second edition page 221 eq. 19
	float sigEoE02 = params.cq * pow(params.gammar,2) * integrals.I3 / (2 * integrals.I2 + integrals.I4x + integrals.I4y);
	float sigsEquilib =  params.gammar * sigEoE02;
	//(CUDA_C_F * abs(params.eta) / params.omegas) * sqrt(sigEoE02);

	// Chao handbook second edition page 221 eq. 12
	float Jx = 1. - integrals.I4x / integrals.I2;
    float Jy = 1. - integrals.I4y / integrals.I2;

    // transverse equilibrium
    float EmitEquilibx = params.cq * pow(params.gammar,2) * integrals.I5x / (Jx * integrals.I2);
    float EmitEquiliby = params.cq * pow(params.gammar,2) * integrals.I5y / (Jy * integrals.I2);

    if (EmitEquiliby == 0.0)
   		EmitEquiliby = params.cq * params.betyRingAverage * integrals.I3 / (2 * Jy * integrals.I2);


    result.x     = 1.0 / alphax; // damping time returned in seconds
    // std::cout << "tx :" << result.x  << std::endl;
    result.px    = 1.0 / alphay;
    result.y     = 1.0 / alphas; 
    result.py    = EmitEquilibx;
    result.t     = EmitEquiliby;
    result.delta = sigsEquilib  /CUDA_C_F; // sigs returned in seconds 
   return result;

}

// function to return Radiation integrals selected on method
radiationIntegrals STE_Radiation::getIntegrals( int methodint )
{
	switch(methodint){
		case 1 : {
			return RadIntegralsApprox;
		}
		case 2 : {
			return RadIntegralsLattice;
		}

		default :
		{
			return RadIntegralsApprox;
		}
	}
} 
// end function get radiation integrals

// function to set damping times and equilib sizes
void STE_Radiation::setDampingAndEquilib( radiationIntegralsParameters params, int methodint )
{
	switch(methodint){
		case 1 :{
			DampingTimesAndEquilibrium =  CalculateRadiationDampingTimesAndEquilib( params, RadIntegralsApprox);
			break;
		}
		case 2 :{
			DampingTimesAndEquilibrium =  CalculateRadiationDampingTimesAndEquilib( params, RadIntegralsLattice);
			break;
		}
	}
}
// end function setting damping and equilib 

// initialization
void STE_Radiation::Init( radiationIntegralsParameters params, thrust::device_vector<tfsTableData> tfstable )
{
	// calculate approximate radiation integrals
	RadIntegralsApprox = CalculateRadiationIntegralsApprox( params );
	RadIntegralsLattice = CalculateRadiationIntegralsLatticeRing( tfstable, params );

}

// function for getting damping times and equilib
float6 STE_Radiation::getDampingAndEquilib()
{
	return DampingTimesAndEquilibrium;
}
// end get damping times and equilib

// function to calculate the average energy lost per turn due to radiation
void STE_Radiation::setAverageRadiationPower( radiationIntegralsParameters params , int methodint)
{
	// chao handbook page 218 eq 2
	float cgamma = (4 * CUDA_PI_F / 3) * (params.ParticleRadius / pow(params.ParticleRestEnergy/1.e9,3));
	// std::cout << "cgamma " << cgamma << std::endl; 
	// std::cout << "trev " << params.trev << std::endl; 
	switch(methodint){
		case 1 :{
			// multply 1.0e9 * trev - units are GeV/s => eV per turn 
			AverageRadiationPower = (CUDA_C_F * cgamma) / (2 * CUDA_PI_F * params.acceleratorLength) * pow(params.ParticleEnergy/1.e9,4) * RadIntegralsApprox.I2 * 1.0e9 * params.trev;
			break;
		}
		case 2 :{
			AverageRadiationPower = (CUDA_C_F * cgamma) / (2 * CUDA_PI_F * params.acceleratorLength) * pow(params.ParticleEnergy/1.e9,4) * RadIntegralsLattice.I2 * 1.0e9 * params.trev;
			break;
		}

	}
}
// end function average energy lost due to radiation
	
float STE_Radiation::getAverageRadiationPower()
{
	return AverageRadiationPower;
}


void STE_Radiation::printApproxIntegrals()
{
	std::cout << std::endl;
	std::cout << "********** RAD INTEGRALS APPROX ********" << std::endl; 
	std::cout << RadIntegralsApprox;
	std::cout << "****************************************" << std::endl; 
}

void STE_Radiation::printLatticeIntegrals()
{
	std::cout << std::endl;
	std::cout << "********** RAD INTEGRALS LATTICE ********" << std::endl; 
	std::cout << RadIntegralsLattice;
	std::cout << "****************************************" << std::endl; 
}

void STE_Radiation::printDampingTimes()
{
	std::cout << std::endl;
	std::cout << "********** DAMPING TIMES AND EQUILIB ********" << std::endl; 
	std::cout << std::setw(15) << "x" << std::setw(15) << "y" << std::setw(15) << "z" << std::endl;
	std::cout << std::setw(15) << DampingTimesAndEquilibrium.x << std::setw(15) << DampingTimesAndEquilibrium.px
	<< std::setw(15) << DampingTimesAndEquilibrium.y << std::endl;
	std::cout << std::setw(15) << DampingTimesAndEquilibrium.py << std::setw(15) << DampingTimesAndEquilibrium.t
	<< std::setw(15) << DampingTimesAndEquilibrium.delta << std::endl;
	std::cout << "Average Radiation Power : " << AverageRadiationPower << std::endl;
	std::cout << "*********************************************" << std::endl; 
}

void STE_Radiation::printParameters( radiationIntegralsParameters params)
{
	std::cout << std::endl;
	std::cout << "********** RAD PARAMETERS    ***************" << std::endl; 
	std::cout << "length          :" << params.acceleratorLength << std::endl;
	std::cout << "betx            :" << params.betxRingAverage << std::endl;
	std::cout << "bety            :" << params.betyRingAverage << std::endl;
	std::cout << "gammtr          :" << params.gammaTransition << std::endl;
	std::cout << "bendingradius   :" << params.DipoleBendingRadius << std::endl;
	std::cout << "particleradius  :" << params.ParticleRadius << std::endl;
	std::cout << "energy          :" << params.ParticleEnergy << std::endl;
	std::cout << "cq              :" << params.cq << std::endl;
	std::cout << "gammar          :" << params.gammar << std::endl;
	std::cout << "eta             :" << params.eta << std::endl;
	std::cout << "omegas          :" << params.omegas << std::endl;
	std::cout << "p0              :" << params.p0 << std::endl;
	std::cout << "*********************************************" << std::endl; 
}