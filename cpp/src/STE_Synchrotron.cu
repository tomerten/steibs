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

#include <iterator>
#include <sstream>
#include <fstream>
#include <algorithm>

// load double 6 datastructure
#include "STE_DataStructures_double.cuh"

// load tfsTableData datastructure
#include "STE_TFS_double.cuh"

// load synchrotron 
#include "STE_Synchrotron_double.cuh"


STE_Synchrotron::STE_Synchrotron( synchrotronParameters& params )
{
	setSynchronousPhase( params );
	setSynchronousPhaseNext( params );
	setTauhat( params );
	setSynchrotronTune( params );

}

void STE_Synchrotron::setSynchronousPhase( synchrotronParameters& params )
{
	/* debug */
	// std::cout << "in synch function search "<< params.search << std::endl;
	// std::cout << "in synch function searchWidth "<< params.searchWidth << std::endl;
	// std::cout << "in synch function voltages "<< params.voltages.x << std::endl;
	// std::cout << "in synch function harmonicNumbers "<< params.harmonicNumbers.x << std::endl;

	synchronousPhase = synchronousPhaseFunctorDeriv(params.energyLostPerTurn,
		params.voltages, params.harmonicNumbers, params.particleCharge,
		params.search, params.search - params.searchWidth, params.search + params.searchWidth);
	// std::cout << "in synch function  "<< synchronousPhase << std::endl;
}

void STE_Synchrotron::setSynchronousPhaseNext( synchrotronParameters& params )
{
	synchronousPhaseNext = synchronousPhaseFunctorDeriv(params.energyLostPerTurn,
		params.voltages, params.harmonicNumbers, params.particleCharge,
		params.search + params.searchWidth, params.search + params.searchWidth / 2 , 
		params.search + 2 * params.searchWidth);
}

// function to calculate the half bucket phase acceptance
void STE_Synchrotron::setTauhat( synchrotronParameters& params )
{
	tauhat = abs(( synchronousPhaseNext - synchronousPhase ) / ( params.harmonicNumbers.x * params.omega0));
}


// function to calculate synchrotron tune
void STE_Synchrotron::setSynchrotronTune( synchrotronParameters& params )
{
	double h0 = params.harmonicNumbers.x;
	double eta = params.eta;
	double charge = params.particleCharge;
	double p0 = params.p0;

	double v0 = params.voltages.x;
	double v1 = params.voltages.y;
	double v2 = params.voltages.z;

	double h1oh0 = params.harmonicNumbers.y / params.harmonicNumbers.x;
	double h2oh0 = params.harmonicNumbers.z / params.harmonicNumbers.x;

	double Omega2 = (
		h0 * eta * charge)/(2 * CUDA_PI_F * p0) * (v0 * cos(synchronousPhase) 
		+ v1 * (h1oh0) * cos((h1oh0) * synchronousPhase)
		+ v2 * (h2oh0) * cos((h2oh0) * synchronousPhase)
		);

	synchrotronTune =  sqrt(abs(Omega2));
}

double STE_Synchrotron::getSynchronousPhase()
{
	return synchronousPhase;
}

double STE_Synchrotron::getSynchronousPhaseNext()
{
	return synchronousPhaseNext;
}

double STE_Synchrotron::getTauhat()
{
	return tauhat;
}

double STE_Synchrotron::getSynchrotronTune()
{
	return synchrotronTune;
}
void STE_Synchrotron::printSynchrotron()
{
	std::cout << "********* SYNCHRONOUS PHASES ***********" << std::endl;
	std::cout << "Synchronous Phase      : " << synchronousPhase << std::endl;
	std::cout << "Synchronous Next Phase : " << synchronousPhaseNext  << std::endl;
	std::cout << "Tauhat                 : " << tauhat << std::endl;
	std::cout << "Synchrotron Tune       : " << synchrotronTune << std::endl;
	std::cout << "****************************************" << std::endl;
}