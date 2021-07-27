#include "bunch.h"
#include "simParameters.cuh"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <cstdlib>

#ifndef CUDA_PI_F 
#define CUDA_PI_F 3.141592654f
#endif



// constructor
bunch::bunch(uint numMacroParticles, uint bunchId, float3 bunchEmittances, ):
	m_bInitialized(false),
	m_numParticles(numMacroParticles),
	m_bunchEmittances(bunchEmittances),
	{
		
		_initializeBunch(numMacroParticles,bunchId,bunchEmittances);
	}


// destructor
bunch::~bunch()
{
	_finalizeBunch();
	m_numParticles = 0;
}

// init bunch
void
bunch::_initializeBunch(uint numMacroParticles,uint bunchId, float3 bunchEmittances,transverseDistributionOptions tOpt, longitudinalDistributionOptions lOpt)
{
	// check if not already initialized
	assert(!m_bInitialized);
		
	switch (tOpt)
	{
		default:
		case TRANSVERSEBIGAUSSIAN:
		{
			thrust::host_vector<vec4> h_vertical(numMacroParticles);
			thrust::generate(h_vertical.begin(),h_vertical.end(),makeRandomTransverseBiGaussian(betx, emitx, bety,emitx))
			vec4 verticalRandom;

		}
	}

	// mark as initialized
	m_bInitialized = True;
}