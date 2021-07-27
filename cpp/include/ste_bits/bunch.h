/*
 * bunch.h
 *
 *  Created on: Dec 11, 2017
 *      Author: tmerten
 */

#ifndef BUNCH_H_
#define BUNCH_H_

#include "simParameters.cuh"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


// bunch class
class bunch
{
public:
	// constructor
	bunch(uint numMacroParticles, uint bunchId, float3 bunchEmittances, transverseDistributionOptions tOpt, longitudinalDistributionOptions lOpt);

	// destructor
	~bunch();


	enum bunchArray
		{
			POSITION,
			MOMENTUM
		};

	// set/reset the bunch distribution
	void reset(bunchConfig config);
	//, transverseDistributionOptions transverseOption, longitudinalDistributionOptions longitudinalOption


	// get/set position/momentum arrays
	float *getBunchArray(bunchArray array);
	void setBunchArray(bunchArray array, const float *data, int start, int count);

	// get number of particles in the bunch
	int getNumParticles() const
		{
			return m_numParticles;
		}

	// print (selection) particle distribution to screen/file
	void dumpParticles(uint start, uint count);

	void writeParticlesToFile(uint start, uint count, string filename);

	// get/set bucket number where bunch is located -- important for multi RF systems and collision order for colliders
	uint getBunchId();
	uint setBunchId();

private:
	void _initializeBunch(int numParticles, uint bunchId, float3 bunchEmittances, transverseDistributionOptions tOpt, longitudinalDistributionOptions lOpt);

	void _finalizeBunch();

	float3 m_bunchEmittances;

	bool m_bInitialized;
	uint m_numMacroParticles;


	vec6 h_particle;
	vec6 d_particle;


	//params
	SimParams m_params;
	
};
#endif /* BUNCH_H_ */
