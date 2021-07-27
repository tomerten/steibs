#ifndef SIMPARAMETERS_H
#define SIMPARAMETERS_H



typedef unsigned int uint;


// simulation parameters
struct SimParams
{
	// general settings
	// ****************

	// + random generator seed
	uint iseed;
	
	// + number of simulation turns
	uint numSimulationTurns;

	// + ratio of simulation to real turns (1 simulation turn is n machine turns)
	uint ratioSimToRealTurns;

	

	// + number of RF systems
	uint numRfSystems;

	// + harmonic numbers and voltages for the different rf systems
	float rfHarmonicNumbers[5];
	float rfVoltages[5];

	// different available ibs routines
	enum ibsroutine
	{
		PIWINSKI,
		PIWINSKISMOOTH,
		NAGAITSEV,
		BANE,
		NASH
	};

	// options for initial distributions
	enum transverseDistributionOptions
		{
			TRANSVERSEEXT,
			TRANSVERSEBIGAUSSIAN
		};

	enum longitudinalDistributionOptions
		{
			LONGITUDINALEXT,
			LONGITUDINALBIGAUSSIAN,
			SMOKERING,
			PSEUDOGAUSSIAN
		};

	transverseDistributionOptions tOpt;
	longitudinalDistributionOptions lOpt;

	// physics routine flags
	bool flagBetatron;
	bool flagSynchrotron;
	bool flagIbs;
	bool flagCollision;

	// input flags
	bool flagExternalDistribution;

	// output flags
	bool flagWriteCoordinates;
	bool flagWriteIbs;
	bool flagWriteEmit;

	

};

#endif