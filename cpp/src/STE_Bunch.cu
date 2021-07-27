
#include <cstdlib>
#include <cmath>
#include <math.h>
#include <string>
#include <vector>
#include <set>
#include <time.h>
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

// load float 6 datastructure
// #include "STE_DataStructures.cuh"

// load tfsTableData datastructure
// #include "STE_TFS.cuh"

// load ibs
// #include "STE_IBS.cuh"

// load bunch header
#include "STE_Bunch.cuh"


#include "STE_Radiation.cuh"
// #include "STE_Longitudinal_Hamiltonian.cuh"
// #include "STE_Synchrotron.cuh"
// #include "STE_Random.cuh"
__host__ __device__ float VoltageTripleRf(float phi, float3 voltages, float3 harmonicNumbers){
	float volt1, volt2, volt3;
	volt1 =  voltages.x * sin(phi);
	volt2 =  voltages.y * sin((harmonicNumbers.y / harmonicNumbers.x) * phi);
	volt3 =  voltages.z * sin((harmonicNumbers.z / harmonicNumbers.x) * phi);
	return volt1 + volt2 + volt3;
}; 

__host__ __device__ float tcoeff( hamiltonianParameters& params )
{
	return  (params.angularFrequency * params.eta * params.harmonicNumbers.x);
}

__host__ __device__ float pcoeff( hamiltonianParameters& params, float voltage ){
	return  (params.angularFrequency * voltage * params.particleCharge) / (2 * CUDA_PI_F * params.p0 * params.betar);
};




struct RfUpdateRoutineFunctor  {
	RfUpdateRoutineFunctor( float fmix, longitudinalParameters& params , hamiltonianParameters& hparams  , ibsparameters& iparams, int bucket , float energylost): 
	fmix(fmix), params(params),  hparams(hparams), iparams(iparams), bucket(bucket), energylost(energylost) {}

	__host__ __device__ float6 operator()(float6 particle)
	{
		float6 result;

		// using parameters to generate parameters for functions 
		float3 voltages        = make_float3(params.v0, params.v1, params.v2);
		float3 harmonicNumbers = make_float3(params.h0, params.h1, params.h2);
		float tcoeffValue      = tcoeff(hparams) / (params.h0 * params.omega0);

		// Lee Third edition page 233 eqs 3.6 3.16 3.13
		
		// the phase  = h0 omega0 t
		float phi = params.h0 * params.omega0 * particle.t ;

		// Delta delta 
		float pcoeffValue = pcoeff(hparams , VoltageTripleRf(phi, voltages, harmonicNumbers) - 
			VoltageTripleRf(params.phiSynchronous - bucket * 2 * CUDA_PI_F, voltages, harmonicNumbers));

		result.x     = particle.x;
		result.px    = particle.px;
		result.y     = particle.y;
		result.py    = particle.py;
		// result.t     = particle.t + fmix * 2 * CUDA_PI_F * params.eta * particle.delta / params.omega0;
		result.t     = particle.t + fmix * params.eta * particle.delta *  2 * CUDA_PI_F * iparams.timeratio  / (params.omega0) ; // omega0 * trev = 2 pi
		float voltdiff = VoltageTripleRf(phi, voltages, harmonicNumbers) - 
		VoltageTripleRf(params.phiSynchronous - bucket * 2 * CUDA_PI_F, voltages, harmonicNumbers);

		// voltdiff = VoltageTripleRf(phi, voltages, harmonicNumbers) + energylost;
		// to simulate phase offsets use below
		// result.delta = particle.delta + fmix * voltdiff * params.charge / (2 * CUDA_PI_F * params.betar * params.p0);

		// ideal "forced" case voltdiff == energylost per turn
		result.delta = particle.delta + fmix *  voltdiff * params.charge  / ( params.betar * params.p0) * iparams.timeratio ; //2 * CUDA_PI_F *
		return result;

	}
	private:
		longitudinalParameters params;
		hamiltonianParameters hparams;
		ibsparameters iparams;
		float fmix;
		int bucket;
		float energylost;
};

/* ************************************************************ */
/* 																*/
/*  					constructor          					*/
/* 																*/
/* ************************************************************ */

STE_Bunch::STE_Bunch( bunchparameters& params ) {
	float h0 = params.hamparams.harmonicNumbers.x;
	float h1 = params.hamparams.harmonicNumbers.y;
	float h2 = params.hamparams.harmonicNumbers.z;

	float omega0 = params.synparams.omega0;
	int n = params.numberMacroParticles;


	thrust::device_vector<float6> tempdistribution( n );

	bucketNumber = params.bucket;

	// create TFS object to access twiss data
	tfs = new STE_TFS( params.tfsfile , false );
	// coding comment ptrs need to be dereferenced
	// (*ptr). == ptr->

	// create Radiation object to access synchrotron radiation functions and data
	rad = new STE_Radiation( params.radparams ,
		(*tfs).LoadTFSTableToDevice(tfs->getTFSTableData()) ,
		params.methodRadiationIntegrals
		);

	rad->printLatticeIntegrals();
	// create hamiltonina object to perfom longitudinal hamiltonian calculations
	ham = new STE_Longitudinal_Hamiltonian(params.hamparams);

	// innitialize the average radiation losses per turn
	energyLostPerTurn = rad->getAverageRadiationPower();
	params.synparams.energyLostPerTurn = energyLostPerTurn;

	// create Synchrotron object for longitudinal dynamics calculations
	// init synchronous phases , longitudiinal acceptance, and input paramaters
	syn                              = new STE_Synchrotron( params.synparams );
	synchronousPhase                 = syn->getSynchronousPhase(); 
	synchronousPhaseNext             = syn->getSynchronousPhaseNext();
	params.hamparams.phis            = synchronousPhase;
	params.ibsparams.phiSynchronous  = synchronousPhase;
	params.hamparams.t               = synchronousPhaseNext / (min(min(h0,h1),h2) * omega0);
	hamMax                           = ham->HamiltonianTripleRf( params.hamparams );
	std::cout << "hamMax " << hamMax << std::endl;
	std::cout << "phis " << params.hamparams.phis<< std::endl;
	tauhat                           = syn->getTauhat() ;
	params.longparams.tauhat         = tauhat;
	params.ibsparams.tauhat          = tauhat;
    synchrotronTune                  = syn->getSynchrotronTune();
	params.radparams.omegas          = synchrotronTune * omega0;
	params.longparams.sige           = synchrotronTune * omega0 * params.longparams.sigs / (CUDA_C_F * params.synparams.eta);
	params.longparams.phiSynchronous = synchronousPhase;
	params.longparams.hammax         = hamMax;

	// calculate the current
	current = params.realNumberOfParticles*  params.particleCharge * CUDA_ELEMENTARY_CHARGE / params.trev;
	
	/* debug */
	// std::cout << "bucketNumber                     :" << bucketNumber << std::endl;
	// std::cout << "energyLostPerTurn                :" << energyLostPerTurn << std::endl;
	// std::cout << "synchronousPhase                 :" << synchronousPhase << std::endl;
	// std::cout << "synchronousPhaseNext             :" << synchronousPhaseNext << std::endl;
	// std::cout << "params.hamparams.phis            :" << bparams.hamparams.phis << std::endl;
	// std::cout << "params.hamparams.t               :" << bparams.hamparams.t << std::endl;
	// std::cout << "hamMax                           :" << hamMax << std::endl;
	// std::cout << "tauhat                           :" << tauhat << std::endl;
	// std::cout << "params.longparams.tauhat         :" << bparams.hamparams.t << std::endl;
	// std::cout << "synchrotronTune                  :" << synchrotronTune << std::endl;
	// std::cout << "params.longparams.sige           :" << bparams.longparams.sige << std::endl;
	// std::cout << "params.longparams.phiSynchronous :" << bparams.longparams.phiSynchronous << std::endl;
	// std::cout << "params.longparams.hammax         :" << bparams.longparams.hammax << std::endl;
	// print to check values
	// std::cout << "seed          : " << bparams.longparams.seed << std::endl;

	// generate particle distribution
	switch( params.methodLongitudinalDist ){
		case 1:
		{
			thrust::transform(thrust::counting_iterator<int>(0),thrust::counting_iterator<int>( n ),
				tempdistribution.begin(),rand_6d_gauss<float6>( params.longparams ));
			break;
		}
		case 2:
		{
			thrust::transform(thrust::counting_iterator<int>(0),thrust::counting_iterator<int>( n ),
				tempdistribution.begin(),rand6DTransverseBiGaussLongitudinalMatched<float6>( params.longparams ));
		}

	}
	distribution = tempdistribution;

	// calculate emittance and update input parameters
	setEmittance( params );
	

	// store initial emittance for simulation output
	mapEmittances.insert(std::make_pair<int, float6>( 0, emittance) );

	// update radiation damping parameters
	rad->printLatticeIntegrals();

	rad = new STE_Radiation( params.radparams ,
		(*tfs).LoadTFSTableToDevice(tfs->getTFSTableData()) ,
		params.methodRadiationIntegrals
		);

	rad->printLatticeIntegrals();

	rad->setDampingAndEquilib( params.radparams, bparams.methodRadiationIntegrals );

	

	// create intrabeam scattering object
	ibs = new STE_IBS( params.ibsparams, distribution , tfs->LoadTFSTableToDevice(tfs->getTFSTableData()));
	bparams = params;

	// init IBSLifeTimes
	float3 growthRates = ibs->getIBSGrowthRates();
	updateIBSLifeTimes( 0 , growthRates );
	
	std::cout << "Init Growth rates " << growthRates.x << " " << growthRates.y << " " << growthRates.z << " " << std::endl ;
	rad->printDampingTimes();

	// add initial values to mapIntensity
	setIntensity( bparams , 0 );

	// init debunchlosses
	debunchlosses = 0;
 }

/* ************************************************************ */
/* 																*/
/*  					helper functions      					*/
/* 																*/
/* ************************************************************ */



#define MAX_DATE 12

std::string STE_Bunch::get_date(){

	/* function returning current date as string for creating timestamped output files */

   	time_t now;

   	// struct tm * timeinfo;
   	char the_date[MAX_DATE];
   	// char buffer [80];
	
   	// time_t rawtime;
   	// timeinfo = localtime (&rawtime);
   	// strftime (buffer,80,"Now it's %I:%M%p.",timeinfo);
	
   	// std::stringstream ss;
   	// ss << fn << "_" << "%d_%m_%Y" << "_" << buffer << ".dat";
   	// std::string s = ss.str();
	
   	the_date[0] = '\0';
   	
   	now = time(NULL);
	
   	if (now != -1)
   	{
   	   strftime(the_date, MAX_DATE, "%d_%m_%Y", gmtime(&now));
   	}
	
   	return std::string(the_date);
 }


float6 STE_Bunch::calculateEmittance(thrust::device_vector<float6> distributionin, bunchparameters& params ){	

	/* Function for calculating emittance assuming Gaussian shape of the bunch */
	/* TODO : update with functions using Gaussian with cut tails              */

	float betax, betay;

	betax = params.radparams.betxRingAverage;
	betay = params.radparams.betyRingAverage;

	/* debug */
    // std::cout << betax << std::endl;
   
	float6 sum;
	sum.x= 0.0;
	sum.px=0.0;
	sum.y=0.0;
	sum.py=0.0;
	sum.t=0.0;
	sum.delta=0.0;
	float6 average = thrust::reduce(distributionin.begin(),
		distributionin.end(),
		sum,
		addFloat6()
		);

	/* debug */
	// std::cout << average.x << std::endl;

	int n = params.numberMacroParticles;

	average.x     = - average.x / n;
	average.px    = - average.px / n;
	average.y     = - average.y / n;
	average.py    = - average.py / n;
	average.t     = - average.t / n;
	average.delta = - average.delta / n;

	thrust::transform(distributionin.begin(), 
		distributionin.end(),
		thrust::make_constant_iterator(average),
		distributionin.begin(),
		addFloat6()
		);	


	thrust::transform(distributionin.begin(),
		distributionin.end(),
		distributionin.begin(),
		squareFunctor<float6> ()
		);

	float6 outsum = thrust::reduce(distributionin.begin(),
		distributionin.end(),
		sum,
		addFloat6()
		);
	
	outsum.x = outsum.x / (n * betax);
	outsum.y = outsum.y / (n * betay);
	outsum.px = sqrt(outsum.px/n);
	outsum.py = sqrt(outsum.py/n);
	outsum.t = sqrt(outsum.t/n);
	outsum.delta = sqrt(outsum.delta/n);

    
    return outsum;
 
 }

/* ************************************************************ */
/* 																*/
/*  					set functions      		     			*/
/* 																*/
/* ************************************************************ */

void STE_Bunch::setEmittance( bunchparameters& params){

	/* Function for determiniing and setting the class stored current emittance */

	thrust::device_vector<float6> distributionin = distribution;
	emittance = calculateEmittance( distributionin, params );

	// update input parameters
	bparams.longparams.emitx          = emittance.x;
	bparams.longparams.emity          = emittance.y;
	bparams.longparams.sigs           = emittance.t;
	bparams.ibsparams.emitx           = emittance.x;
	bparams.ibsparams.emity           = emittance.y;
	bparams.ibsparams.dponp           = emittance.delta;

 }


void STE_Bunch::resetDebunchLosses(){
	debunchlosses = 0;
 }

void STE_Bunch::setIntensity( bunchparameters& params , int turn ){

	/* Function for setting mapIntensity */
	float6 intensity;
	
	// real time
	intensity.x = turn * params.trev * params.timeratio;

	// nMacro
	intensity.px = params.numberMacroParticles;

	// nReal
	intensity.y = params.realNumberOfParticles;

	intensity.py = debunchlosses * params.conversion;

	// Real Debunch Losses
	if(!mapIntensity.empty())
    	intensity.t = ((--mapIntensity.end())->second).t + debunchlosses;
    else
    	intensity.t =  debunchlosses;

    // cumul debunch losses
    intensity.t = intensity.t * params.conversion;

    // update current
    current =  params.realNumberOfParticles * params.particleCharge * CUDA_ELEMENTARY_CHARGE / params.trev;
    intensity.delta = current;

    // add to map
	mapIntensity.insert(std::make_pair<int, float6>( turn, intensity ) );

	// reset debunchlosses
	resetDebunchLosses();

 }


/* ************************************************************ */
/* 																*/
/*  					get functions      		     			*/
/* 																*/
/* ************************************************************ */

bunchparameters STE_Bunch::getparams(){

	/* Function to get the class stored input parameters - note that these are updated by some functions */
	/* NOT the best coding style but otherwise the init of the main program is very dirty with lots of   */
	/* function calls making it hard to get the conceptual program flow - desing choice to improve code  */
	/* readability. */

	return bparams;

 }

int STE_Bunch::getBucketNumber() {
	
	return bucketNumber;

 }

float6 STE_Bunch::getEmittance(){
	return emittance;

 }
		  
std::map<int, float3> STE_Bunch::getIBSLifeTimes(){

	/* get IBS lifetimes as map */

	return  IBSLifeTimes;
 }

/* ************************************************************ */
/* 																*/
/*  					physcis routines    					*/
/* 																*/
/* ************************************************************ */

void STE_Bunch::RadiationRoutine( bunchparameters& params ){

	// Synchrotron radiaion routine 
	// method 1 : smooth ring approximation using averager beta functions etc...
	// method 2 : calculated element by element using twiss tables and summed over all elements

	float6 timesequilib = rad->getDampingAndEquilib();

	rad->printDampingTimes();
	/* debug */
	// std::cout << "equilib " << timesequilib << std::endl;

	thrust::transform( 
		distribution.begin() , 
		distribution.end() , 
		distribution.begin() ,
		RadiationDampingRoutineFunctor( timesequilib , params.trev , params.timeratio , params.longparams.seed )
		);

 };

void STE_Bunch::BetaRoutine ( bunchparameters& params ){

	// simulated betatron motion - average over entire ring

	thrust::transform(
		distribution.begin() , 
		distribution.end() , 
		distribution.begin() ,
		BetatronRoutine( 
			params.tunex , 
			params.tuney , 
			params.ksix ,
			params.ksiy ,
			params.betatroncoupling , 
			params.k2l ,
			params.k2sl
			)
		);

 };

void STE_Bunch::IBSRoutine( bunchparameters& params ){

	// simulation of intra beam scattering
	// method 1 : Piwinski Smooth - using Piwiniski's approximation together with smooth ring approx (avg beta, etc.)
	// method 2 : Piwinski Lattice - Piwinski's approximation but lifetimes calculated element by element and summed over ring
	// method 3 : Piwinski Modified - Same as 2 but taking some vertical dispersion into account
	// method 4 : Nagaitsev approx - using the approximation from Nagaitsev using Carlssons integrals -> fast due to recursion relations

	thrust::device_vector<float> tmp = ibs->getSqrtHistogram();
	float* denlon2k = thrust::raw_pointer_cast( tmp.data() );

	/* debug */
	// std::vector<float> denlon2k = ibs->getSqrtHistogram();

	thrust::transform( distribution.begin() ,
		distribution.end() ,
		distribution.begin() ,
		ibsRoutine( ibs->getIBScoeff(), 
			denlon2k , 
			params.ibsparams
			)
		);
 }

void STE_Bunch::RFRoutine( bunchparameters& params , float fmix ){

	/* Function to update particles position and momenta due to synchrotron motion */

	// update the distribution
	thrust::transform(distribution.begin(),
		distribution.end(),
		distribution.begin(),
		RfUpdateRoutineFunctor( fmix , params.longparams , params.hamparams, params.ibsparams, bucketNumber, energyLostPerTurn )
		);

	int nMacro = distribution.size();

	// std::cout << "distribution size" << nMacro << std::endl;

	float harmonic ;
	// std::cout << "harmonic min " << min(min(params.longparams.h0, params.longparams.h1),params.longparams.h2) << std::endl;
	if ( min(min(params.longparams.h0, params.longparams.h1),params.longparams.h2) == 1 )
		harmonic = params.longparams.h0;
	else
		harmonic = min(min(params.longparams.h0,params.longparams.h1),params.longparams.h2);

	// std::cout << "harmonic " << harmonic << std::endl;
	// remove particles outside of phase-space acceptance
	thrust::device_vector<float6>::iterator new_end = thrust::remove_if(
		distribution.begin(),
		distribution.end(),
		isInLong( params.longparams.tauhat , params.longparams.phiSynchronous  / (harmonic * params.longparams.omega0)
		)
		);

	// registering the debunching losses - cumul until written to file 
	debunchlosses += nMacro - (new_end - distribution.begin());

	/* debug */
	std::cout << "Particles lost due to debunching : " << debunchlosses << std::endl;

	// resizing the vector
	distribution.resize(new_end - distribution.begin());

	std::cout << "distribution size after" << distribution.size() << std::endl;

	// update bparams
	bparams.numberMacroParticles  = (new_end - distribution.begin());
	bparams.realNumberOfParticles = bparams.numberMacroParticles * bparams.conversion;
	bparams.ibsparams.numberRealParticles  = bparams.realNumberOfParticles;
	bparams.ibsparams.numberMacroParticles = bparams.numberMacroParticles;

	// std::cout << bparams.numberMacroParticles << std::endl;
	// std::cout << bparams.realNumberOfParticles << std::endl;
	// std::cout << tauhat << std::endl;
	// std::cout << "phis " << params.hamparams.phis<< std::endl;

}

/* ************************************************************ */
/* 																*/
/*  					updating functions  					*/
/* 																*/
/* ************************************************************ */

// helper functions
// ****************
void STE_Bunch::updateIBSLifeTimes( int turn , float3 ibsGrowthRates ){
	IBSLifeTimes.insert(std::make_pair<int, float3>(turn, 
		ibs->getIBSLifeTimes( ibsGrowthRates ))
	);

 };


// main functions
// **************
void STE_Bunch::updateEmittance( int turn ){
	setEmittance( bparams );
 };


void STE_Bunch::updateRadiation(){
	RadiationRoutine( bparams );
 };

void STE_Bunch::updateBetaTron(){
	BetaRoutine( bparams );
 };

void STE_Bunch::updateIBS( int turn ){

	ibs->update(bparams.ibsparams, 
		distribution, 
		tfs->LoadTFSTableToDevice( tfs->getTFSTableData() )
		);

	float3 ibsGrowthRates = ibs->getIBSGrowthRates();
	// std::cout << "Growth Rates :" << ibsGrowthRates.x << " " << ibsGrowthRates.y << " " << ibsGrowthRates.z << std::endl;
	updateIBSLifeTimes( turn , ibsGrowthRates );
	
 };

void STE_Bunch::updateRF( int turn , int mix ){

	/* function to update rf passage - synchrotron motion */
	// mix is a factor for enhancing or decreasing the RF effect on momentum spread

	// apply RF
	RFRoutine( bparams , mix );

 }

void STE_Bunch::updateBunch( int turn , int RadDamping , int BetaTron , int IBS , int RF , int RFmix, int write ){

	if ( RadDamping == 1 )
		updateRadiation( );

	if ( BetaTron == 1 )
		updateBetaTron(); 

	if ( IBS == 1 ) 
		updateIBS( turn );
		IBSRoutine( bparams );

	if ( RF == 1 )
		updateRF( turn , RFmix );

	// update emittance
	updateEmittance( turn );

	if (turn % write == 0)
		// write emittance to map
		mapEmittances.insert(std::make_pair<int, float6>(turn, emittance) );
		// update intensity and write to map
		setIntensity( bparams ,  turn );

 };


/* ************************************************************ */
/* 																*/
/*  					printing to screen functions     		*/
/* 																*/
/* ************************************************************ */

__host__ std::ostream& operator<< (std::ostream& os, const float6& p){

	// << operator overload to write float6 to screen

	os << std::setw(15) << p.x << std::setw(15) << p.px << std::setw(15) << p.y 
	<< std::setw(15) << p.py << std::setw(15) << p.t <<std::setw(15) << p.delta << std::endl;;
	return os;
 };

__host__ std::ostream& operator<< (std::ostream& os, const float3& p){

	// << operator overload to write float3 to screen

	os << std::setw(15) << p.x << std::setw(15) << p.y << std::setw(15) << p.z << std::endl;
	return os;
 };

void STE_Bunch::printBunchParams(){

	/* Debug function to print out input parameters for checking if they are set correctly */

	// std::cout.precision(19);
	std::cout << "Bucket                  : " << bucketNumber                         << std::endl;
	std::cout << "Energy lost per turn    : " << energyLostPerTurn                    << std::endl;
	std::cout << "tunex                   : " << bparams.tunex                         << std::endl;
	std::cout << "tuney                   : " << bparams.tuney                         << std::endl;
	std::cout << "trev                    : " << bparams.trev                          << std::endl;
	std::cout << "timeratio               : " << bparams.timeratio                     << std::endl;
	std::cout << "particleType            : " << bparams.particleType                  << std::endl;
	std::cout << "particleAtomNumber      : " << bparams.particleAtomNumber            << std::endl;
	std::cout << "particleCharge          : " << bparams.particleCharge                << std::endl;
	std::cout << "methodRadiationIntegrals: " << bparams.methodRadiationIntegrals      << std::endl;
	std::cout << "tfsfile                 : " << bparams.tfsfile                       << std::endl;
	std::cout << "numberMacroParticles    : " << bparams.numberMacroParticles          << std::endl;
	std::cout << "methodLongitudinalDist  : " << bparams.methodLongitudinalDist        << std::endl;
	std::cout << "acceleratorLength       : " << bparams.radparams.acceleratorLength   << std::endl;
	std::cout << "betxRingAverage         : " << bparams.radparams.betxRingAverage     << std::endl;
	std::cout << "betyRingAverage         : " << bparams.radparams.betyRingAverage     << std::endl;
	std::cout << "gammaTransition         : " << bparams.radparams.gammaTransition     << std::endl;
	std::cout << "DipoleBendingRadius     : "  << bparams.radparams.DipoleBendingRadius << std::endl;
	std::cout << "ParticleRadius          : "  << bparams.radparams.ParticleRadius      << std::endl;
	std::cout << "ParticleEnergy          : "  << bparams.radparams.ParticleEnergy      << std::endl;
	std::cout << "cq                      : "  << bparams.radparams.cq                  << std::endl;
	std::cout << "gammar                  : "  << bparams.radparams.gammar              << std::endl;
	std::cout << "eta                     : "  << bparams.radparams.eta                 << std::endl;
	std::cout << "omegas                  : "  << bparams.radparams.omegas              << std::endl;
	std::cout << "p0                      : "  << bparams.radparams.p0                  << std::endl;
	std::cout << "ParticleRestEnergy      : "  << bparams.radparams.ParticleRestEnergy  << std::endl;
	std::cout << "trev                    : "  << bparams.radparams.trev                << std::endl;
	std::cout << "eta                     : "  << bparams.hamparams.eta                 << std::endl;
	std::cout << "angularFrequency        : "  << bparams.hamparams.angularFrequency    << std::endl;
	std::cout << "p0                      : "  << bparams.hamparams.p0                  << std::endl;
	std::cout << "betar                   : "  << bparams.hamparams.betar               << std::endl;
	std::cout << "particleCharge          : "  << bparams.hamparams.particleCharge      << std::endl;
	std::cout << "delta                   : "  << bparams.hamparams.delta               << std::endl;
	std::cout << "t                       : "  << bparams.hamparams.t                   << std::endl;
	// std::cout << "voltages                : "  << bparams.hamparams.voltages            << std::endl;
	// std::cout << "harmonicNumbers         : "  << bparams.hamparams.harmonicNumbers     << std::endl;
	std::cout << "phis                    : "  << bparams.hamparams.phis                << std::endl;
	std::cout << "particleCharge          : "  << bparams.synparams.particleCharge      << std::endl;
	std::cout << "search                  : "  << bparams.synparams.search              << std::endl;
	std::cout << "searchWidth             : "  << bparams.synparams.searchWidth         << std::endl;
	// std::cout << "harmonicNumbers         : "  << bparams.synparams.harmonicNumbers     << std::endl;
	// std::cout << "voltages                : "  << bparams.synparams.voltages            << std::endl;
	std::cout << "omega0                  : "  << bparams.synparams.omega0              << std::endl;
	std::cout << "eta                     : "  << bparams.synparams.eta                 << std::endl;
	std::cout << "p0                      : "  << bparams.synparams.p0                  << std::endl;
	std::cout << "seed                    : "  << bparams.longparams.seed               << std::endl;
	std::cout << "betx                    : "  << bparams.longparams.betx 			   << std::endl;
	std::cout << "bety                    : "  << bparams.longparams.bety 			  << std::endl;
	std::cout << "emitx                   : "  << bparams.longparams.emitx			  << std::endl;
	std::cout << "emity                   : "  << bparams.longparams.emity			  << std::endl;
	std::cout << "sigs                    : "  << bparams.longparams.sigs  			  << std::endl;
	std::cout << "omega0                  : "  << bparams.longparams.omega0			  << std::endl;
	std::cout << "v0                      : "  << bparams.longparams.v0    			  << std::endl;
	std::cout << "v1                      : "  << bparams.longparams.v1    			  << std::endl;
	std::cout << "v2                      : "  << bparams.longparams.v2    			  << std::endl;
	std::cout << "h0                      : "  << bparams.longparams.h0    			  << std::endl;
	std::cout << "h1                      : "  << bparams.longparams.h1    			  << std::endl;
	std::cout << "h2                      : "  << bparams.longparams.h2    			  << std::endl;
	std::cout << "p0                      : "  << bparams.longparams.p0    			  << std::endl;
	std::cout << "betar                   : "  << bparams.longparams.betar 			  << std::endl;
	std::cout << "eta                     : "  << bparams.longparams.eta   			  << std::endl;
	std::cout << "charge                  : "  << bparams.longparams.charge			  << std::endl;
 }	

void STE_Bunch::printDistribution(){

	/* Print current distribution to screen */

	std::copy(distribution.begin(),distribution.end(),std::ostream_iterator<float6>(std::cout));
 }

void STE_Bunch::printEmittance( int turn ){

	/* Print current emittances to screen */
	std::cout << std::setw(6) << "Turn" << std::setw(15) << "ex" 
			<< std::setw(15) << "sigpx" 
			<< std::setw(15) << "ey" 
			<< std::setw(15) << "sigpy" 
			<< std::setw(15) << "sigs" 
			<< std::setw(15) << "sige" 
			<< std::endl;
	std::cout << std::setw(6) << turn << emittance << std::endl;
 }

void STE_Bunch::printHistogramTime(){

	/* Print current longitudinal histogram to screen */
	
	thrust::device_vector<int> hist;
	hist = ibs->getTimeHistogram();
	std::copy( hist.begin() ,
		hist.end() , 
		std::ostream_iterator<int>( std::cout, "\n" )
		);

 }

void STE_Bunch::printSqrtHistogram(){

	/* Print current coefficient histogram to screen */
	thrust::device_vector<float> hist;
	hist = ibs->getSqrtHistogram();
	std::copy(hist.begin() , 
		hist.end() , 
		std::ostream_iterator<float>( std::cout, "\n") 
		);

 }


/* ************************************************************ */
/* 																*/
/*  					printing to file functions     		    */
/* 																*/
/* ************************************************************ */

void STE_Bunch::writeDistributionToFile( int turn ){

	/* writing current distribution to file */
	/* WARNING : can produce large files    */

	std::stringstream ss;
	ss << "Distribution_bucket_" << bucketNumber << "_" << get_date() << "_turn_" << 
	std::setw(10) << std::setfill('0') << turn << ".dat";

	std::string s;
	s = ss.str();

	std::ofstream ofile(s.c_str());

	thrust::copy(distribution.begin() ,
		distribution.end() , 
		std::ostream_iterator<float6>( ofile )
		);

	std::cout << "writing vector size = " << distribution.size() << std::endl;
	ofile.close();

 }

void STE_Bunch::writeEmittanceToFile(){

 	std::stringstream ss;
 	ss << "STE_Emittances_" << bucketNumber << "_" << get_date() << ".dat";

 	std::string s;
	s = ss.str();

	std::ofstream ofile(s.c_str());

	// write headers
	ofile << std::setw(6) << "Turn" << std::setw(15) << "ex" 
			<< std::setw(15) << "sigpx" 
			<< std::setw(15) << "ey" 
			<< std::setw(15) << "sigpy" 
			<< std::setw(15) << "sigs" 
			<< std::setw(15) << "sige" 
			<< std::endl;

 	// printing the int maps
	for(std::map<int, float6>::iterator me = mapEmittances.begin(); me != mapEmittances.end(); me++)
	{
		ofile << std::setw(6) << me->first <<  me->second ;
	}
	ofile.close();
 }

void STE_Bunch::writeIBSLifeTimes(){

 	std::stringstream ss;
 	ss << "STE_IBS_LifeTimes_" << bucketNumber << "_" << get_date() << ".dat";

 	std::string s;
	s = ss.str();

	std::ofstream ofile(s.c_str());

	// write headers
	ofile << std::setw(6) << "Turn" << std::setw(15) << "Tex" 
			<< std::setw(15) << "Tey" 
			<< std::setw(15) << "Tp" 
			<< std::endl;

 	// printing the int maps
	for(std::map<int, float3>::iterator mi = IBSLifeTimes.begin(); mi != IBSLifeTimes.end(); mi++)
	{
		ofile << std::setw(6) << mi->first <<  mi->second ;
	}
	ofile.close();
 }