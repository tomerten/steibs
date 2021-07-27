
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

// load double 6 datastructure
// #include "STE_DataStructures.cuh"

// load tfsTableData datastructure
// #include "STE_TFS.cuh"

// load ibs
// #include "STE_IBS.cuh"

// load bunch header
#include "STE_Bunch_double.cuh"


#include "STE_Radiation_double.cuh"
// #include "STE_Longitudinal_Hamiltonian.cuh"
// #include "STE_Synchrotron.cuh"
// #include "STE_Random.cuh"

/* ************************************************************ */
/* 																*/
/*  					Helper functions       					*/
/* 																*/
/* ************************************************************ */
__host__ std::ostream& operator<< (std::ostream& os, const double3& p){

	// << operator overload to write double3 to screen

	os << std::setw(15) << p.x << std::setw(15) << p.y << std::setw(15) << p.z << std::endl;
	return os;
 };

__host__ __device__ double VoltageTripleRf(double phi, double3 voltages, double3 harmonicNumbers){
	double volt1, volt2, volt3;
	volt1 =  voltages.x * sin(phi);
	volt2 =  voltages.y * sin((harmonicNumbers.y / harmonicNumbers.x) * phi);
	volt3 =  voltages.z * sin((harmonicNumbers.z / harmonicNumbers.x) * phi);
	return volt1 + volt2 + volt3;
 }; 

__host__ __device__ double tcoeff( hamiltonianParameters& params ){
	return  (params.angularFrequency * params.eta * params.harmonicNumbers.x);
 }

__host__ __device__ double pcoeff( hamiltonianParameters& params, double voltage ){
	return  (params.angularFrequency * voltage * params.particleCharge) / (2 * CUDA_PI_F * params.p0 * params.betar);
	};

__host__ std::ostream& operator<< (std::ostream& os, const double4& p){

	// << operator overload to write double3 to screen

	os << std::setw(15) << p.x << std::setw(15) << p.y << std::setw(15) << p.z << std::setw(15) << p.w << std::endl;
	return os;
 };

double6 CalcRadDecayExitation(double6 radiationDampParams, double trev, double timeratio , bunchparameters& bp){

	double6 result;

	// timeratio is real machine turns over per simulation turn 
	double coeffdecaylong  = exp( - trev * timeratio / (radiationDampParams.y)) ;

	// excitation uses a uniform deviate on [-1:1]
	// sqrt(3) * sigma => +/-3 sigma**2
	// see also lecture 2 Wolski on linear dynamics and radiation damping
	double coeffexcitelong = sqrt(radiationDampParams.delta * CUDA_C_F) * sqrt(3.) * sqrt((2 * trev / ( radiationDampParams.y )) * timeratio);

	// the damping time is for EMITTANCE, therefore need to multiply by 2
    double coeffdecayx     = exp( - (  trev * timeratio / ( 2 * radiationDampParams.x  ) ) );
    double coeffdecayy     = exp( - (  trev * timeratio / ( 2 * radiationDampParams.px ) ) );
    
    // exact     coeffgrow= sigperp*sqrt(3.)*sqrt(1-coeffdecay**2)
    // squared because sigma and not emit
    double coeffgrowx       = sqrt(radiationDampParams.py * bp.radparams.betxRingAverage ) * sqrt(3.) * sqrt(1 - pow(coeffdecayx,2));
    double coeffgrowy       = sqrt(radiationDampParams.t  * bp.radparams.betyRingAverage ) * sqrt(3.) * sqrt(1 - pow(coeffdecayy,2));

	// decay coefficients
	result.x  = coeffdecayx;
	result.px = coeffdecayy;
	result.y  = coeffdecaylong;

	// excitation coefficients
	result.py    = coeffgrowx;
	result.t     = coeffgrowy;
	result.delta = coeffexcitelong; //CUDA_C

	// std::cout << "cd " << radiationDampParams.py  <<std::endl;
	std::cout << "cd " << result.x  << std::endl;
	std::cout << "cd " << result.px << std::endl;
	std::cout << "cd " << result.y  << std::endl;
	std::cout << "cd " << result.py    << std::endl;
	std::cout << "cd " << result.t     << std::endl;
	std::cout << "cd " << result.delta << std::endl;
	return result;

 };

void sampleLongMatched(thrust::host_vector<double6>& dist, int n, int seed, randomNumber& rng, bunchparameters& params ){

	
	double kinetic, potential1, potential2, potential3, prob, test, ham;

	double tc = tcoeff( params.hamparams );

	double ptmax = sqrt(2 * params.longparams.hammax / tc);
	longitudinalParameters in = params.longparams;

	double pc = (in.omega0 * in.charge) / (2 * CUDA_PI_F * in.p0 * in.betar);

	for(int i=0;i<n;i++){
		do
		{
			do 
			{
				dist[i].t  = params.longparams.tauhat  * (2 * rng(seed) - 1);
				dist[i].delta = ptmax * (2 * rng(seed) - 1);

				kinetic = 0.5 * tc * pow(dist[i].delta,2);

				potential1 = pc * in.v0 * (cos(in.h0 * in.omega0 * dist[i].t) - cos(in.phiSynchronous) + (in.h0 * in.omega0 * dist[i].t - in.phiSynchronous) * sin(in.phiSynchronous));
				potential2 = pc * in.v1 * (in.h0 / in.h1) * (cos(in.h1 * in.omega0 * dist[i].t) - cos(in.h1 * in.phiSynchronous / in.h0) + 
						(in.h1 * in.omega0 * dist[i].t - in.h1 * in.phiSynchronous / in.h0) * sin(in.h1 * in.phiSynchronous / in.h0));
				potential3 = pc * in.v2 * (in.h0 / in.h2) * (cos(in.h2 * in.omega0 * dist[i].t) - cos(in.h2 * in.phiSynchronous / in.h0) + 
					(in.h2 * in.omega0 * dist[i].t - in.h2 * in.phiSynchronous / in.h0) * sin(in.h2 * in.phiSynchronous / in.h0));
				
				ham = kinetic + potential1 + potential2 + potential3;
			}
			while ( ham > params.longparams.hammax );

			prob = exp( -ham / params.longparams.hammax );
			test = rng(seed);

		}
		while ( prob < test );
	}
  };

 // void longMatch(){

 // 		ham1sig0 = -nharm2*omega0*vrf2/(clight**2*v00)*rmsBunchLen**2
 // 	};


struct RfUpdateRoutineFunctor  {
	RfUpdateRoutineFunctor( double fmix, longitudinalParameters& params , hamiltonianParameters& hparams  , ibsparameters& iparams, int bucket , double energylost): 
	fmix(fmix), params(params),  hparams(hparams), iparams(iparams), bucket(bucket), energylost(energylost) {}

	__host__ __device__ double6 operator()(double6 particle)
	{
		double6 result;

		// using parameters to generate parameters for functions 
		double3 voltages        = make_double3(params.v0, params.v1, params.v2);
		double3 harmonicNumbers = make_double3(params.h0, params.h1, params.h2);
		double tcoeffValue      = tcoeff(hparams) / (params.h0 * params.omega0);

		// Lee Third edition page 233 eqs 3.6 3.16 3.13
		
		// the phase  = h0 omega0 t
		double voltdiff;
		double phi = params.h0 * params.omega0 * particle.t ;

		// Delta delta 
		// double pcoeffValue = pcoeff(hparams , VoltageTripleRf(phi, voltages, harmonicNumbers) - 
		// 	VoltageTripleRf(params.phiSynchronous - bucket * 2 * CUDA_PI_F, voltages, harmonicNumbers));

		result.x     = particle.x;
		result.px    = particle.px;
		result.y     = particle.y;
		result.py    = particle.py;


	
		phi       = ( phi + fmix * 2 * CUDA_PI_F * params.h0 * params.eta  * particle.delta) ;
		result.t  =  phi / (params.h0 * params.omega0);


		voltdiff = VoltageTripleRf( phi, voltages, harmonicNumbers ) - 
				   VoltageTripleRf( params.phiSynchronous - bucket * 2 * CUDA_PI_F, voltages, harmonicNumbers);

		result.delta = particle.delta  + fmix *  voltdiff * params.charge  / (  pow(params.betar,2) * params.p0 );

		


		
		

		// result.delta = particle.delta  + fmix *  voltdiff * params.charge  / (  pow(params.betar,2) * params.p0 );
		// result.t     = particle.t + fmix * params.eta * params.h0 * particle.delta * 2 * CUDA_PI_F / (params.h0 * params.omega0);
		

		// double voltdiff = VoltageTripleRf(phi, voltages, harmonicNumbers) - 
		// VoltageTripleRf(params.phiSynchronous - bucket * 2 * CUDA_PI_F, voltages, harmonicNumbers);
		// result.delta = particle.delta  + fmix *  voltdiff * params.charge  / ( params.betar * params.p0 * CUDA_C_F);
		// result.t     = particle.t + fmix * params.eta * result.delta * 2 * CUDA_PI_F  / (params.omega0) ;
		// voltdiff = VoltageTripleRf(phi, voltages, harmonicNumbers) + energylost;
		// to simulate phase offsets use below
		// result.delta = particle.delta + fmix * voltdiff * params.charge / (2 * CUDA_PI_F * params.betar * params.p0);

		// ideal "forced" case voltdiff == energylost per turn
		//2 * CUDA_PI_F *
		return result;

	}
	private:
		longitudinalParameters params;
		hamiltonianParameters hparams;
		ibsparameters iparams;
		double fmix;
		int bucket;
		double energylost;
 };

/* ************************************************************ */
/* 																*/
/*  					constructor          					*/
/* 																*/
/* ************************************************************ */

STE_Bunch::STE_Bunch( bunchparameters& params , int raddampon) {
	
	// init and store simulation paramters
	bparams      = params;
	bucketNumber = bparams.bucket;

	// helper variables
	double h0 = bparams.hamparams.harmonicNumbers.x;
	double h1 = bparams.hamparams.harmonicNumbers.y;
	double h2 = bparams.hamparams.harmonicNumbers.z;

	double omega0 = bparams.synparams.omega0;

	int n = bparams.numberMacroParticles;

	// setting minimum of harmonic numbers as this will determine the main bucket locations
	if ( min(min(h0, h1),h2) == 1 )
		minharmonic = h0;
	else
		minharmonic = min(min(h0,h1),h2);

	// create TFS object to access twiss data
	tfs = new STE_TFS( bparams.tfsfile , false );

	// *********************************************
	// coding comment ptrs need to be dereferenced
	// (*ptr). == ptr->
	// **********************************************

	// create Radiation object to access synchrotron radiation functions and data
	rad = new STE_Radiation( bparams.radparams ,
		(*tfs).LoadTFSTableToDevice(tfs->getTFSTableData()) ,
		bparams.methodRadiationIntegrals
		);

	
	// create hamiltonian object to perfom longitudinal hamiltonian calculations
	ham = new STE_Longitudinal_Hamiltonian( bparams.hamparams );

	// initialize the average radiation losses per turn
	energyLostPerTurn = rad->getAverageRadiationPower();
	bparams.synparams.energyLostPerTurn = energyLostPerTurn;

	// create Synchrotron object for longitudinal dynamics calculations
	// init synchronous phases , longitudiinal acceptance, and input paramaters
	syn                               = new STE_Synchrotron( bparams.synparams );

	synchronousPhase                  = syn->getSynchronousPhase(); 
	synchronousPhaseNext              = syn->getSynchronousPhaseNext();
	bparams.hamparams.phis            = synchronousPhase;
	bparams.ibsparams.phiSynchronous  = synchronousPhase;
	bparams.hamparams.t               = synchronousPhaseNext /  minharmonic ;//(min(min(h0,h1),h2) * omega0);
	hamMax                            = ham->HamiltonianTripleRf( bparams.hamparams );
	tauhat                            = syn->getTauhat() ;
	bparams.longparams.tauhat         = tauhat;
	bparams.ibsparams.tauhat          = tauhat;
    synchrotronTune                   = syn->getSynchrotronTune();
	bparams.radparams.omegas          = synchrotronTune * omega0;
	bparams.longparams.sige           = synchrotronTune * omega0 * bparams.longparams.sigs / (CUDA_C_F * bparams.synparams.eta);
	bparams.longparams.phiSynchronous = synchronousPhase;
	bparams.longparams.hammax         = hamMax;


	// calculate the current
	current = bparams.realNumberOfParticles*  bparams.particleCharge * CUDA_ELEMENTARY_CHARGE / bparams.trev;
	
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
	// // print to check values
	// std::cout << "seed          : " << bparams.longparams.seed << std::endl;

	// generate particle distribution
	thrust::device_vector<double6> tempdistribution( n );
	switch( bparams.methodLongitudinalDist ){
		case 1:
		{
			thrust::transform(thrust::counting_iterator<int>(0),thrust::counting_iterator<int>( n ),
				tempdistribution.begin(),rand_6d_gauss<double6>( bparams.longparams ));
			break;
		}
		case 2:
		{
			thrust::transform(thrust::counting_iterator<int>(0),thrust::counting_iterator<int>( n ),
				tempdistribution.begin(),rand6DTransverseBiGaussLongitudinalMatched<double6>( bparams.longparams ));
		}

	}
	distribution = tempdistribution;

	// calculate emittance and update input parameters
	setEmittance( bparams );
	
	// store initial emittance for simulation output
	mapEmittances.insert(std::make_pair<int, double6>( 0, emittance) );

	// update radiation damping parameters
	// rad->printLatticeIntegrals();

	rad = new STE_Radiation( bparams.radparams ,
		(*tfs).LoadTFSTableToDevice(tfs->getTFSTableData()) ,
		bparams.methodRadiationIntegrals
		);
	rad->setDampingAndEquilib( bparams.radparams, bparams.methodRadiationIntegrals );

	
	// create intrabeam scattering object
	ibs = new STE_IBS( bparams.ibsparams, distribution , tfs->LoadTFSTableToDevice(tfs->getTFSTableData()), 
		synchronousPhase /  (minharmonic * bparams.longparams.omega0)); // min(min(h0,h1),h2)

	// init IBSLifeTimes
	double3 growthRates = ibs->getIBSGrowthRates();
	updateIBSLifeTimes( 0 , growthRates );
	
	/* debug */
	std::cout << "Init Emit Growth rates " << growthRates.x << " " << growthRates.y << " " << growthRates.z << " " << std::endl ;
	std::cout << "Init Emit Growth times " << bparams.timeratio/growthRates.x << " " << bparams.timeratio/growthRates.y << " " << bparams.timeratio/growthRates.z << " " << std::endl ;
	// rad->printDampingTimes();

	for(std::map<int, double3>::iterator mi = IBSLifeTimes.begin(); mi != IBSLifeTimes.end(); mi++)
	{
		std::cout << std::setw(6) << mi->first <<  mi->second ;
	}

	// add initial values to mapIntensity
	setIntensity( bparams , 0 );

	// init debunchlosses
	debunchlosses = 0;

	// update equilibria and print to screen
	double6 timesequilib = rad->getDampingAndEquilib();
	rad->printDampingTimes();

	RadDecayExitationCoeff =  CalcRadDecayExitation( timesequilib , bparams.trev , bparams.timeratio , bparams );
	// std::cout << "synchrotronTune " << synchrotronTune << std::endl;
	std::cout << "conversion " << bparams.conversion << std::endl;
	lastTurn = 0;
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


double6 STE_Bunch::calculateEmittance(thrust::device_vector<double6> distributionin, bunchparameters& params ){	

	/* Function for calculating emittance assuming Gaussian shape of the bunch */
	/* TODO : update with functions using Gaussian with cut tails              */

	double betax, betay;

	betax = params.radparams.betxRingAverage;
	betay = params.radparams.betyRingAverage;

	/* debug */
    // std::cout << betax << std::endl;
   
	double6 sum;
	sum.x= 0.0;
	sum.px=0.0;
	sum.y=0.0;
	sum.py=0.0;
	sum.t=0.0;
	sum.delta=0.0;
	double6 average = thrust::reduce(distributionin.begin(),
		distributionin.end(),
		sum,
		adddouble6()
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
		adddouble6()
		);	


	thrust::transform(distributionin.begin(),
		distributionin.end(),
		distributionin.begin(),
		squareFunctor<double6> ()
		);

	double6 outsum = thrust::reduce(distributionin.begin(),
		distributionin.end(),
		sum,
		adddouble6()
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

	thrust::device_vector<double6> distributionin = distribution;
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

    double6 intens;
	intens.x      = turn;
	intens.px     = bparams.numberMacroParticles;
	intens.y      = CUDA_ELEMENTARY_CHARGE *(abs(bparams.particleCharge ) * bparams.numberMacroParticles * bparams.conversion) / ( bparams.trev );
	intens.py     = bparams.numberMacroParticles * bparams.conversion;
	intens.t      = debunchlosses * bparams.conversion;
	if (intens.t != 0)
	{
		// lastTurn keeps track of last debunchlosses so that delta t for Touschek can be calculated
		intens.delta  = ( ( intens.t + intens.py ) * bparams.trev * bparams.timeratio * (turn-lastTurn) ) / intens.t ;
		lastTurn = turn;
	}
	else
		intens.delta = 0.0;

	mapIntensity.insert(std::make_pair<int, double6>( turn, intens ) );

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

double6 STE_Bunch::getEmittance(){
	return emittance;

 }
		  
std::map<int, double3> STE_Bunch::getIBSLifeTimes(){

	/* get IBS lifetimes as map */

	return  IBSLifeTimes;
 }

/* ************************************************************ */
/* 																*/
/*  					physcis routines    					*/
/* 																*/
/* ************************************************************ */
void STE_Bunch::BlowUp(thrust::device_vector<double6>& dist, double6 multiplication,  int seed) {

	double6 E;
	E.x     = 0.0;
	E.px    = 0.0; //multiplication.px * sqrt(mapEmittances[0].x * bparams.radparams.betxRingAverage);
	E.y     = 0.0;
	E.py    = multiplication.py * sqrt(mapEmittances[0].y * bparams.radparams.betyRingAverage);
	E.t     = 0.0;
	E.delta = 0.0;

	// thrust::transform(
	// 	distribution.begin() , 
	// 	distribution.end() , 
	// 	distribution.begin() ,
	// 	blowupfunctor( E )
	// 	);

	int n = dist.size();
	thrust::device_vector<double6> excitation(n);
	thrust::device_vector<double6> R(n);

	thrust::fill( excitation.begin() , excitation.end() , E);

	thrust::transform(thrust::make_counting_iterator(0),
		thrust::make_counting_iterator(n),
        R.begin(),
        GenRand(seed)
        );
	// std::cout << "multi x " << multiplication.px << std::endl;
	// std::cout << "multi y " << multiplication.py << std::endl;
	thrust::transform( excitation.begin(), excitation.end(), R.begin(), excitation.begin() , multidouble6());
	// std::cout << "**exitation * rand ***" << std::endl;
	// std::copy(excitation.begin(),excitation.end(),std::ostream_iterator<double6>(std::cout));

	thrust::transform( dist.begin(), dist.end(), excitation.begin() , dist.begin() , adddouble6());
	// std::cout << "***dist after blowup **" << std::endl;
	// std::copy(dist.begin(),dist.end(),std::ostream_iterator<double6>(std::cout));
 }


void STE_Bunch::Radiation( thrust::device_vector<double6>& dist, double6 decayExitationCoeff,  int seed) {

	// typedef typename vector6::value_type value_type;

	double6 D;
	D.x     = decayExitationCoeff.x;
	D.px    = decayExitationCoeff.x;
	D.y     = decayExitationCoeff.px;
	D.py    = decayExitationCoeff.px;
	D.t     = 1.0;
	D.delta = decayExitationCoeff.y;

	double6 E;
	E.x     = decayExitationCoeff.py;
	E.px    = decayExitationCoeff.py;
	E.y     = decayExitationCoeff.t;
	E.py    = decayExitationCoeff.t;
	E.t     = 0.0;
	E.delta = decayExitationCoeff.delta;

	// std::cout << "cd " << std::endl;
	// std::cout << "decayx " << decayExitationCoeff.x  << std::endl;
	// std::cout << "decayy " << decayExitationCoeff.px << std::endl;
	// std::cout << "decaydelya " << decayExitationCoeff.y  << std::endl;
	// std::cout << "exx " << decayExitationCoeff.py    << std::endl;
	// std::cout << "exy " << decayExitationCoeff.t     << std::endl;
	// std::cout << "exdelta " << decayExitationCoeff.delta << std::endl;

	int n = dist.size();
	thrust::device_vector<double6> decay(n);
	thrust::device_vector<double6> excitation(n);
	thrust::device_vector<double6> R(n);


	thrust::fill( decay.begin()      , decay.end()      , D);
	thrust::fill( excitation.begin() , excitation.end() , E);




	thrust::transform(thrust::make_counting_iterator(0),
		thrust::make_counting_iterator(n),
        R.begin(),
        GenRand(seed)
        );

	// std::cout << "**before ***" << std::endl;
	// std::copy(dist.begin(),dist.end(),std::ostream_iterator<double6>(std::cout));
	// std::copy(decay.begin(),decay.end(),std::ostream_iterator<double6>(std::cout));
	// std::copy(dist.begin(),dist.end(),std::ostream_iterator<double6>(std::cout));

	thrust::transform( dist.begin(), dist.end(), decay.begin(), dist.begin() , multidouble6());
	// std::cout << "**decay * dist ***" << std::endl;
	// std::copy(dist.begin(),dist.end(),std::ostream_iterator<double6>(std::cout));
	
	thrust::transform( excitation.begin(), excitation.end(), R.begin(), excitation.begin() , multidouble6());
	// std::cout << "**exitation * rand ***" << std::endl;
	// std::copy(excitation.begin(),excitation.end(),std::ostream_iterator<double6>(std::cout));

	thrust::transform( dist.begin(), dist.end(), excitation.begin() , dist.begin(), adddouble6());
	// std::cout << "***dist after rad **" << std::endl;
	// std::copy(dist.begin(),dist.end(),std::ostream_iterator<double6>(std::cout));


 };

void STE_Bunch::RadiationRoutine( bunchparameters& params , int turn ){

	// Synchrotron radiaion routine 
	// method 1 : smooth ring approximation using averager beta functions etc...
	// method 2 : calculated element by element using twiss tables and summed over all elements

	Radiation( distribution , RadDecayExitationCoeff , bparams.longparams.seed + turn + bucketNumber);
	
	// old version
	// thrust::transform( 
	// 	distribution.begin() , 
	// 	distribution.end() , 
	// 	distribution.begin() ,
	// 	RadiationDampingRoutineFunctor( timesequilib , params.trev , params.timeratio , params.longparams.seed, radrnin )
	// 	);

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

void STE_Bunch::IBSRoutine( bunchparameters& params , int turn ){

	// simulation of intra beam scattering
	// method 0 : Piwinski Smooth - using Piwiniski's approximation together with smooth ring approx (avg beta, etc.)
	// method 1 : Piwinski Lattice - Piwinski's approximation but lifetimes calculated element by element and summed over ring
	// method 2 : Piwinski Modified - Same as 2 but taking some vertical dispersion into account
	// method 3 : Nagaitsev approx - using the approximation from Nagaitsev using Carlssons integrals -> fast due to recursion relations

	/* debug */
	// std::cout << "Bucket "<< bucketNumber << std::endl;

	// load histogram data with the multiplication factors
	thrust::device_vector<double> tmp = ibs->getSqrtHistogram();
	double* denlon2k = thrust::raw_pointer_cast( tmp.data() );
	// std::copy(tmp.begin(),tmp.end(),std::ostream_iterator<double>(std::cout));
	// std::cout <<  std::endl <<  std::endl;

	// create random vector
	int n = distribution.size();
	thrust::device_vector<double6> R(n);

	// create random vector - in order to not repeat random number add bucket number and turn to seed
	thrust::transform(thrust::make_counting_iterator(0),
		thrust::make_counting_iterator(n),
        R.begin(),
        GenRandIBS( bparams.longparams.seed + turn + bucketNumber * 100 )
        );

	// std::cout << " ************************* Random IBS vector *********************** " << std::endl;
	// std::copy(R.begin(),R.end(),std::ostream_iterator<double6>(std::cout));
	// std::cout <<  std::endl <<  std::endl;

	/* debug */
	// std::vector<double> denlon2k = ibs->getSqrtHistogram();

	// std::cout << "***before ibs **" << std::endl;
	// std::copy(distribution.begin(),distribution.end(),std::ostream_iterator<double6>(std::cout));
	// std::cout <<  std::endl <<  std::endl;


	double4 inputcoeff = ibs->getIBScoeff();
	// std::cout << "IBS coeff " << inputcoeff << std::endl;
	// std::cout <<  std::endl <<  std::endl;

	thrust::transform( distribution.begin() ,
		distribution.end() , R.begin(),
		distribution.begin() ,
		ibsRoutineNew( inputcoeff, 
			denlon2k , 
			params.ibsparams, 
			synchronousPhase / (bparams.longparams.h0 * bparams.longparams.omega0)
			)
		);

	// std::cout << "**after ibs ***" << std::endl;
	// std::copy(distribution.begin(),distribution.end(),std::ostream_iterator<double6>(std::cout));
	
 }

void STE_Bunch::RFRoutine( bunchparameters& params , double fmix ){

	/* Function to update particles position and momenta due to synchrotron motion */
 
	// update the distribution
	thrust::transform(distribution.begin(),
		distribution.end(),
		distribution.begin(),
		RfUpdateRoutineFunctor( fmix , params.longparams , params.hamparams, params.ibsparams, bucketNumber, energyLostPerTurn )
		);

	// std::cout << "** AFter rf update ***" << std::endl;
	// std::copy(distribution.begin(),distribution.end(),std::ostream_iterator<double6>(std::cout));

	// std::cout << "tauhat " << params.ibsparams.tauhat  << std::endl;
	
	int nMacro = distribution.size();

	// std::cout << "distribution size" << nMacro << std::endl;
	// std::cout << "synchronous time " <<  params.longparams.phiSynchronous  / (harmonic * params.longparams.omega0)<< std::endl;
	// std::cout << "harmonic " << harmonic << std::endl;


	// remove particles outside of phase-space acceptance
	thrust::device_vector<double6>::iterator new_end = thrust::remove_if(
		distribution.begin(),
		distribution.end(),
		isInLong( params.longparams.tauhat , params.longparams.phiSynchronous  / (minharmonic * params.longparams.omega0)
		)
		);

	// registering the debunching losses - cumul until written to file 
	debunchlosses += nMacro - (new_end - distribution.begin());

	/* debug */
	// std::cout << "Particles lost due to debunching : " << debunchlosses << std::endl;

	// resizing the vector
	distribution.resize(new_end - distribution.begin());

	// std::cout << "distribution size after " << distribution.size() << std::endl;

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
void STE_Bunch::updateIBSLifeTimes( int turn , double3 ibsGrowthRates ){

	// growth rate are divided by timeratio need to correct for that
	double3 growthratesRatioCompensated;
	growthratesRatioCompensated.x = ibsGrowthRates.x / bparams.timeratio;
	growthratesRatioCompensated.y = ibsGrowthRates.y / bparams.timeratio;
	growthratesRatioCompensated.z = ibsGrowthRates.z / bparams.timeratio;

	IBSLifeTimes.insert(std::make_pair<int, double3>(turn, 
		ibs->getIBSLifeTimes( growthratesRatioCompensated ))
	);

 };


// main functions
// **************
void STE_Bunch::updateEmittance( int turn ){
	setEmittance( bparams );
 };

void STE_Bunch::updateRadiation( int turn ){
	RadiationRoutine( bparams , turn );
 };

void STE_Bunch::updateBetaTron(){
	BetaRoutine( bparams );
 };

void STE_Bunch::updateIBS( int turn ){

	ibs->update(bparams.ibsparams, 
		distribution, 
		tfs->LoadTFSTableToDevice( tfs->getTFSTableData() ),
		synchronousPhase / (bparams.longparams.omega0 * minharmonic)
			);

	double3 ibsGrowthRates = ibs->getIBSGrowthRates();
	// std::cout << "Growth Rates :" << ibsGrowthRates.x << " " << ibsGrowthRates.y << " " << ibsGrowthRates.z << std::endl;
	// std::cout << "Growth Times :" << 1/ibsGrowthRates.x << " " << 1/ibsGrowthRates.y << " " << 1/ibsGrowthRates.z << std::endl;

	// std::cout << "syncphase :" << synchronousPhase  << std::endl;
	// std::cout << "synctime :" << synchronousPhase / (bparams.longparams.omega0 * minharmonic) << std::endl;
	
	// write ibs lifetimes to map
	updateIBSLifeTimes( turn , ibsGrowthRates );
	
 };

void STE_Bunch::updateRF( int turn , double mix ){

	/* function to update rf passage - synchrotron motion */
	// mix is a factor for enhancing or decreasing the RF effect on momentum spread

	// apply RF
	RFRoutine( bparams , mix );

 }

void STE_Bunch::updateBunch( int turn , int RadDamping , int BetaTron , int IBS , int RF , double RFmix, int write , 
	double blowupx, double blowupy){

	
	if ( RadDamping == 1 )
		{
			if (turn == 1)
				std::cout << "Radiation Damping ON" << std::endl;
			updateRadiation( turn );

		}
	else
	{
		// double h0 = bparams.hamparams.harmonicNumbers.x;
		// double h1 = bparams.hamparams.harmonicNumbers.y;
		// double h2 = bparams.hamparams.harmonicNumbers.z;

		double omega0 = bparams.synparams.omega0;
		
		// rad dampin off, no energy  lost per turn 
		energyLostPerTurn =0.0;
		bparams.synparams.energyLostPerTurn = energyLostPerTurn;

		syn->setSynchronousPhase( bparams.synparams );
		syn->setSynchronousPhaseNext( bparams.synparams );
		syn->setTauhat( bparams.synparams );
		syn->setSynchrotronTune( bparams.synparams );
		
		synchronousPhase                  = syn->getSynchronousPhase(); 
		synchronousPhaseNext              = syn->getSynchronousPhaseNext();
		bparams.hamparams.phis            = synchronousPhase;
		bparams.ibsparams.phiSynchronous  = synchronousPhase;
		bparams.hamparams.t               = synchronousPhaseNext / (minharmonic * omega0);
		hamMax                            = ham->HamiltonianTripleRf( bparams.hamparams );
		tauhat                            = syn->getTauhat() ;
		bparams.longparams.tauhat         = tauhat;
		bparams.ibsparams.tauhat          = tauhat;
	    synchrotronTune                   = syn->getSynchrotronTune();
		bparams.radparams.omegas          = synchrotronTune * omega0;
		bparams.longparams.sige           = synchrotronTune * omega0 * bparams.longparams.sigs / (CUDA_C_F * bparams.synparams.eta);
		bparams.longparams.phiSynchronous = synchronousPhase;
		bparams.longparams.hammax         = hamMax;
		// std::cout << "hamMax " << hamMax << std::endl;
		// std::cout << "phis " << bparams.hamparams.phis<< std::endl;
	}

	if ( BetaTron == 1 )
	{
		if (turn == 1)
			std::cout << "Betatron Motion ON" << std::endl;
		updateBetaTron(); 
	}

	if ( IBS == 1 ) 
	{
		if (turn == 1)
			std::cout << "IBS ON" << std::endl;
		updateIBS( turn );
		IBSRoutine( bparams , turn );
	}

	if ( RF == 1 )
	{
		if (turn == 1)
			std::cout << "RF ON" << std::endl;
		// std::cout << "RFmix " << RFmix << std::endl;
		updateRF( turn , RFmix );
	}

	if ((blowupx != 0.0) || (blowupy != 0.0))
	{
		double6 mult;
		mult.x = 1.0;
		mult.px = blowupx;
		mult.y = 1.0;
		mult.py = blowupy;
		mult.t = 1.0;
		mult.delta = 1.0;
		// std::cout << "blowup " << blowupy << std::endl;

		BlowUp( distribution , mult , bparams.longparams.seed + turn + bucketNumber );
		// std::cout << "***dist after blowup **" << std::endl;
		// std::cout << "bucket " << bucketNumber << std::endl;
		// std::cout << "Turn   " << turn << std::endl;
		// std::copy(distribution.begin(),distribution.end(),std::ostream_iterator<double6>(std::cout));
	}

	// std::cout << "***after blowup**" << std::endl;
	// update emittance
	updateEmittance( turn );

	// std::copy(distribution.begin(),distribution.end(),std::ostream_iterator<double6>(std::cout));

	if (turn % write == 0)
		{
		// write emittance to map
		mapEmittances.insert(std::make_pair<int, double6>(turn, emittance) );

		// update intensity and write to map debunch losses are also reset in this function
		setIntensity( bparams ,  turn );

	
	}

 };


/* ************************************************************ */
/* 																*/
/*  					printing to screen functions     		*/
/* 																*/
/* ************************************************************ */

__host__ std::ostream& operator<< (std::ostream& os, const double6& p){

	// << operator overload to write double6 to screen

	os << std::setw(15) << p.x << std::setw(15) << p.px << std::setw(15) << p.y 
	<< std::setw(15) << p.py << std::setw(15) << p.t <<std::setw(15) << p.delta << std::endl;;
	return os;
 };



void STE_Bunch::printBunchParams(){

	/* Debug function to print out input parameters for checking if they are set correctly */

	std::stringstream ss;
 	ss << "STE_bunch_parameters_" << bucketNumber << "_" << get_date() << ".dat";

 	std::string s;
	s = ss.str();

	std::ofstream ofile(s.c_str());

	// write to file
	ofile << "Bucket                    " << bucketNumber                          << std::endl;
	ofile << "Energy lost per turn      " << energyLostPerTurn                     << std::endl;
	ofile << "tunex                     " << bparams.tunex                         << std::endl;
	ofile << "tuney                     " << bparams.tuney                         << std::endl;
	ofile << "trev                      " << bparams.trev                          << std::endl;
	ofile << "timeratio                 " << bparams.timeratio                     << std::endl;
	ofile << "particleType              " << bparams.particleType                  << std::endl;
	ofile << "particleAtomNumber        " << bparams.particleAtomNumber            << std::endl;
	ofile << "particleCharge            " << bparams.particleCharge                << std::endl;
	ofile << "methodRadiationIntegrals  " << bparams.methodRadiationIntegrals      << std::endl;
	ofile << "tfsfile                   " << bparams.tfsfile                       << std::endl;
	ofile << "numberMacroParticles      " << bparams.numberMacroParticles          << std::endl;
	ofile << "methodLongitudinalDist    " << bparams.methodLongitudinalDist        << std::endl;
	ofile << "acceleratorLength         " << bparams.radparams.acceleratorLength   << std::endl;
	ofile << "betxRingAverage           " << bparams.radparams.betxRingAverage     << std::endl;
	ofile << "betyRingAverage           " << bparams.radparams.betyRingAverage     << std::endl;
	ofile << "gammaTransition           " << bparams.radparams.gammaTransition     << std::endl;
	ofile << "DipoleBendingRadius       " << bparams.radparams.DipoleBendingRadius << std::endl;
	ofile << "ParticleRadius            " << bparams.radparams.ParticleRadius      << std::endl;
	ofile << "ParticleEnergy            " << bparams.radparams.ParticleEnergy      << std::endl;
	ofile << "cq                        " << bparams.radparams.cq                  << std::endl;
	ofile << "gammar                    " << bparams.radparams.gammar              << std::endl;
	ofile << "eta                       " << bparams.radparams.eta                 << std::endl;
	ofile << "omegas                    " << bparams.radparams.omegas              << std::endl;
	ofile << "p0                        " << bparams.radparams.p0                  << std::endl;
	ofile << "ParticleRestEnergy        " << bparams.radparams.ParticleRestEnergy  << std::endl;
	ofile << "trev                      " << bparams.radparams.trev                << std::endl;
	ofile << "eta                       " << bparams.hamparams.eta                 << std::endl;
	ofile << "angularFrequency          " << bparams.hamparams.angularFrequency    << std::endl;
	ofile << "p0                        " << bparams.hamparams.p0                  << std::endl;
	ofile << "betar                     " << bparams.hamparams.betar               << std::endl;
	ofile << "particleCharge            " << bparams.hamparams.particleCharge      << std::endl;
	ofile << "delta                     " << bparams.hamparams.delta               << std::endl;
	ofile << "t                         " << bparams.hamparams.t                   << std::endl;
	ofile << "phis                      " << bparams.hamparams.phis                << std::endl;
	ofile << "particleCharge            " << bparams.synparams.particleCharge      << std::endl;
	ofile << "search                    " << bparams.synparams.search              << std::endl;
	ofile << "searchWidth               " << bparams.synparams.searchWidth         << std::endl;
	ofile << "omega0                    " << bparams.synparams.omega0              << std::endl;
	ofile << "eta                       " << bparams.synparams.eta                 << std::endl;
	ofile << "p0                        " << bparams.synparams.p0                  << std::endl;
	ofile << "seed                      " << bparams.longparams.seed               << std::endl;
	ofile << "betx                      " << bparams.longparams.betx 			   << std::endl;
	ofile << "bety                      " << bparams.longparams.bety 			   << std::endl;
	ofile << "emitx                     " << bparams.longparams.emitx			   << std::endl;
	ofile << "emity                     " << bparams.longparams.emity			   << std::endl;
	ofile << "sigs                      " << bparams.longparams.sigs  			   << std::endl;
	ofile << "omega0                    " << bparams.longparams.omega0			   << std::endl;
	ofile << "v0                        " << bparams.longparams.v0    			   << std::endl;
	ofile << "v1                        " << bparams.longparams.v1    			   << std::endl;
	ofile << "v2                        " << bparams.longparams.v2    			   << std::endl;
	ofile << "h0                        " << bparams.longparams.h0    			   << std::endl;
	ofile << "h1                        " << bparams.longparams.h1    			   << std::endl;
	ofile << "h2                        " << bparams.longparams.h2    			   << std::endl;
	ofile << "p0                        " << bparams.longparams.p0    			   << std::endl;
	ofile << "betar                     " << bparams.longparams.betar 			   << std::endl;
	ofile << "eta                       " << bparams.longparams.eta   			   << std::endl;
	ofile << "charge                    " << bparams.longparams.charge			   << std::endl;
	ofile << "tauhat                    " << tauhat                 			   << std::endl;
	ofile << "betatroncoupling          " << bparams.betatroncoupling  			   << std::endl;
	double6 timesequilib = rad->getDampingAndEquilib();
	ofile << "radequitx                 " << timesequilib.x			         	   << std::endl;
	ofile << "radequity                 " << timesequilib.px			           << std::endl;
	ofile << "radequitt                 " << timesequilib.y			          	   << std::endl;
	ofile << "radequix                  " << timesequilib.py			           << std::endl;
	ofile << "radequiy                  " << timesequilib.t			         	   << std::endl;
	ofile << "radequis                  " << bparams.hamparams.eta  * sqrt( timesequilib.delta  * CUDA_C_F ) / bparams.radparams.omegas  << std::endl;
	radiationIntegrals radint = rad->getIntegrals( 2 );
 	ofile << "I2                        " << radint.I2			         	       << std::endl;
	ofile << "I3                        " << radint.I3			                   << std::endl;
	ofile << "I4x                       " << radint.I4x			          	       << std::endl;
	ofile << "I4y                       " << radint.I4y			                   << std::endl;
	ofile << "I5x                       " << radint.I5x			         	       << std::endl;
	ofile << "I5y                       " << radint.I5y			         	       << std::endl;
	ofile.close();

	// std::cout.precision(19);
	// std::cout << "Bucket                  : " << bucketNumber                         << std::endl;
	// std::cout << "Energy lost per turn    : " << energyLostPerTurn                    << std::endl;
	// std::cout << "tunex                   : " << bparams.tunex                         << std::endl;
	// std::cout << "tuney                   : " << bparams.tuney                         << std::endl;
	// std::cout << "trev                    : " << bparams.trev                          << std::endl;
	// std::cout << "timeratio               : " << bparams.timeratio                     << std::endl;
	// std::cout << "particleType            : " << bparams.particleType                  << std::endl;
	// std::cout << "particleAtomNumber      : " << bparams.particleAtomNumber            << std::endl;
	// std::cout << "particleCharge          : " << bparams.particleCharge                << std::endl;
	// std::cout << "methodRadiationIntegrals: " << bparams.methodRadiationIntegrals      << std::endl;
	// std::cout << "tfsfile                 : " << bparams.tfsfile                       << std::endl;
	// std::cout << "numberMacroParticles    : " << bparams.numberMacroParticles          << std::endl;
	// std::cout << "methodLongitudinalDist  : " << bparams.methodLongitudinalDist        << std::endl;
	// std::cout << "acceleratorLength       : " << bparams.radparams.acceleratorLength   << std::endl;
	// std::cout << "betxRingAverage         : " << bparams.radparams.betxRingAverage     << std::endl;
	// std::cout << "betyRingAverage         : " << bparams.radparams.betyRingAverage     << std::endl;
	// std::cout << "gammaTransition         : " << bparams.radparams.gammaTransition     << std::endl;
	// std::cout << "DipoleBendingRadius     : "  << bparams.radparams.DipoleBendingRadius << std::endl;
	// std::cout << "ParticleRadius          : "  << bparams.radparams.ParticleRadius      << std::endl;
	// std::cout << "ParticleEnergy          : "  << bparams.radparams.ParticleEnergy      << std::endl;
	// std::cout << "cq                      : "  << bparams.radparams.cq                  << std::endl;
	// std::cout << "gammar                  : "  << bparams.radparams.gammar              << std::endl;
	// std::cout << "eta                     : "  << bparams.radparams.eta                 << std::endl;
	// std::cout << "omegas                  : "  << bparams.radparams.omegas              << std::endl;
	// std::cout << "p0                      : "  << bparams.radparams.p0                  << std::endl;
	// std::cout << "ParticleRestEnergy      : "  << bparams.radparams.ParticleRestEnergy  << std::endl;
	// std::cout << "trev                    : "  << bparams.radparams.trev                << std::endl;
	// std::cout << "eta                     : "  << bparams.hamparams.eta                 << std::endl;
	// std::cout << "angularFrequency        : "  << bparams.hamparams.angularFrequency    << std::endl;
	// std::cout << "p0                      : "  << bparams.hamparams.p0                  << std::endl;
	// std::cout << "betar                   : "  << bparams.hamparams.betar               << std::endl;
	// std::cout << "particleCharge          : "  << bparams.hamparams.particleCharge      << std::endl;
	// std::cout << "delta                   : "  << bparams.hamparams.delta               << std::endl;
	// std::cout << "t                       : "  << bparams.hamparams.t                   << std::endl;
	// // std::cout << "voltages                : "  << bparams.hamparams.voltages            << std::endl;
	// // std::cout << "harmonicNumbers         : "  << bparams.hamparams.harmonicNumbers     << std::endl;
	// std::cout << "phis                    : "  << bparams.hamparams.phis                << std::endl;
	// std::cout << "particleCharge          : "  << bparams.synparams.particleCharge      << std::endl;
	// std::cout << "search                  : "  << bparams.synparams.search              << std::endl;
	// std::cout << "searchWidth             : "  << bparams.synparams.searchWidth         << std::endl;
	// // std::cout << "harmonicNumbers         : "  << bparams.synparams.harmonicNumbers     << std::endl;
	// // std::cout << "voltages                : "  << bparams.synparams.voltages            << std::endl;
	// std::cout << "omega0                  : "  << bparams.synparams.omega0              << std::endl;
	// std::cout << "eta                     : "  << bparams.synparams.eta                 << std::endl;
	// std::cout << "p0                      : "  << bparams.synparams.p0                  << std::endl;
	// std::cout << "seed                    : "  << bparams.longparams.seed               << std::endl;
	// std::cout << "betx                    : "  << bparams.longparams.betx 			   << std::endl;
	// std::cout << "bety                    : "  << bparams.longparams.bety 			  << std::endl;
	// std::cout << "emitx                   : "  << bparams.longparams.emitx			  << std::endl;
	// std::cout << "emity                   : "  << bparams.longparams.emity			  << std::endl;
	// std::cout << "sigs                    : "  << bparams.longparams.sigs  			  << std::endl;
	// std::cout << "omega0                  : "  << bparams.longparams.omega0			  << std::endl;
	// std::cout << "v0                      : "  << bparams.longparams.v0    			  << std::endl;
	// std::cout << "v1                      : "  << bparams.longparams.v1    			  << std::endl;
	// std::cout << "v2                      : "  << bparams.longparams.v2    			  << std::endl;
	// std::cout << "h0                      : "  << bparams.longparams.h0    			  << std::endl;
	// std::cout << "h1                      : "  << bparams.longparams.h1    			  << std::endl;
	// std::cout << "h2                      : "  << bparams.longparams.h2    			  << std::endl;
	// std::cout << "p0                      : "  << bparams.longparams.p0    			  << std::endl;
	// std::cout << "betar                   : "  << bparams.longparams.betar 			  << std::endl;
	// std::cout << "eta                     : "  << bparams.longparams.eta   			  << std::endl;
	// std::cout << "charge                  : "  << bparams.longparams.charge			  << std::endl;
 }	

void STE_Bunch::printDistribution(){

	/* Print current distribution to screen */

	std::copy(distribution.begin(),distribution.end(),std::ostream_iterator<double6>(std::cout));
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
	thrust::device_vector<double> hist;
	hist = ibs->getSqrtHistogram();
	std::copy(hist.begin() , 
		hist.end() , 
		std::ostream_iterator<double>( std::cout, "\n") 
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
		std::ostream_iterator<double6>( ofile )
		);

	// std::cout << "writing vector size = " << distribution.size() << std::endl;
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
	for(std::map<int, double6>::iterator me = mapEmittances.begin(); me != mapEmittances.end(); me++)
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
	for(std::map<int, double3>::iterator mi = IBSLifeTimes.begin(); mi != IBSLifeTimes.end(); mi++)
	{
		ofile << std::setw(6) << mi->first <<  mi->second ;
	}
	ofile.close();
 }

void STE_Bunch::writeIntensity(){
 	std::stringstream ss;
 	ss << "STE_Intensity_" << bucketNumber << "_" << get_date() << ".dat";

 	std::string s;
	s = ss.str();

	std::ofstream ofile(s.c_str());

	// write headers
	ofile << std::setw(15) << "Turn" << std::setw(15) << "NMacro" 
			<< std::setw(15) << "Current" 
			<< std::setw(15) << "NReal" 
			<< std::setw(15) << "NRealLost" 
			<< std::setw(15) << "Ttou" 
			<< std::endl;

 	// printing the int maps
	for(std::map<int, double6>::iterator mi = mapIntensity.begin(); mi != mapIntensity.end(); mi++)
	{
		ofile  <<  mi->second ;
	}
	ofile.close();
 }
  