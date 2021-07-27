#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <fstream>
// #include <array> c++11 standard 
#include <algorithm>

#include "ste_global_functions.cu"
// #include "radiation_damping.cu"
// #include "read_tfs.cu"

using namespace std;

int main(int argc, char const *argv[])
{
	int n = 100;
	// ************************
	// testing random generator
	// ************************
	cout << "Test of random generator function" << endl;	
	cout << ran3(12489) << endl;

	// **********************************
	// testing random generator on device
	// **********************************
	cout << "Test of random generator on device" << endl;	
	// create mem space
	thrust::device_vector<float> Y(10);
	// fill with random numbers
	thrust::fill(Y.begin(),Y.end(),ran3(12489));
	// print to screen
	// thrust::copy(Y.begin(),Y.end(),std::ostream_iterator<float>(std::cout,"\n"));

	// *********************************************************
	// create an xor_combine_engine from minstd_rand and minstd_rand0
	// *********************************************************
	cout << "Test of thrust XOR random generator " << endl;
  	// use a shift of 0 for each
  	thrust::xor_combine_engine<thrust::minstd_rand,0,thrust::minstd_rand0,0> rng;
  	// print a random number to standard output
  	std::cout << (float)rng()/(float)rng.max << std::endl;

  	// *********************************************************
	// test xor random generator on device
	// *********************************************************
	cout << "Test of thrust XOR random generator " << endl;
	
	thrust::device_vector<float> X(n);
	// thrust::transform(X.begin(),X.end(),X.begin(),randxor_functor<float>(seed));
	// thrust::transform(X.begin(), X.end(), X.begin(), sigmoid_function<float>(0.1));
	// thrust::transform(X.begin(), X.end(), X.begin(), randxor_functor<float>(12489));
	// thrust::for_each(thrust::counting_iterator<int>(0),thrust::counting_iterator<int>(10),randxor_functor<float>(12489));

	// apply function using iterator
	thrust::transform(thrust::counting_iterator<int>(0),thrust::counting_iterator<int>(n),X.begin(),randxor_functor<float>(12489));
	// copy output to screen
	// thrust::copy(X.begin(),X.end(),std::ostream_iterator<float>(std::cout, "\n"));

	// *********************************************************
	// test xor bigaussian random generator on device and write to file
	// *********************************************************
	cout << "Test of bi-gaussian generator " << endl;
	thrust::device_vector<float2> vec2(n);
	thrust::transform(thrust::counting_iterator<int>(0),thrust::counting_iterator<int>(n),vec2.begin(),rand_2d_gauss<float2>(make_float3(12489,10.0,1e-9)));
	// std::copy(vec2.begin(),vec2.end(),std::ostream_iterator<float2>(std::cout, "\n"));

	ofstream ofile("rands_bigaussian_out.txt");
	thrust::copy(vec2.begin(),vec2.end(), ostream_iterator<float2>(ofile, "\n"));
	ofile.close();

	// *********************************************************
	// test root finding algorithm for finding synchronous phases
	// *********************************************************
	cout << "Test of root/synchronous phase finding algorithm " << endl;
	// setting input variable
	double energyLostPerTurn = 170000; // radation losses per turn per particle
	double acceleratorLength = 240.00839; // length in meter
	double gammar = 3326.817037; // relativistic gamma 
	double eta = 0.0007038773471 - 1 / pow(gammar ,2); // slip factor approx alpha - 1/ gammar**2
	double betar = sqrt(1-1 / pow(gammar ,2)); // relativistic beta
	double trev = acceleratorLength / (betar * CUDA_C_F);
	float h0 = 400.0;
	float h1 = 1200.0;
	float h2 = 1400.0;
	float v0 = 1.4e6;
	float v1 = 20.0e6;
	float v2 = 17.14e6;
	double omega0 = (2 * CUDA_PI_F) / trev;
	double p0 = 1.7e9;
	float charge = -1.0;

	// In order to find roots in the right intervals and locations we need the period of the RF system with the lowest freq and the period of the 
	// RF system with the highest freqency
	// search locations  : 0 and period RF low
	// interval width : +/- period RF high
	double search1 = trev * h0 * omega0/ (8* max(max(h0,h1),h2) ); // give positive offset to find upstream root and not downstream root
	double search2 = trev * h0 * omega0/ (8* max(max(h0,h1),h2) ) - trev * h0 * omega0/ min(min(h0,h1),h2); // give positive offset to find upstream root and not downstream root
	double searchWidth = trev * h0 * omega0/ (2* max(max(h0,h1),h2) );
	double synchronousPhase0 = synchronousPhaseFunctorDeriv(energyLostPerTurn,make_float3(v0,v1,v2),make_float3(h0,h1,h2),charge,search1,search1 - searchWidth,search1 + searchWidth);
	double synchronousPhase1 = synchronousPhaseFunctorDeriv(energyLostPerTurn,make_float3(v0,v1,v2),make_float3(h0,h1,h2),charge,search2,search2 - searchWidth,search2 + searchWidth);

	cout << "synchronous phase around " << search1 << " : " << synchronousPhase0 << endl;
	cout << "synchronous phase around " << search2 << " : " << synchronousPhase1 << endl;

	cout << "Find next extremum of Hamiltonian" << endl; //upstream as at synchronous phases dH/dt > 0

	double synchronousPhase0Next = synchronousPhaseFunctorDeriv(energyLostPerTurn,make_float3(v0,v1,v2),make_float3(h0,h1,h2),charge,
		search1 + searchWidth,search1+searchWidth/2 ,search1 + 2*searchWidth);
	double synchronousPhase1Next = synchronousPhaseFunctorDeriv(energyLostPerTurn,make_float3(v0,v1,v2),make_float3(h0,h1,h2),charge,
		search2 + searchWidth,search2+searchWidth/2 ,search2 + 2*searchWidth);

	cout << "synchronous next phase around " << search1 + searchWidth << " : " << synchronousPhase0Next << "," << synchronousPhase0Next / (h0 * omega0) << endl;
	cout << "synchronous next phase around " << search2 + searchWidth << " : " << synchronousPhase1Next << "," << synchronousPhase1Next / (h0 * omega0) << endl;

	// *********************************************************
	// test hamiltonian and related functions
	// *********************************************************
	cout << "Test of Hamiltonian and related functions" << endl;
	float tcoeffValue = tcoeff(eta,omega0,h0);
	cout << "tcoeff " << tcoeffValue / (h0 * omega0) << endl;
	cout << endl;
	float hamMax = HamiltonianTripleRf(tcoeffValue, make_float3(v0,v1,v2), make_float3(h0,h1,h2), synchronousPhase1, synchronousPhase1Next/(h0*omega0), 0.0 , omega0, p0, betar,charge);

	cout << "Maximum value Hamiltonian : " << hamMax << " at phi = " << synchronousPhase1Next << " or t =" << synchronousPhase1Next / (h0 * omega0) << endl;


	// *********************************************************
	// test xor bigaussian random generator on device 6D and synchrotron tune
	// *********************************************************
	cout << "Test of bi-gaussian generator 6D for two bunches - bunch 1 phis ~ 0" << endl;

	thrust::device_vector<float6> vec60(n);
	thrust::device_vector<float6> vec61(n);
	longitudinalParameters inputvar;
	inputvar.seed = 12486;
	inputvar.betx = 10.0;
	inputvar.bety = 20.0;
	inputvar.emitx = 1e-9;
	inputvar.emity = 2e-9;
	inputvar.tauhat = abs((synchronousPhase0Next - synchronousPhase0)/(h0 * omega0));
	inputvar.sigs = 0.0028;
	inputvar.sige = 0.0007;
	inputvar.omega0 = omega0;
	inputvar.v0 = v0;
	inputvar.v1 = v1;
	inputvar.v2 = v2;
	inputvar.h0 = h0;
	inputvar.h1 = h1;
	inputvar.h2 = h2;
	inputvar.phiSynchronous = synchronousPhase0;
	inputvar.p0 = p0;
	inputvar.betar = betar;
	inputvar.eta = eta;
	inputvar.hammax = hamMax;
	inputvar.charge = charge;

	cout << "Synchrotron tune : " << synchrotronTune(inputvar) << endl;
	cout << "Synchrotron tune (Hz): " << synchrotronTune(inputvar) * omega0 / (2 * CUDA_PI_F)<< endl;

	// bunch around phis = 0
	thrust::transform(thrust::counting_iterator<int>(0),thrust::counting_iterator<int>(n),vec60.begin(),rand_6d_gauss<float2>(inputvar));
	// std::copy(vec60.begin(),vec60.end(),std::ostream_iterator<float6>(std::cout, "\n"));
	ofstream ofile2("rands_6D_bigauss0_out.txt");
	thrust::copy(vec60.begin(),vec60.end(), ostream_iterator<float6>(ofile2, "\n"));
	ofile2.close();

	cout << "Test of bi-gaussian generator 6D for two bunches - bunch 1 phis ~ -2 pi" << endl;
	// bunch around phis = -2 pi
	inputvar.sigs = 0.0075;
	inputvar.phiSynchronous = synchronousPhase1;
	inputvar.tauhat = abs((synchronousPhase1Next - synchronousPhase1)/(h0 * omega0));

	cout << "Synchrotron tune : " << synchrotronTune(inputvar) << endl;
	cout << "Synchrotron tune (Hz): " << synchrotronTune(inputvar) * omega0 / (2 * CUDA_PI_F)<< endl;

	inputvar.seed = 2;
	thrust::transform(thrust::counting_iterator<int>(0),thrust::counting_iterator<int>(n),vec61.begin(),rand_6d_gauss<float2>(inputvar));
	// std::copy(vec61.begin(),vec61.end(),std::ostream_iterator<float6>(std::cout, "\n"));
	ofstream ofile3("rands_6D_bigauss1_out.txt");
	thrust::copy(vec61.begin(),vec61.end(), ostream_iterator<float6>(ofile3, "\n"));
	ofile3.close();


	// *********************************************************
	// test xor bigaussian random generator on device 6D matched
	// *********************************************************
	cout << "Test of bi-gaussian generator 6D longitudinal matched for two bunches - bunch 1 phis ~ 0" << endl;
	inputvar.tauhat = abs((synchronousPhase0Next - synchronousPhase0)/(h0 * omega0));
	inputvar.sigs = 0.0028;
	inputvar.phiSynchronous = synchronousPhase0;
	// bunch around phis = 0
	inputvar.seed = 12486;
	thrust::transform(thrust::counting_iterator<int>(0),thrust::counting_iterator<int>(n),vec60.begin(),rand6DTransverseBiGaussLongitudinalMatched<float6>(inputvar));
	// std::copy(vec60.begin(),vec60.end(),std::ostream_iterator<float6>(std::cout, "\n"));
	ofstream ofile4("rands_6D_bigauss0_matched_out.txt");
	thrust::copy(vec60.begin(),vec60.end(), ostream_iterator<float6>(ofile4, "\n"));
	ofile4.close();

	cout << "Test of bi-gaussian generator 6D longitudinal matched for two bunches - bunch 1 phis ~ -2 pi" << endl;
	// bunch around phis = -2 pi
	inputvar.sigs = 0.0075;
	inputvar.phiSynchronous = synchronousPhase1;
	inputvar.tauhat = abs((synchronousPhase1Next - synchronousPhase1)/(h0 * omega0));
	inputvar.seed = 2;
	thrust::transform(thrust::counting_iterator<int>(0),thrust::counting_iterator<int>(n),vec61.begin(),rand6DTransverseBiGaussLongitudinalMatched<float2>(inputvar));
	// std::copy(vec61.begin(),vec61.end(),std::ostream_iterator<float6>(std::cout, "\n"));
	ofstream ofile5("rands_6D_bigauss1_matched_out.txt");
	thrust::copy(vec61.begin(),vec61.end(), ostream_iterator<float6>(ofile5, "\n"));
	ofile5.close();

	// *********************************************************
	// test square functor on float6
	// *********************************************************
	cout << "Test square functor on float6" << endl;

	// thrust::transform(vec60.begin(),vec60.end(),vec60.begin(),squareFunctor<float6> ());
	// float6 outsum = thrust::reduce(vec60.begin(),vec60.end(),sum,addFloat6());
	// std::copy(vec60.begin(),vec60.end(),std::ostream_iterator<float6>(std::cout, "\n"));
	// cout << "[" << outsum.x << "," << outsum.px << "," << outsum.y << ","<< outsum.py << ","<< outsum.t << ","<< outsum.delta << "]" << endl;
	// x -> ex, y-> ey delta->sigE t->sigt
	int nTurns = 1000;
	int nWrite = 1;
	int arrLength = nTurns / nWrite;
	float6 emit1[arrLength];
	float6 emit2[arrLength];

	emit1[0] = caluculateEmittance(vec60, make_float2(10.,20.),acceleratorLength,n,gammar);
	emit2[0] = caluculateEmittance(vec61, make_float2(10.,20.),acceleratorLength,n,gammar);
	cout <<  "emittances" << endl;
	cout << emit1[0] << endl;
 	cout << emit2[0] << endl;
 	cout <<  "check if original distribution has changed by operations" << endl;
 	// std::copy(vec60.begin(),vec60.end(),std::ostream_iterator<float6>(std::cout, "\n"));

 	// *********************************************************
	// test reading of tfs file
	// *********************************************************
	cout << "Testing the reading of twiss table" << endl;

 	vector<vector<string> > out;
	vector<vector<float> > fout;

	// twiss file to use
	string in = "/home/tmerten/mad/2017-12-21/twiss/Long-corrected-LongPMMM-2017-12-21.tfs";
	out = ReadTfsTable(in,false);
	map<string, int> maptest = mapOfColumns(out[0],false);
	fout = TfsTableConvertStringToFloat(out,maptest);
	thrust::device_vector<tfsTableData> testdata;
	testdata = TfsTableToDevice(fout);
	cout << endl;
	cout << "Printing first 10 rows and first 4 columns of tfs table data" << endl;
	std::copy(testdata.begin(),testdata.begin()+10,std::ostream_iterator<tfsTableData>(std::cout));
	
	// *********************************************************
	// test rfupdate 
	// *********************************************************

	inputvar.phiSynchronous = synchronousPhase0;
	inputvar.tauhat = abs((synchronousPhase0Next - synchronousPhase0)/(h0 * omega0));
	thrust::transform(vec60.begin(),vec60.end(),vec60.begin(),RfUpdateRoutineFunctor(1.0f,inputvar));

	// remove element out of phase-space acceptance
	thrust::device_vector<float6>::iterator new_end = thrust::remove_if(vec60.begin(),vec60.end(),isInLong(inputvar.tauhat));

	// registering the debunching losses
	int debunchlosses = n - (new_end - vec60.begin());
	cout << "Particles lost due to debunching : " << debunchlosses << endl;

	// resizing the vector
	vec60.resize(new_end - vec60.begin());

	ofstream ofile7("rands_6D_bigauss0_matched_out_rfupdate.txt");
	thrust::copy(vec60.begin(),vec60.end(), ostream_iterator<float6>(ofile7, "\n"));
	ofile7.close();


	// *********************************************************
	// test radiation damping
	// *********************************************************
	cout << "Testing the intial radiation damping calcuations" << endl;

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

	cout << "Radiation integrals using approx routine" << endl;
	radtest = CalculateRadiationIntegralsApprox(intpars);
	cout << radtest<< endl;

	cout << "Radiation integrals using lattice element by element routine" << endl;
	radiationIntegrals total =  CalculateRadiationIntegralsLatticeRing(testdata, intpars);
	cout << total << endl;

	cout << "Radiation damping times and equilibrium (for short and long bunch)" << endl;

	float6 timesequilib = CalculateRadiationDampingTimesAndEquilib(intpars, total);
	cout << setw(15) << "t_emitx(s)" << setw(15) << "t_emity(s)" << setw(15) << "t_emits(s)" << setw(15) << "emitx" << setw(15) << "emity" << setw(15) << "sigt"  << endl;
	cout << timesequilib<< endl;
	intpars.omegas = 0.00784862 * omega0;
	timesequilib = CalculateRadiationDampingTimesAndEquilib(intpars, total);
	cout << timesequilib<< endl;


	
	float6 sumavg;
	sumavg.x= 0.0;
	sumavg.px=0.0;
	sumavg.y=0.0;
	sumavg.py=0.0;
	sumavg.t=0.0;
	sumavg.delta=0.0;
	float6 deltaavg = thrust::reduce(vec60.begin(),vec60.end(),sumavg,addFloat6());
	cout << "writing average of delta" << deltaavg.delta / n << endl;

	thrust::transform(vec60.begin(),vec60.end(),vec60.begin(),RadiationDampingRoutineFunctor(timesequilib,trev,1.0,12489));
	ofstream ofile6("rands_6D_bigauss0_matched_out_rfupdate_raddamp.txt");
	thrust::copy(vec60.begin(),vec60.end(), ostream_iterator<float6>(ofile6, "\n"));
	ofile6.close();
	emit1[1] = caluculateEmittance(vec60, make_float2(10.,20.),acceleratorLength,n,gammar);
	
	cout <<  "emittances" << endl;
	cout << emit1[1] << endl;
	// Reading tfs header
	map<string, float> tfsh = ReadTfsHeader(in,false);

	// for(map<string, float>::const_iterator it = tfsh.begin();it != tfsh.end(); ++it)
	// 		{
	// 		 
	   // cout << it->first << " " << it->second << "\n";
	// 		}
	float qx = tfsh.at("Q1");
	float qy = tfsh.at("Q2");
	float ksix = tfsh.at("DQ1");
	float ksiy = tfsh.at("DQ2");
 	cout << "Horizontal Tune   : " << qx << endl;
 	cout << "Vertical   Tune   : " << qy << endl;
 	cout << "Horizontal Chroma : " << ksix << endl;
 	cout << "Vertical   Chroma : " << ksiy << endl;

 	// *********************************************************
	// test betatron motion
	// *********************************************************
	cout << "Testing the betatron motion" << endl;
	for(int i=0; i<=10; i++){
		thrust::transform(vec60.begin(),vec60.end(),vec60.begin(),BetatronRoutine(qx,qy,ksix,ksiy,0.0f,0.0f,0.0f));
		std::stringstream ss;
		ss << "rands_6D_bigauss0_matched_out_rfupdate_raddamp_betatron_" << i << ".txt";
		std::string s;
		s = ss.str();
		std::ofstream ofile(s.c_str());
		thrust::copy(vec60.begin(),vec60.end(), ostream_iterator<float6>(ofile));
		// ofstream ofile8("rands_6D_bigauss0_matched_out_rfupdate_raddamp_betatron.txt");
		// thrust::copy(vec60.begin(),vec60.end(), ostream_iterator<float6>(ofile8, "\n"));
		// ofile8.close();
		ofile.close();
	};
	emit1[2] = caluculateEmittance(vec60, make_float2(10.,20.),acceleratorLength,n,gammar);
	
	cout <<  "emittances" << endl;
	cout << emit1[2] << endl;

	// *********************************************************
	// test getting time component fro binning
	// *********************************************************
	thrust::device_vector<int> hist;
	hist = ParticleTimesToHistogram(vec60,100,inputvar.tauhat);
	// std::copy(hist.begin(),hist.end(),std::ostream_iterator<int>(std::cout, "\n"));
	float6 avg = CalcRMS(vec60,n);
	cout << "RMS" << avg << endl;

	ibsparameters ibsparams;
	ibsparams.acceleratorLength = acceleratorLength;
	ibsparams.gammaTransition = 1 / sqrt(0.0007038773471);
	ibsparams.betx = acceleratorLength / (2 * CUDA_PI_F * qx);
	ibsparams.bety = acceleratorLength / (2 * CUDA_PI_F * qy);
	ibsparams.emitx = 0.5e-9;
	ibsparams.emity = 1.0e-9;
	ibsparams.dponp = avg.delta;
	ibsparams.ParticleRadius= CUDA_C_R_ELECTRON;;
	ibsparams.betar=betar;
	ibsparams.gamma0=gammar;
	ibsparams.numberRealParticles =1.0e8;
	ibsparams.sigs = avg.t * CUDA_C_F * betar;
	ibsparams.tauhat = abs((synchronousPhase0Next - synchronousPhase0)/(h0 * omega0));
	ibsparams.ibsCoupling = 0.0;
	ibsparams.fracibstot = 1.0 ; // artificially increase or decrease ibs with a scaling factor
	ibsparams.timeratio = 1.0;
	ibsparams.numberMacroParticles = n;
	ibsparams.trev = trev;
	ibsparams.nbins = 100;
	ibsparams.fmohlNumPoints = 1000;
	ibsparams.coulomblog = 20.0;
	// int m = testdata.size();
	// thrust::device_vector<float3> ibsgrowthratestfs(m);
 //  	thrust::transform(testdata.begin(),testdata.end(),ibsgrowthratestfs.begin(),ibsPiwinskiLattice(ibsparams));
 //  	float3 ibsgrowthrates = thrust::reduce(ibsgrowthratestfs.begin(),ibsgrowthratestfs.end(),make_float3(0.0,0.0,0.0),addFloat3());
 //  	cout << ibsgrowthrates.x << endl;
	float4 ibscoeff = CalcIbsCoeff(ibsparams,0,inputvar.phiSynchronous,testdata);
	cout << "ibs coeffs smooth " << ibscoeff << endl;

	ibscoeff = CalcIbsCoeff(ibsparams,1,inputvar.phiSynchronous,testdata);
	
	cout << "ibs coeffs lattice " << ibscoeff << endl;

	ibscoeff = CalcIbsCoeff(ibsparams,2,inputvar.phiSynchronous,testdata);
	
	cout << "ibs coeffs lattice modified " << ibscoeff << endl;

	ibscoeff = CalcIbsCoeff(ibsparams,3,inputvar.phiSynchronous,testdata);
	
	cout << "ibs coeffs nagaitsev " << ibscoeff << endl;


	thrust::device_vector<float> sqrttest = HistogramToSQRTofCumul(hist,ibscoeff.w);
	cout << "sqrt hist test "  << endl;
	thrust::device_vector<float>::iterator end = sqrttest.end()-1; // .end() iterator returns an iterator on past the last element
	cout << *end << endl;
	// thrust::copy(sqrttest.begin(),sqrttest.end(),std::ostream_iterator<float>(std::cout, "\n"));

	// *********************************************************
	// test piwilattice
	// *********************************************************


	thrust::transform(vec60.begin(),vec60.end(),vec60.begin(),ibsRoutine(ibscoeff, *end, ibsparams));
	ofstream ofile9("rands_6D_bigauss0_matched_out_rfupdate_raddamp_betatron_ibs.txt");
	thrust::copy(vec60.begin(),vec60.end(), ostream_iterator<float6>(ofile9, "\n"));
	ofile9.close();


	return 0;
}