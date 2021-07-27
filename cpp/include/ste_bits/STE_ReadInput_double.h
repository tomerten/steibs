// include guard
#ifndef READCLASS_H_INCLUDED
#define READCLASS_H_INCLUDED

// included dependencies
#include <map>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <math.h>
#include <string>



class STE_ReadInput
{
public:
	STE_ReadInput( std::string );
	void ReadInput();
	void PrintInput();

	void setNumberOfMacroParticles( std::string , int );
	int getNumberOfMacroParticles();

	void setRandomSeed( std::string , int );
	int getRandomSeed();

	void setTimeRatio( std::string , int );
	int getTimeRatio();

	void setNumberOfTurns( std::string, int );
	int getNumberOfTurns();

	void setWriteDistribution( std::string, int );
	int getWriteDistribution();

	void setWriteTurn( std::string, int);
	int getWriteTurn();

	void setMethodRadiationIntegrals( std::string , int );
	int getMethodRadiationIntegrals();
	
	void setMethodLongitudinalDistribution( std::string , int );
	int getMethodLongitudinalDistribution();

	void setDipoleBendingRadius( std::string , double );
	double getDipoleBendingRadius();

	void setRFVoltageV1( std::string , double );
	void setRFVoltageV2( std::string , double );
	void setRFVoltageV3( std::string , double );

	void setHarmonicNumber1( std::string, double );
	void setHarmonicNumber2( std::string, double );
	void setHarmonicNumber3( std::string, double );


	double getRFVoltageV1();
	double getRFVoltageV2();
	double getRFVoltageV3();

	double getHarmonicNumber1();
	double getHarmonicNumber2();
	double getHarmonicNumber3();

	void setTfsFile1( std::string, std::string );
	void setTfsFile2( std::string, std::string );

	std::string getTfsFile1();
	std::string getTfsFile2();

	void setBunchFileName1( std::string ,std::string );
	void setBunchFileName2( std::string ,std::string );

	std::string getBunchFileName1();
	std::string getBunchFileName2();

	void setParticleType1( std::string , std::string );
	void setParticleType2( std::string , std::string );

	void setParticleAtomNumber1( std::string , int );
	void setParticleAtomNumber2( std::string , int );

	std::string getParticleType1();
	std::string getParticleType2();

	int getParticleAtomNumber1();
	int getParticleAtomNumber2();

	void setBetatronCoupling( std::string, double );
	double getBetatronCoupling();

	void setk2l( std::string, double );
	double getk2l();

	void setk2sl( std::string, double );
	double getk2sl();


	void setRadiationDamping( std::string , int );
	int getRadiationDamping();

	void setBetatron( std::string , int );
	int getBetatron();


	void setNumberOfBins( std::string , int );
	int getNumberofBins();

	void setMethodIBS( std::string , int );
	int getMethodIBS();


	void setibsCoupling( std::string, double );
	double getibsCoupling();

	void setibsfrac( std::string , double );
	double getibsfrac();




	void setIBSupdate( std::string , int );
	int getIBSupdate();

	void setRFupdate( std::string , int );
	int getRFupdate();

	void setRFmix( std::string , double );
	double getRFmix();

	void setnwrite( std::string , int );
	int getnwrite();

	void setfmohlNumPoints( std::string , int);
	int getfmohlNumPoints();

	void setcoulomblog( std::string , double );
	double getcoulomblog();

	void setblowupx( std::string , double );
	double getblowupx();

	void setblowupy( std::string , double );
	double getblowupy();



	std::string getFilename();
	void setFilename( std::string );

	// ~STE_ReadInput();

private:
	std::map<std::string,int> inputMapInt;
	std::map<std::string,double> inputMapdouble;
	std::map<std::string,std::string> inputMapString;
	std::string filename;
};

#endif