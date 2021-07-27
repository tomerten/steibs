// included dependencies
#include <map>
#include <fstream>
#include <sstream>
#include <boost/lexical_cast.hpp>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <exception>
#include <iterator>

// contains class to read standardized simulation in put file
#include "STE_ReadInput_double.h"

// object constructor
STE_ReadInput::STE_ReadInput( std::string  fn )
{
	setFilename( fn );
	ReadInput();
} 
// end constructor


// function to set filename
void STE_ReadInput::setFilename( std::string fn )
{
	filename = fn;
} 
// end function to set filename


// function for setting number of macro particles
void STE_ReadInput::setNumberOfMacroParticles(std::string k, int v)
{
	if (k=="numberOfMacroParticles") 
		inputMapInt[k] = v;
	else
	{
		std::cout << "Key should be numberOfMacroParticles" << std::endl;
		throw std::exception();
	}
}
// end of funtion set number of macro particles

// function for setting random seed
void STE_ReadInput::setRandomSeed(std::string k, int v)
{
	if (k=="seed") 
		inputMapInt[k] = v;
	else
	{
		std::cout << "Key should be seed" << std::endl;
		throw std::exception();
	}
}
// end of funtion set random seed





// function to set method for calculating radiation integrals
void STE_ReadInput::setMethodRadiationIntegrals(std::string k, int v)
{
	if (k=="methodRadiationIntegral") 
	{
		inputMapInt[k] = v;
	}
	else
	{
		std::cout << "Key should be methodRadiationIntegral" << std::endl;
		throw std::exception();
	}
}
// end of function to set radiation integrals method


// function to set method for generating longitudinal distribution
void STE_ReadInput::setMethodLongitudinalDistribution(std::string k, int v)
{
	if (k=="methodLongitudinalDist") 
	{
		inputMapInt[k] = v;
	}
	else
	{
		std::cout << "Key should be methodLongitudinalDist" << std::endl;
		throw std::exception();
	}
}
// end of function to generating longitudinal distribution

// funtion to set dipole bending radius
void STE_ReadInput::setDipoleBendingRadius( std::string k, double v)
{
	if (k=="dipoleBendingRadius") 
	{
		inputMapdouble[k] = v;
	}
	else
	{
		std::cout << "Key should be dipoleBendingRadius" << std::endl;
		throw std::exception();
	}
}
// end set dipole bending radius

// set RF voltage 1
void STE_ReadInput::setRFVoltageV1( std::string k, double v )
{
	if (k=="RFVoltage1") 
	{
		inputMapdouble[k] = v;
	}
	else
	{
		std::cout << "Key should be RFVoltage1" << std::endl;
		throw std::exception();
	}
}
// end set RF voltage 1

// set RF voltage 2
void STE_ReadInput::setRFVoltageV2( std::string k, double v )
{
	if (k=="RFVoltage2") 
	{
		inputMapdouble[k] = v;
	}
	else
	{
		std::cout << "Key should be RFVoltage2" << std::endl;
		throw std::exception();
	}
}
// end set RF voltage 2

// set RF voltage 3
void STE_ReadInput::setRFVoltageV3( std::string k, double v )
{
	if (k=="RFVoltage3") 
	{
		inputMapdouble[k] = v;
	}
	else
	{
		std::cout << "Key should be RFVoltage3" << std::endl;
		throw std::exception();
	}
}
// end set RF voltage 3

// set RF HarmonicNumber1
void STE_ReadInput::setHarmonicNumber1( std::string k, double v )
{
	if (k=="HarmonicNumber1") 
	{
		if (v == 0.0)
			v = 1.0;
		inputMapdouble[k] = v;
	}
	else
	{
		std::cout << "Key should be HarmonicNumber1" << std::endl;
		throw std::exception();
	}
}
// end set RF HarmonicNumber1

// set RF HarmonicNumber2
void STE_ReadInput::setHarmonicNumber2( std::string k, double v )
{
	if (k=="HarmonicNumber2") 
	{
		if (v == 0.0)
			v = 1.0;
		inputMapdouble[k] = v;
	}
	else
	{
		std::cout << "Key should be HarmonicNumber2" << std::endl;
		throw std::exception();
	}
}
// end set RF HarmonicNumber2

// set RF HarmonicNumber3
void STE_ReadInput::setHarmonicNumber3( std::string k, double v )
{
	if (k=="HarmonicNumber3") 
	{
		if (v == 0.0)
			v = 1.0;
		inputMapdouble[k] = v;
	}
	else
	{
		std::cout << "Key should be HarmonicNumber3" << std::endl;
		throw std::exception();
	}
}
// end set RF HarmonicNumber3

void STE_ReadInput::setTfsFile1( std::string k, std::string v)
{
	if (k=="tfsFile1") 
	{
		inputMapString[k] = v;
	}
	else
	{
		std::cout << "Key should be tfsFile1" << std::endl;
		throw std::exception();
	}
}

void STE_ReadInput::setTfsFile2( std::string k, std::string v)
{
	if (k=="tfsFile2") 
	{
		inputMapString[k] = v;
	}
	else
	{
		std::cout << "Key should be tfsFile2" << std::endl;
		throw std::exception();
	}
}

void STE_ReadInput::setBunchFileName1( std::string k, std::string v)
{
	if (k=="bunchFile1") 
	{
		inputMapString[k] = v;
	}
	else
	{
		std::cout << "Key should be bunchFile1" << std::endl;
		throw std::exception();
	}
}

void STE_ReadInput::setBunchFileName2( std::string k, std::string v)
{
	if (k=="bunchFile2") 
	{
		inputMapString[k] = v;
	}
	else
	{
		std::cout << "Key should be bunchFile2" << std::endl;
		throw std::exception();
	}
}

void STE_ReadInput::setTimeRatio( std::string k , int v )
{
	if (k=="timeRatio") 
	{
		inputMapInt[k] = v;
	}
	else
	{
		std::cout << "Key should be timeRatio" << std::endl;
		throw std::exception();
	}
}

void STE_ReadInput::setParticleType1( std::string k, std::string v)
{
	if (k=="particleType1") 
	{
		inputMapString[k] = v;
	}
	else
	{
		std::cout << "Key should be particleType1" << std::endl;
		throw std::exception();
	}
}

void STE_ReadInput::setParticleType2( std::string k, std::string v)
{
	if (k=="particleType2") 
	{
		inputMapString[k] = v;
	}
	else
	{
		std::cout << "Key should be particleType2" << std::endl;
		throw std::exception();
	}
}

void STE_ReadInput::setParticleAtomNumber1( std::string k , int v)
{
	if (k=="particleAtomNumber1") 
	{
		inputMapInt[k] = v;
	}
	else
	{
		std::cout << "Key should be particleAtomNumber1" << std::endl;
		throw std::exception();
	}
}

void STE_ReadInput::setParticleAtomNumber2( std::string k , int v)
{
	if (k=="particleAtomNumber2") 
	{
		inputMapInt[k] = v;
	}
	else
	{
		std::cout << "Key should be particleAtomNumber2" << std::endl;
		throw std::exception();
	}
}

void STE_ReadInput::setNumberOfTurns( std::string k , int v)
{
	if (k=="numberOfTurns") 
	{
		inputMapInt[k] = v;
	}
	else
	{
		std::cout << "Key should be numberOfTurns" << std::endl;
		throw std::exception();
	}
}

void STE_ReadInput::setWriteDistribution( std::string k , int v)
{
	if (k=="writeDistribution") 
	{
		inputMapInt[k] = v;
	}
	else
	{
		std::cout << "Key should be writeDistribution" << std::endl;
		throw std::exception();
	}
}

void STE_ReadInput::setWriteTurn( std::string k , int v)
{
	if (k=="writeTurn") 
	{
		inputMapInt[k] = v;
	}
	else
	{
		std::cout << "Key should be writeTurn" << std::endl;
		throw std::exception();
	}
}

void STE_ReadInput::setRadiationDamping( std::string k , int v)
{
	if (k=="radiationDamping") 
	{
		inputMapInt[k] = v;
	}
	else
	{
		std::cout << "Key should be radiationDamping" << std::endl;
		throw std::exception();
	}
}

void STE_ReadInput::setBetatron( std::string k , int v)
{
	if (k=="betatron") 
	{
		inputMapInt[k] = v;
	}
	else
	{
		std::cout << "Key should be betatron" << std::endl;
		throw std::exception();
	}
}


void STE_ReadInput::setBetatronCoupling( std::string k, double v )
{
	if (k=="betatronCoupling") 
	{

		inputMapdouble[k] = v;
	}
	else
	{
		std::cout << "Key should be betatronCoupling" << std::endl;
		throw std::exception();
	}
}

void STE_ReadInput::setk2l( std::string k, double v )
{
	if (k=="k2l") 
	{
		
		inputMapdouble[k] = v;
	}
	else
	{
		std::cout << "Key should be k2l" << std::endl;
		throw std::exception();
	}
}

void STE_ReadInput::setk2sl( std::string k, double v )
{
	if (k=="k2sl") 
	{
		
		inputMapdouble[k] = v;
	}
	else
	{
		std::cout << "Key should be k2sl" << std::endl;
		throw std::exception();
	}
}


void STE_ReadInput::setNumberOfBins( std::string k , int v)
{
	if (k=="numberOfBins") 
	{
		inputMapInt[k] = v;
	}
	else
	{
		std::cout << "Key should be numberOfBins" << std::endl;
		throw std::exception();
	}
}

void STE_ReadInput::setMethodIBS( std::string k , int v)
{
	if (k=="methodIBS") 
	{
		inputMapInt[k] = v;
	}
	else
	{
		std::cout << "Key should be methodIBS" << std::endl;
		throw std::exception();
	}
}

void STE_ReadInput::setIBSupdate( std::string k , int v)
{
	if (k=="IBSupdate") 
	{
		inputMapInt[k] = v;
	}
	else
	{
		std::cout << "Key should be IBSupdate" << std::endl;
		throw std::exception();
	}
}


void STE_ReadInput::setRFupdate( std::string k , int v)
{
	if (k=="RFupdate") 
	{
		inputMapInt[k] = v;
	}
	else
	{
		std::cout << "Key should be RFupdate" << std::endl;
		throw std::exception();
	}
}

void STE_ReadInput::setRFmix( std::string k , double v)
{
	if (k=="RFmix") 
	{
		inputMapdouble[k] = v;
	}
	else
	{
		std::cout << "Key should be RFmix" << std::endl;
		throw std::exception();
	}
}


void STE_ReadInput::setnwrite( std::string k , int v)
{
	if (k=="nwrite") 
	{
		inputMapInt[k] = v;
	}
	else
	{
		std::cout << "Key should be nwrite" << std::endl;
		throw std::exception();
	}
}

void STE_ReadInput::setfmohlNumPoints( std::string k , int v)
{
	if (k=="fmohlNumPoints") 
	{
		inputMapInt[k] = v;
	}
	else
	{
		std::cout << "Key should be fmohlNumPoints" << std::endl;
		throw std::exception();
	}
}


void STE_ReadInput::setcoulomblog( std::string k , double v)
{
	if (k=="coulomblog") 
	{
		inputMapdouble[k] = v;
	}
	else
	{
		std::cout << "Key should be coulomblog" << std::endl;
		throw std::exception();
	}
}

void STE_ReadInput::setibsCoupling( std::string k , double v)
{
	if (k=="ibsCoupling") 
	{
		inputMapdouble[k] = v;
	}
	else
	{
		std::cout << "Key should be coulomblog" << std::endl;
		throw std::exception();
	}
}

void STE_ReadInput::setibsfrac( std::string k , double v)
{
	if (k=="fracibstot") 
	{
		if (v == 0.0)
			v = 1.0;
		inputMapdouble[k] = v;
	}
	else
	{
		std::cout << "Key should be coulomblog" << std::endl;
		throw std::exception();
	}
}


void STE_ReadInput::setblowupx( std::string k , double v)
{
	if (k=="blowupx") 
	{
		inputMapdouble[k] = v;
	}
	else
	{
		std::cout << "Key should be blowupx" << std::endl;
		throw std::exception();
	}
}

void STE_ReadInput::setblowupy( std::string k , double v)
{
	if (k=="blowupy") 
	{
		inputMapdouble[k] = v;
	}
	else
	{
		std::cout << "Key should be blowupy" << std::endl;
		throw std::exception();
	}
}

/* ***************************
** Start of getting functions
** ***************************
**/ 
double STE_ReadInput::getRFVoltageV1()
{
	return inputMapdouble["RFVoltage1"];
}

double STE_ReadInput::getRFVoltageV2()
{
	return inputMapdouble["RFVoltage2"];
}

double STE_ReadInput::getRFVoltageV3()
{
	return inputMapdouble["RFVoltage3"];
}


double STE_ReadInput::getHarmonicNumber1()
{
	return inputMapdouble["HarmonicNumber1"];
}

double STE_ReadInput::getHarmonicNumber2()
{
	return inputMapdouble["HarmonicNumber2"];
}

double STE_ReadInput::getHarmonicNumber3()
{
	return inputMapdouble["HarmonicNumber3"];
}


// function for getting number of macro particles
int STE_ReadInput::getNumberOfMacroParticles()
{
	return inputMapInt["numberOfMacroParticles"];
}
// end of function for getting number of macro particles

// function for getting random seed
int STE_ReadInput::getRandomSeed()
{
	return inputMapInt["seed"];
}
// end function getting random seed

// function to get radiation integrals method
int STE_ReadInput::getMethodRadiationIntegrals()
{
	return inputMapInt["methodRadiationIntegral"];
}
// end of function for getting method for calculating radiation integrals

// function to get radiation integrals method
int STE_ReadInput::getMethodLongitudinalDistribution()
{
	return inputMapInt["methodLongitudinalDist"];
}
// end of function for getting method for calculating radiation integrals


// function to get dipole bending radius
double STE_ReadInput::getDipoleBendingRadius()
{
	return inputMapdouble["dipoleBendingRadius"];
}
// end function get dipole bending radius

std::string STE_ReadInput::getTfsFile1()
{
	return inputMapString["tfsFile1"];
}

std::string STE_ReadInput::getTfsFile2()
{
	return inputMapString["tfsFile2"];
}

std::string STE_ReadInput::getBunchFileName1()
{
	return inputMapString["bunchFile1"];
}

std::string STE_ReadInput::getBunchFileName2()
{
	return inputMapString["bunchFile2"];
}

int STE_ReadInput::getTimeRatio()
{
	return inputMapInt["timeRatio"];
}

std::string STE_ReadInput::getParticleType1()
{
	return inputMapString["particleType1"];
}

std::string STE_ReadInput::getParticleType2()
{
	return inputMapString["particleType2"];
}

int STE_ReadInput::getParticleAtomNumber1()
{
	return inputMapInt["particleAtomNumber1"];
}

int STE_ReadInput::getParticleAtomNumber2()
{
	return inputMapInt["particleAtomNumber2"];
}

int STE_ReadInput::getNumberOfTurns()
{
	return inputMapInt["numberOfTurns"];
}

int STE_ReadInput::getWriteDistribution()
{
	return inputMapInt["writeDistribution"];
}

int STE_ReadInput::getWriteTurn()
{
	return inputMapInt["writeTurn"];
}

int STE_ReadInput::getRadiationDamping()
{
	return inputMapInt["radiationDamping"];
}

int STE_ReadInput::getBetatron()
{
	return inputMapInt["betatron"];
}

double STE_ReadInput::getBetatronCoupling()
{
	return inputMapdouble["betatronCoupling"];
}

double STE_ReadInput::getk2l()
{
	return inputMapdouble["k2l"];
}

double STE_ReadInput::getk2sl()
{
	return inputMapdouble["k2sl"];
}


int STE_ReadInput::getNumberofBins()
{
	return inputMapInt["numberOfBins"];
}


int STE_ReadInput::getMethodIBS()
{
	return inputMapInt["methodIBS"];
}

int STE_ReadInput::getIBSupdate()
{
	return inputMapInt["IBSupdate"];
}

int STE_ReadInput::getRFupdate()
{
	return inputMapInt["RFupdate"];
}

double STE_ReadInput::getRFmix()
{
	return inputMapdouble["RFmix"];
}

int STE_ReadInput::getnwrite()
{
	return inputMapInt["nwrite"];
}

int STE_ReadInput::getfmohlNumPoints()
{
	return inputMapInt["fmohlNumPoints"];
}

double STE_ReadInput::getcoulomblog()
{
	return inputMapdouble["coulomblog"];
}

double STE_ReadInput::getibsCoupling()
{
	return inputMapdouble["ibsCoupling"];
}

double STE_ReadInput::getibsfrac()
{
	return inputMapdouble["fracibstot"];
}

double STE_ReadInput::getblowupx()
{
	return inputMapdouble["blowupx"];
}

double STE_ReadInput::getblowupy()
{
	return inputMapdouble["blowupy"];
}

// function to read file line by line and return a map
void STE_ReadInput::ReadInput()
{
	std::string line;
	std::ifstream inputfile;
	std::string key;

	int valueInt;
	double valuedouble;
	std::string valueString;

	// open file stream
	try {
		inputfile.open(filename.c_str());
	}
	catch(std::exception const& e) {
		std::cout << filename << ": " << e.what() << "\n";
	};

	// start of reading file
	if (inputfile.good())
	{
		// read number of macro particles
		getline(inputfile, line);
		std::stringstream iss(line);

		if ( iss >> key >> valueInt )
		{
			setNumberOfMacroParticles(key, valueInt);
		}
		else
		{
			std::cout << "Error reading number of macro particles" << std::endl;
			throw std::exception();
		}

		// getting random seed
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		if (iss >> key >> valueInt)
		{
			setRandomSeed(key, valueInt);
		}
		else
		{
			std::cout << "Error reading random seed." << std::endl;
			throw std::exception();
		}


		// get time ratio = real machine turns per simulation turn
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		if (iss >> key >> valueInt)
		{
			setTimeRatio(key, valueInt);
		}
		else
		{
			std::cout << "Error reading time ratio." << std::endl;
			throw std::exception();
		}

		// read the number of simulation turns to perform
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		if (iss >> key >> valueInt)
		{
			setNumberOfTurns(key, valueInt);
		}
		else
		{
			std::cout << "Error reading number of simulation turns." << std::endl;
			throw std::exception();
		}

		// read if distributions needs to be written to files - 0 : no 1 : yes 
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		if (iss >> key >> valueInt)
		{
			setWriteDistribution(key, valueInt);
		}
		else
		{
			std::cout << "Error reading if distributions need to be written." << std::endl;
			throw std::exception();
		}

		// if write distributions - define when needs to be written
		// 1 -> every turn
		// 10 -> every multiple of 10 turns
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		if (iss >> key >> valueInt)
		{
			setWriteTurn(key, valueInt);
		}
		else
		{
			std::cout << "Error reading write turns." << std::endl;
			throw std::exception();
		}


		// read method  for calculating radiation integrals
		// 1 : approx
		// 2 : lattice from tgs
		// read number of macro particles
		iss.clear();
		getline(inputfile, line);
		// iss << line;
		iss.str(line);
		
		if ( iss >> key >> valueInt )
		{
			// std::cout << key << valueInt << std::endl;
			setMethodRadiationIntegrals(key, valueInt);
		}
		else
		{
			std::cout << "Error reading method for calculating radiation integrals" << std::endl;
			throw std::exception();
		}

		// read method for generating longitudinal distribution
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		
		if ( iss >> key >> valueInt )
		{
			// std::cout << key << valueInt << std::endl;
			setMethodLongitudinalDistribution(key, valueInt);
		}
		else
		{
			std::cout << "Error reading method for generating longitudinal distribution." << std::endl;
			throw std::exception();
		}

		// read dipole bending radius
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		
		if ( iss >> key >> valuedouble )
		{
			// std::cout << key << valueInt << std::endl;
			setDipoleBendingRadius(key, valuedouble);
		}
		else
		{
			std::cout << "Error reading dipole bending radius." << std::endl;
			throw std::exception();
		}

		// read rf voltage 1
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		
		if ( iss >> key >> valuedouble )
		{
			// std::cout << key << valueInt << std::endl;
			setRFVoltageV1(key, valuedouble);
		}
		else
		{
			std::cout << "Error reading RFVoltageV1." << std::endl;
			throw std::exception();
		}

		// read rf voltage 2
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		
		if ( iss >> key >> valuedouble )
		{
			// std::cout << key << valueInt << std::endl;
			setRFVoltageV2(key, valuedouble);
		}
		else
		{
			std::cout << "Error reading RFVoltageV2." << std::endl;
			throw std::exception();
		}

		// read rf voltage 3
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		
		if ( iss >> key >> valuedouble )
		{
			// std::cout << key << valueInt << std::endl;
			setRFVoltageV3(key, valuedouble);
		}
		else
		{
			std::cout << "Error reading RFVoltageV3." << std::endl;
			throw std::exception();
		}

		// read rf harmonic number  1
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		
		if ( iss >> key >> valuedouble )
		{
			// std::cout << key << valueInt << std::endl;
			setHarmonicNumber1(key, valuedouble);
		}
		else
		{
			std::cout << "Error reading HarmonicNumber1." << std::endl;
			throw std::exception();
		}

		// read rf harmonic number  2
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		
		if ( iss >> key >> valuedouble )
		{
			// std::cout << key << valueInt << std::endl;
			setHarmonicNumber2(key, valuedouble);
		}
		else
		{
			std::cout << "Error reading HarmonicNumber2." << std::endl;
			throw std::exception();
		}

		// read rf harmonic number  3
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		
		if ( iss >> key >> valuedouble )
		{
			// std::cout << key << valueInt << std::endl;
			setHarmonicNumber3(key, valuedouble);
		}
		else
		{
			std::cout << "Error reading HarmonicNumber3." << std::endl;
			throw std::exception();
		}

		// read tfsfile 1
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		
		if ( iss >> key >> valueString )
		{
			// std::cout << key << valueInt << std::endl;
			setTfsFile1(key, valueString);
		}
		else
		{
			std::cout << "Error reading tfsFile1." << std::endl;
			throw std::exception();
		}

		// read tfsfile 2
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		
		if ( iss >> key >> valueString )
		{
			// std::cout << key << valueInt << std::endl;
			setTfsFile2(key, valueString);
		}
		else
		{
			std::cout << "Error reading tfsFile2." << std::endl;
			throw std::exception();
		}

		// read bunchfilename 1
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		
		if ( iss >> key >> valueString )
		{
			// std::cout << key << valueInt << std::endl;
			setBunchFileName1(key, valueString);
		}
		else
		{
			std::cout << "Error reading bunchFile1." << std::endl;
			throw std::exception();
		}

		// read bunchfilename 2
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		
		if ( iss >> key >> valueString )
		{
			// std::cout << key << valueInt << std::endl;
			setBunchFileName2(key, valueString);
		}
		else
		{
			std::cout << "Error reading bunchFile2." << std::endl;
			throw std::exception();
		}

		// read particle type1
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		
		if ( iss >> key >> valueString )
		{
			// std::cout << key << valueInt << std::endl;
			setParticleType1(key, valueString);
		}
		else
		{
			std::cout << "Error reading particle type1." << std::endl;
			throw std::exception();
		}

		// read particle type2
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		
		if ( iss >> key >> valueString )
		{
			// std::cout << key << valueInt << std::endl;
			setParticleType2(key, valueString);
		}
		else
		{
			std::cout << "Error reading particle type2." << std::endl;
			throw std::exception();
		}

		// read particle atom number 1
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		
		if ( iss >> key >> valueInt )
		{
			// std::cout << key << valueInt << std::endl;
			setParticleAtomNumber1(key, valueInt);
		}
		else
		{
			std::cout << "Error reading particle atom number 1." << std::endl;
			throw std::exception();
		}

		// read particle atom number 2
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		
		if ( iss >> key >> valueInt )
		{
			// std::cout << key << valueInt << std::endl;
			setParticleAtomNumber2(key, valueInt);
		}
		else
		{
			std::cout << "Error reading particle atom number 2." << std::endl;
			throw std::exception();
		}

		

		// get betatron coupling factor
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		
		if ( iss >> key >> valuedouble )
		{
			// std::cout << key << valueInt << std::endl;
			setBetatronCoupling(key, valuedouble);
		}
		else
		{
			std::cout << "Error reading betatron coupling." << std::endl;
			throw std::exception();
		}

		// get k2l
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		
		if ( iss >> key >> valuedouble )
		{
			// std::cout << key << valueInt << std::endl;
			setk2l(key, valuedouble);
		}
		else
		{
			std::cout << "Error reading k2l." << std::endl;
			throw std::exception();
		}

		// get k2sl
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		
		if ( iss >> key >> valuedouble )
		{
			// std::cout << key << valueInt << std::endl;
			setk2sl(key, valuedouble);
		}
		else
		{
			std::cout << "Error reading k2sl." << std::endl;
			throw std::exception();
		}

		// read activation radiation damping routine
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		
		if ( iss >> key >> valueInt )
		{
			// std::cout << key << valueInt << std::endl;
			setRadiationDamping(key, valueInt);
		}
		else
		{
			std::cout << "Error reading activation radiation damping routine." << std::endl;
			throw std::exception();
		}

		// read activation betatron routine
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		
		if ( iss >> key >> valueInt )
		{
			// std::cout << key << valueInt << std::endl;
			setBetatron(key, valueInt);
		}
		else
		{
			std::cout << "Error reading activation betatron routine." << std::endl;
			throw std::exception();
		}
		////////////////////////////////////////////////////////////////////////////////////////////////////////

		// read anumber of bins
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		
		if ( iss >> key >> valueInt )
		{
			// std::cout << key << valueInt << std::endl;
			setNumberOfBins(key, valueInt);
		}
		else
		{
			std::cout << "Error reading  number of bins." << std::endl;
			throw std::exception();
		}

		// read number of method ibs
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		
		if ( iss >> key >> valueInt )
		{
			// std::cout << key << valueInt << std::endl;
			setMethodIBS(key, valueInt);
		}
		else
		{
			std::cout << "Error reading  IBS method." << std::endl;
			throw std::exception();
		}


		// get ibs coupling strength
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		
		if ( iss >> key >> valuedouble )
		{
			// std::cout << key << valueInt << std::endl;
			setibsCoupling(key, valuedouble);
		}
		else
		{
			std::cout << "Error reading IBS coupling." << std::endl;
			throw std::exception();
		}


		// get ibs fraction
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		
		if ( iss >> key >> valuedouble )
		{
			// std::cout << key << valueInt << std::endl;
			setibsfrac(key, valuedouble);
		}
		else
		{
			std::cout << "Error reading fraction strength IBS." << std::endl;
			throw std::exception();
		}






		// read activation ibs method
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		
		if ( iss >> key >> valueInt )
		{
			// std::cout << key << valueInt << std::endl;
			setIBSupdate(key, valueInt);
		}
		else
		{
			std::cout << "Error reading activation IBS routine." << std::endl;
			throw std::exception();
		}

		// read activation rf method
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		
		if ( iss >> key >> valueInt )
		{
			// std::cout << key << valueInt << std::endl;
			setRFupdate(key, valueInt);
		}
		else
		{
			std::cout << "Error reading activation RF method." << std::endl;
			throw std::exception();
		}

		// get RF mix factor
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		
		if ( iss >> key >> valuedouble )
		{
			// std::cout << key << valueInt << std::endl;
			setRFmix(key, valuedouble);
		}
		else
		{
			std::cout << "Error reading RF mix factor." << std::endl;
			throw std::exception();
		}


		// read nwrite every nth turn is written to output
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		
		if ( iss >> key >> valueInt )
		{
			// std::cout << key << valueInt << std::endl;
			setnwrite(key, valueInt);
		}
		else
		{
			std::cout << "Error reading nwrite." << std::endl;
			throw std::exception();
		}


		// read fmohl numpoints
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		
		if ( iss >> key >> valueInt )
		{
			// std::cout << key << valueInt << std::endl;
			setfmohlNumPoints(key, valueInt);
		}
		else
		{
			std::cout << "Error reading number of fmohl numpoints ." << std::endl;
			throw std::exception();
		}


		// get coulomblog
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		
		if ( iss >> key >> valuedouble )
		{
			// std::cout << key << valueInt << std::endl;
			setcoulomblog(key, valuedouble);
		}
		else
		{
			std::cout << "Error reading coulomblog." << std::endl;
			throw std::exception();
		}


		// get blowupx
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		
		if ( iss >> key >> valuedouble )
		{
			// std::cout << key << valueInt << std::endl;
			setblowupx(key, valuedouble);
		}
		else
		{
			std::cout << "Error reading blowupx." << std::endl;
			throw std::exception();
		}

		// get blowupy
		iss.clear();
		getline(inputfile, line);
		iss.str(line);
		
		if ( iss >> key >> valuedouble )
		{
			// std::cout << key << valueInt << std::endl;
			setblowupy(key, valuedouble);
		}
		else
		{
			std::cout << "Error reading blowupy." << std::endl;
			throw std::exception();
		}



	}
	
}

void STE_ReadInput::PrintInput()
{
	// printing the int maps
	for(std::map<std::string,int>::iterator mii = inputMapInt.begin(); mii != inputMapInt.end(); mii++)
	{
		std::cout << mii->first << " : " << mii->second << std::endl;
	}

	// printing the double maps
	for(std::map<std::string,double>::iterator mif = inputMapdouble.begin(); mif != inputMapdouble.end(); mif++)
	{
		std::cout << mif->first << " : " << mif->second << std::endl;
	}

	// printing the string maps
	for(std::map<std::string,std::string>::iterator mis = inputMapString.begin(); mis != inputMapString.end(); mis++)
	{
		std::cout << mis->first << " : " << mis->second << std::endl;
	}
}