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

#include "STE_DataStructures_double.cuh"

#include "STE_ReadBunchFiles_double.cuh"

STE_ReadBunchFiles::STE_ReadBunchFiles( std::string fn )
{
	readFile( fn );
}

__host__ std::ostream& operator<< (std::ostream& os, const bunchFileRow& p)
{
	os << std::setw(15) << p.realNumberParticles << std::setw(15) << p.ex << std::setw(15) << p.ey 
	<< std::setw(15) << p.sigs << std::endl;
	return os;
};

void STE_ReadBunchFiles::readFile( std::string fn )
{
	std::string line;
	std::ifstream inputfile;
	std::string key;

	bunchFileRow row;
	int bucket;
	double realnumberparticles;
	double emitx;
	double emity;
	double sigs;

	// open file stream
	try {
		inputfile.open(fn.c_str());
	}
	catch(std::exception const& e) {
		std::cout << fn << ": " << e.what() << "\n";
	};

	// start of reading file
	if (inputfile.good())
	{
		getline(inputfile,line); // read header line and skip it

		while(getline(inputfile,line))
			{
				std::stringstream iss(line);
				// std::cout << "row bucket" << iss << std::endl;
				if ( iss >> bucket >> realnumberparticles >> emitx >> emity >> sigs )
				{
					// std::cout << "row bucket " << bucket << std::endl;
					// std::cout << "row number " << realnumberparticles << std::endl;
					row.realNumberParticles = realnumberparticles;
					row.ex   = emitx;
					row.ey   = emity;
					row.sigs = sigs;
					// std::cout << "ex  -- " << emitx << std::endl;
					// std::cout << "ey  -- " << emity << std::endl;
					// std::cout << "sigs  -- " << sigs << std::endl;
					bunchFileRows[bucket] = row;
					// std::cout << "row --" << row << std::endl;
				}
				else
				{
					std::cout << "Error reading row for bucket " << bucket << std::endl;
					throw std::exception();
				}
			};
	};
		
}


void STE_ReadBunchFiles::printBunchFile()
{
	// printing the int maps
	for(std::map<int,bunchFileRow>::iterator mii = bunchFileRows.begin(); mii != bunchFileRows.end(); mii++)
	{
		std::cout << mii->first << " : " << mii->second ;
	}
}

bunchFileRow STE_ReadBunchFiles::getBunchRow( int bucket)
{
	return bunchFileRows[bucket];
}


std::map<int, bunchFileRow> STE_ReadBunchFiles::getRows()
{
	return bunchFileRows;
}