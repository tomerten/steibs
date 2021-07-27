// include guard
#ifndef BUNCHFILE_H_INCLUDED
#define BUNCHFILE_H_INCLUDED

// included dependencies
#include <map>
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

#include "STE_DataStructures.cuh"


struct bunchFileRow
{
	long realNumberParticles;
	float ex;
	float ey;
	float sigs;
};



class STE_ReadBunchFiles
{
public:
	STE_ReadBunchFiles( std::string );

	bunchFileRow getBunchRow( int );

	std::map<int, bunchFileRow> getRows();

	void printBunchFile();
	// ~STE_ReadBunchFiles();

private:
	void readFile( std::string );
	std::map<int, bunchFileRow> bunchFileRows;
};

#endif