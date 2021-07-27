// include guard
#ifndef TFS_H_INCLUDED
#define TFS_H_INCLUDED

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

// simulation relevant columns
struct tfsTableData
{
	float s;
	float l;
	float px;
	float py;
	float betx;
	float bety;
	float dx;
	float dy;
	float dpx;
	float dpy;
	float angle;
	float k1l;
	float alfx;
	float alfy;
	float k1sl;
};



class STE_TFS
{
public:
	const static std::string tfsColumnsNeededList[];


	STE_TFS( std::string, bool);

	// print header data
	void printHeader();

	// print first n rows of table
	void printTableN( int );

	// get the table data
	std::vector<std::vector<float> > getTFSTableData();

	// load the tfs table data in a device vector for parallel use
	thrust::device_vector<tfsTableData> LoadTFSTableToDevice( std::vector<std::vector<float> > );

	// get the tfs header data as a map
	std::map<std::string, float> getTFSHeader();

	// ~STE_TFS();
	
private:
	void Init( bool );
	void setFilename( std::string );

	// functions to read and clean/transform the data to the desired format
	std::vector<std::vector<std::string> > ReadTfsTable( std::string, bool );
	std::vector<std::vector<float> > TfsTableConvertStringToFloat( std::vector<std::vector<std::string> > , std::map<std::string,int> );
	std::map<std::string, int> mapOfColumns( std::vector<std::string>, bool );
	std::map<std::string, float> ReadTfsHeader( std::string , bool );
	

	// actual data contained in the class
	std::string filename;
	std::vector<std::vector<float> > TFSTableData;
	std::map<std::string, float> TFSHeaderData;

	
};

#endif