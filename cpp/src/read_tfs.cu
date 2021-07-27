#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <boost/lexical_cast.hpp>
#include <map>
#include <iostream>
#include <iterator>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
// copy begin() + 1 to device 

using namespace std;

// list of mandatory columns 
const string tfsColumnsNeededList[] = {"S","L","PX","PY","BETX","BETY","DX","DY","DPX","DPY","ANGLE","K1L","ALFX","ALFY","K1SL"}; //ORDER  -> tfsTableDATA!!!!

set<string> tfsColumnsNeeded(tfsColumnsNeededList, tfsColumnsNeededList + sizeof(tfsColumnsNeededList) / sizeof(tfsColumnsNeededList[0]));


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
// structure of arrays - cuda performance !!! ( instead of array of structs)
// struct tfsTableData 
// {
// 	thrust::device_vector<float> s;
// 	thrust::device_vector<float> l;
// 	thrust::device_vector<float> px;
// 	thrust::device_vector<float> py;
// 	thrust::device_vector<float> betx;
// 	thrust::device_vector<float> bety;
// 	thrust::device_vector<float> dx;
// 	thrust::device_vector<float> dy;
// 	thrust::device_vector<float> dpx;
// 	thrust::device_vector<float> dpy;
// 	thrust::device_vector<float> angle;
// 	thrust::device_vector<float> k1l;
// 	thrust::device_vector<float> alfx;
// 	thrust::device_vector<float> alfy;
// };


// function to read twiss table data 
vector<vector<string> > ReadTfsTable(string pathFilename, bool debug){
	vector<vector<string> > out;
	fstream infile(pathFilename.c_str());

	if (infile) 
		{
			string line;
			set<string> colset;

			//skip tfs header
			for (int i=1;i<47;i++)
			{
				getline(infile,line);
			}

			// last line read has the column headers
			// we check if the simulation required ones are present
			istringstream iss(line);
			string colname;
			out.push_back(vector<string>());

			// write the headers to a set 
			while(iss >> colname)
			{
				// skip the * in tfs file as first column
				if (colname!="*")
				{
					colset.insert(colname);
					// write headers to vector for later use
					out.back().push_back(colname);
				};
			}
			
			// check if required set in set of available headers
			if (std::includes(colset.begin(), colset.end(),tfsColumnsNeeded.begin(), tfsColumnsNeeded.end()))
				cout << "Required twiss columns are present : OK !" << endl;
			else
			{
				cout << "Missing one or more required twiss columns : FAIL !" << endl << endl;

				// get the set difference of available columns and required columns
				set<string> diff;
				std::set_difference(tfsColumnsNeeded.begin(), tfsColumnsNeeded.end(), colset.begin(), colset.end(),  std::inserter(diff, diff.begin()));

				// printing missing columns to screen
				set<string>::const_iterator it;
				cout << "List of missing columns : " << endl;
				cout << "--------------------------" << endl;
				for(it=diff.begin(); it != diff.end(); it++)
				{
					cout << *it << endl;
				}
				cout << "--------------------------" << endl;
				exit(1);
				
			}

			getline(infile,line); // read type line tfs and skip it

			while(getline(infile,line))
			{
				
				out.push_back(vector<string>());
				stringstream split(line);
				string value;
				while (split >> value)
				{
					// cout << value << " ";
					out.back().push_back(value);
				};
				// cout << endl;

			};

			// write out for debugging
			if (debug == true)
			{
				for(int j=0;j<10;j++)
					{
					for(int k=0;k<8;k++)
						cout << std::setw(15) << out[j][k] << "\t";
					cout << endl;
					}
			}
		}
	else
		{
			cout << "Reading error" << endl;
			exit(1);
		}; 
	return out;
}

vector<vector<float> > TfsTableConvertStringToFloat(vector<vector<string> > inputTable, map<string,int> requiredColumnMap)
{
	vector<vector<float> > out;

	for(vector<vector<string> >::iterator it = inputTable.begin()+1; it != inputTable.end(); ++it)
	{

		out.push_back(vector<float>());

		for(map<string, int>::iterator jt = requiredColumnMap.begin(); jt != requiredColumnMap.end();jt++)
		{

			float value = boost::lexical_cast<float>((*it)[jt->second]);
			// for debugging
			// if (it == inputTable.begin()+1)
			// 	cout << jt->first<< "," <<jt->second << ":" <<value << endl;
			out.back().push_back(value);
		}	
	}
	return out;

};

thrust::device_vector<tfsTableData> TfsTableToDevice(vector<vector<float> > input)
{
	thrust::device_vector<tfsTableData> dout;
	tfsTableData out;
	// cout << "size is " << input.size() << endl;
	for(int i =0; i<  input.size() ;i++)
	{
		// map is sorted alphabetically on the keys

		out.alfx  = input[i][0];
		out.alfy  = input[i][1];
		out.angle = input[i][2];
		out.betx  = input[i][3];
		out.bety  = input[i][4];
		out.dpx   = input[i][5];
		out.dpy   = input[i][6];
		out.dx    = input[i][7];
		out.dy    = input[i][8];
		out.k1l   = input[i][9];
		out.k1sl  = input[i][10];
		out.l     = input[i][11];
		out.px    = input[i][12];
		out.py    = input[i][13];
		out.s     = input[i][14];

		dout.push_back(out);
	};
	// cout << (input.begin())[0][0] <<endl;
	return dout;
};


map<string, int> mapOfColumns(vector<string> headers,bool debug)
{
	map<string, int> outputmap;
	for(std::vector<string>::iterator it = headers.begin(); it != headers.end(); ++it) {
		if (tfsColumnsNeeded.find(*it) != tfsColumnsNeeded.end())
	    	outputmap.insert(make_pair(*it,it-headers.begin()));
	}

	if (debug == true)
	{
		for(map<string, int>::const_iterator it = outputmap.begin();it != outputmap.end(); ++it)
			{
			    cout << it->first << " " << it->second << "\n";
			}
	}

	return outputmap;
};

// function to read the twiss file headers
map<string, float>  ReadTfsHeader(string pathFilename, bool debug)
{
	// vector<vector<string> > out;
	map<string, float> outputmap;
	fstream infile(pathFilename.c_str());

	if (infile) 
		{
			string line;

			// skip first 4 lines
			for(int i=1;i<=4;i++)
				getline(infile,line);

			
			for (int i=1;i<38;i++)
			{
				getline(infile,line);

				// out.push_back(vector<string>());
				stringstream split(line);
				string key;
				string tmp; 
				string value;

				// skip the @ symbol
				split >> value;

				split >> key;

				// out.back().push_back(value);
				if (debug == true)
						cout << std::setw(15) << key << " ";
					
				// skip the type %le
				split >> value;

				split >> value;
				if (debug == true)
					cout << value << endl;
				outputmap.insert(make_pair(key, boost::lexical_cast<float>(value)));

				// out.back().push_back(value);
				if (debug == true)
						cout << std::setw(15) << value << " ";

				while (split >> value)
				{	
					if (debug == true)
						cout << std::setw(15) << value << " ";
					// out.back().push_back(value);
				};
				if (debug==1)
					cout << endl;
			}

		}
	else
		{
			cout << "Reading error" << endl;
			exit(1);
		}; 
	return outputmap;

};


// int main(int argc, char const *argv[])
// {

// 	vector<vector<string> > out;
// 	vector<vector<float> > fout;
// 	string in = "/home/tmerten/mad/2018-01-03/twiss/2018-01-03-12-11-54-twiss.tfs";
// 	out = ReadTfsTable(in,true);
// 	map<string, int> maptest = mapOfColumns(out[0],true);
// 	fout = TfsTableConvertStringToFloat(out,maptest);
// 	// write out for debugging
// 	// for(int j=0;j<100;j++)
// 	// {
// 	// 	for(int k=0;k<8;k++)
// 	// 		cout << std::setw(15) << fout[j][k] << "\t";
// 	// 	cout << endl;
// 	// };
// 	cout <<  fout.size() << endl;;
// 	thrust::device_vector<tfsTableData> testdata;
// 	testdata = TfsTableToDevice(fout);
// 	// std::copy(testdata.s.begin(),testdata.s.end(),std::ostream_iterator<float>(std::cout, "\n"));
// 	// cout << endl;
// 	// std::copy(testdata.px.begin(),testdata.px.end(),std::ostream_iterator<float>(std::cout, "\n"));
	
// 	// out = ReadTfsHeader(in,true);
// 	// cout << fout.size() << endl;
// 	// thrust::host_vector<vector<float> > hout;
// 	// thrust::copy(fout.begin(),fout.end(),hout.begin());
// 	// thrust::device_vector<vector<float> >test(fout);
// 	// test = fout;

// 	return 0;
// }