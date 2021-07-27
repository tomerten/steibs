#include "STE_TFS.cuh"

const std::string STE_TFS::tfsColumnsNeededList[] = {"S","L","PX","PY","BETX","BETY","DX","DY","DPX","DPY","ANGLE","K1L","ALFX","ALFY","K1SL"}; 
std::set<std::string> tfsColumnsNeeded(STE_TFS::tfsColumnsNeededList, STE_TFS::tfsColumnsNeededList + sizeof(STE_TFS::tfsColumnsNeededList) / sizeof(STE_TFS::tfsColumnsNeededList[0]));


// object constructor
STE_TFS::STE_TFS( std::string fn, bool debug )
{
	setFilename( fn );
	Init( debug );

}

// function to set filename
void STE_TFS::setFilename( std::string fn )
{
	filename = fn;
}

// end function	set filename


// function to read the tfs table as string data - needs to be transformed TfsTableConvertStringToFloat
std::vector<std::vector<std::string> > STE_TFS::ReadTfsTable( std::string pathFilename, bool debug )
{
	// variable for storing the result
	std::vector<std::vector<std::string> > out;

	// creating stream object
	std::fstream infile(pathFilename.c_str());

	// if the input file is ok
	if (infile) 
		{
			// variable to store the line
			std::string line;
			// set for keeping track of the column names
			std::set<std::string> colset;

			//skip tfs header - will be read with separate function
			for (int i=1;i<47;i++)
			{
				getline(infile,line);
			}

			// last line read has the column headers
			// we check if the simulation required ones are present
			std::istringstream iss(line);
			std::string colname;

			out.push_back(std::vector<std::string>());

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
				std::cout << "Required twiss columns are present : OK !" << std::endl;

			else
			{
				std::cout << "Missing one or more required twiss columns : FAIL !" << std::endl << std::endl;

				// get the set difference of available columns and required columns
				std::set<std::string> diff;
				std::set_difference(tfsColumnsNeeded.begin(), tfsColumnsNeeded.end(), colset.begin(), colset.end(),  std::inserter(diff, diff.begin()));

				// printing missing columns to screen
				std::set<std::string>::const_iterator it;

				std::cout << "List of missing columns : " << std::endl;
				std::cout << "--------------------------" << std::endl;

				for(it=diff.begin(); it != diff.end(); it++)
				{
					std::cout << *it << std::endl;
				}

				std::cout << "--------------------------" << std::endl;
				exit(1);
				
			} // endif required columns are available

			getline(infile,line); // read type line tfs and skip it

			// reading the table data line by line
			while(getline(infile,line))
			{
				
				out.push_back(std::vector<std::string>());

				std::stringstream split(line);
				std::string value;

				while (split >> value)
				{
					out.back().push_back(value);
				};

			};

			// write out for debugging
			if (debug == true)
			{
				for(int j=0;j<10;j++)
					{
					for(int k=0;k<8;k++)
						std::cout << std::setw(15) << out[j][k] << "\t";
					std::cout << std::endl;
					}
			}
		}
	else
		{
			std::cout << "Reading error" << std::endl;
			exit(1);
		}; 
	return out;
}
// end of function to read tfs table data as string

// function to convert tfs table data from string format to floats
std::vector<std::vector<float> > STE_TFS::TfsTableConvertStringToFloat(std::vector<std::vector<std::string> > inputTable, std::map<std::string,int> requiredColumnMap)
{
	// variable for returning the result
	std::vector<std::vector<float> > out;

	for(std::vector<std::vector<std::string> >::iterator it = inputTable.begin()+1; it != inputTable.end(); ++it)
	{

		out.push_back(std::vector<float>());

		for(std::map<std::string, int>::iterator jt = requiredColumnMap.begin(); jt != requiredColumnMap.end();jt++)
		{

			float value = boost::lexical_cast<float>((*it)[jt->second]);
			// for debugging
			// if (it == inputTable.begin()+1)
			// 	cout << jt->first<< "," <<jt->second << ":" <<value << endl;
			out.back().push_back(value);
		}	
	}
	return out;

}
// end function convert string table to float table


// create a map of column headers : int for location of columns
std::map<std::string, int> STE_TFS::mapOfColumns( std::vector<std::string> headers, bool debug )
{
	// result variable
	std::map<std::string, int> outputmap;

	// iterate over the headers
	for(std::vector<std::string>::iterator it = headers.begin(); it != headers.end(); ++it) {
		if (tfsColumnsNeeded.find(*it) != tfsColumnsNeeded.end())
	    	outputmap.insert(make_pair(*it,it-headers.begin()));
	}

	if (debug == true)
	{
		for(std::map<std::string, int>::const_iterator it = outputmap.begin();it != outputmap.end(); ++it)
			{
			    std::cout << it->first << " " << it->second << "\n";
			}
	}

	return outputmap;
};
// end of creating map of column headers


// function to read the twiss file headers
std::map<std::string, float>  STE_TFS::ReadTfsHeader(std::string pathFilename, bool debug)
{
	// result variable
	std::map<std::string, float> outputmap;

	std::fstream infile(pathFilename.c_str());

	if (infile) 
		{
			std::string line;

			// skip first 4 lines
			for(int i=1;i<=4;i++)
				getline(infile,line);

			
			for (int i=1;i<38;i++)
			{
				getline(infile,line);

				// out.push_back(vector<string>());
				std::stringstream split(line);
				std::string key;
				std::string tmp; 
				std::string value;

				// skip the @ symbol
				split >> value;

				split >> key;

				// out.back().push_back(value);
				if (debug == true)
						std::cout << std::setw(15) << key << " ";
					
				// skip the type %le
				split >> value;

				split >> value;
				if (debug == true)
					std::cout << value << std::endl;
				outputmap.insert(make_pair(key, boost::lexical_cast<float>(value)));

				// out.back().push_back(value);
				if (debug == true)
						std::cout << std::setw(15) << value << " ";

				while (split >> value)
				{	
					if (debug == true)
						std::cout << std::setw(15) << value << " ";
					// out.back().push_back(value);
				};
				if (debug==1)
					std::cout << std::endl;
			}

		}
	else
		{
			std::cout << "Reading error" << std::endl;
			exit(1);
		}; 
	return outputmap;

};

// function for initializing the object
void STE_TFS::Init( bool debug )
{
	// read table to strings
	std::vector<std::vector<std::string> > table = ReadTfsTable( filename, debug );

	// read headers
	std::map<std::string, int> maptest = mapOfColumns( table[0], debug );

	// transform strings table to float table and select required columns
	TFSTableData = TfsTableConvertStringToFloat(table, maptest);

	// create a map of the header data
	TFSHeaderData = ReadTfsHeader( filename, debug);
}
// end of init function


// function to load table in device vector
thrust::device_vector<tfsTableData> STE_TFS::LoadTFSTableToDevice(std::vector<std::vector<float> > input)
{
	thrust::device_vector<tfsTableData> dout;
	tfsTableData out;

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

	return dout;
};
// end of function to load in device vector

// function to print header
void STE_TFS::printHeader()
{
	std::cout << std::endl;
	std::cout << "********** START TFS HEADER ********" << std::endl;

	for(std::map<std::string,float>::iterator mii = TFSHeaderData.begin(); mii != TFSHeaderData.end(); mii++)
	{
		std::cout << mii->first << " : " << mii->second << std::endl;
	}

	std::cout << "********** END   TFS HEADER ********" << std::endl;
	std::cout << std::endl;
}

// write tfsdata required - helper function
__host__ std::ostream& operator<< (std::ostream& os, const tfsTableData& p)
{
	os << std::setw(12) << p.s << std::setw(12) << p.angle << std::setw(10) << p.l << std::setw(12) << p.k1l << std::setw(15) << p.dx <<std::setw(15) << std::endl;
	return os;
}

void STE_TFS::printTableN( int N )
{
	thrust::device_vector<tfsTableData> inputdata;
	inputdata = LoadTFSTableToDevice(TFSTableData);
	std::cout << std::endl;
	std::cout << "********** START TFS TABLE " << N << " ROWS *******" << std::endl;
	std::cout << std::setw(12) << "S" << std::setw(12) << "ANGLE" << std::setw(10) << "L" << std::setw(12) << "K1L" << std::setw(15) << "DX" << std::setw(15) << std::endl;
	std::copy(inputdata.begin(),inputdata.begin()+N,std::ostream_iterator<tfsTableData>(std::cout));
	std::cout << "********** END   TFS TABLE **************" << std::endl;
	
}

// function to get TFSTable data
std::vector<std::vector<float> > STE_TFS::getTFSTableData()
{
	return TFSTableData;
}
// end functionto get TFSTable data

// function to get tfs header as map
std::map<std::string, float> STE_TFS::getTFSHeader()
{
	return TFSHeaderData;
}
// end function get tfs header asmap