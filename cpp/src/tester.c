#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>

#include "ste_global_functions.cu"

using namespace std;

int main(int argc, char const *argv[])
{
	// testing random generator
	cout << "Test of random generator function" << endl;	
	cout << ran3(12489) << endl;

	// testing random generator on device
	cout << "Test of random generator on device" << endl;	
	// create mem space
	thrust::device_vector<float> Y(10);
	// fill with random numbers
	thrust::fill(Y.begin(),Y.end(),ran3(12489));
	// print to screen
	thrust::copy(Y.begin(),Y.end(),std::ostream_iterator<float>(std::cout,"\n"));

	return 0;
}