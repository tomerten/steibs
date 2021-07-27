// include guard
#ifndef SYNCHROTRON_H_INCLUDED
#define SYNCHROTRON_H_INCLUDED

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
#include <boost/math/tools/roots.hpp>
#include <thrust/tuple.h>

#include <iterator>
#include <sstream>
#include <fstream>
#include <algorithm>

// load double 6 datastructure
#include "STE_DataStructures_double.cuh"

// load tfsTableData datastructure
#include "STE_TFS_double.cuh"

struct synchrotronParameters
{
	double energyLostPerTurn;
	double particleCharge;
	double search;
	double searchWidth;
	double omega0;
	double eta;
	double p0;


	double3 harmonicNumbers;
	double3 voltages;
};

template <class T>
struct synchronousPhaseFunctor
{
   synchronousPhaseFunctor(T const& target,double3 voltages, double3 harmonicNumbers, double charge) : U0(target),volts(voltages), hs(harmonicNumbers), ch(charge) {}
   __host__ __device__ thrust::tuple<T, T> operator()(T const& phi)
   {
   	T volt1, volt2, volt3;
   	T dvolt1, dvolt2, dvolt3;
	volt1 =  ch * volts.x * sin(phi);
	volt2 =  ch * volts.y * sin((hs.y / hs.x) * phi);
	volt3 =  ch * volts.z * sin((hs.z / hs.x) * phi);

	dvolt1 =  ch * volts.x * cos(phi);
	dvolt2 =  ch * volts.y * (hs.y / hs.x) * cos((hs.y / hs.x) * phi);
	dvolt3 =  ch * volts.z * (hs.z / hs.x) * cos((hs.z / hs.x) * phi);
    return thrust::make_tuple(volt1 + volt2 + volt3  - U0, dvolt1 + dvolt2 + dvolt3);
   }
private:
   T U0;
   double3 volts;
   double3 hs;
   double ch;
};

template <class T>
T synchronousPhaseFunctorDeriv(T x,double3 voltages, double3 harmnumbers, double charge, T guess, T min, T max)
{ 
  // return cube root of x using 1st derivative and Newton_Raphson.
  using namespace boost::math::tools;
 
  const int digits = std::numeric_limits<T>::digits;  // Maximum possible binary digits accuracy for type T.
  int get_digits = static_cast<int>(digits * 1.0);    // Accuracy doubles with each step, so stop when we have
                                                      // just over half the digits correct.
  const boost::uintmax_t maxit = 100;
  boost::uintmax_t it = maxit;
  T result = newton_raphson_iterate(synchronousPhaseFunctor<T>(x,voltages,harmnumbers,charge), guess, min, max, get_digits, it);
  return result;
}

class STE_Synchrotron
{
public:
	STE_Synchrotron( synchrotronParameters& );
	// ~STE_Synchrotron();

	void setSynchronousPhase( synchrotronParameters& );
	void setSynchronousPhaseNext ( synchrotronParameters& );
	void setTauhat( synchrotronParameters& );
	void setSynchrotronTune( synchrotronParameters& );

	double getSynchronousPhase();
	double getSynchronousPhaseNext();
	double getTauhat();
	double getSynchrotronTune();

	void printSynchrotron();
private:

	double synchronousPhase;
	double synchronousPhaseNext;
	double tauhat;
	double  synchrotronTune;

};
#endif