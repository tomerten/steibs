#include <iostream>
#include <thrust/tuple.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <math.h>
#include <vector>
#include <boost/math/tools/roots.hpp>
#include <curand_kernel.h>
using namespace std;


template <class T>
struct voltage_functor
{
   voltage_functor(T const& target,float3 voltages, float3 harmonicNumbers) : U0(target),volts(voltages), hs(harmonicNumbers) {}
   __host__ __device__ thrust::tuple<T, T> operator()(T const& phi)
   {
   	T volt1, volt2, volt3;
   	T dvolt1, dvolt2, dvolt3;
	volt1 = - volts.x * sin(phi);
	volt2 = - volts.y * sin((hs.y / hs.x) * phi);
	volt3 = - volts.z * sin((hs.z / hs.x) * phi);

	dvolt1 = - volts.x * cos(phi);
	dvolt2 = - volts.y * (hs.y / hs.x) * cos((hs.y / hs.x) * phi);
	dvolt3 = - volts.z * (hs.z / hs.x) * cos((hs.z / hs.x) * phi);
    return thrust::make_tuple(volt1 + volt2 + volt3  - U0, dvolt1 + dvolt2 + dvolt3);
   }
private:
   T U0;
   float3 volts;
   float3 hs;
};

template <class T>
T cbrt_deriv(T x,float3 y, float3 z, T guess, T min, T max)
{ 
  // return cube root of x using 1st derivative and Newton_Raphson.
  using namespace boost::math::tools;
 
  // T guess = 0.0;                    // Rough guess is to divide the exponent by three.
  // T min = -1.0;                     // Minimum possible value is half our guess.
  // T max = 1.0;                      // Maximum possible value is twice our guess.
  const int digits = std::numeric_limits<T>::digits;  // Maximum possible binary digits accuracy for type T.
  int get_digits = static_cast<int>(digits * 0.6);    // Accuracy doubles with each step, so stop when we have
                                                      // just over half the digits correct.
  const boost::uintmax_t maxit = 20;
  boost::uintmax_t it = maxit;
  T result = newton_raphson_iterate(voltage_functor<T>(x,y,z), guess, min, max, get_digits, it);
  return result;
}

class Line{
	float3 volts;
	float3 hs;
	float U0;

public:
	Line(float energylost,float3 voltages, float3 harmonicNumbers):
		 U0(energylost),volts(voltages), hs(harmonicNumbers) {} 

	thrust::tuple<float,float> operator()(float phi){
		float volt1, volt2, volt3;
   		float dvolt1, dvolt2, dvolt3;
		volt1 = - volts.x * sin(phi);
		volt2 = - volts.y * sin((hs.y / hs.x) * phi);
		volt3 = - volts.z * sin((hs.z / hs.x) * phi);

		dvolt1 = - volts.x * cos(phi);
		dvolt2 = - volts.y * (hs.y / hs.x) * cos((hs.y / hs.x) * phi);
		dvolt3 = - volts.z * (hs.z / hs.x) * cos((hs.z / hs.x) * phi);
		return thrust::make_tuple(volt1 + volt2 + volt3  - U0, dvolt1 + dvolt2 + dvolt3);
	}
};

int main () {
	// Line fa;			// y = 1*x + 1
	Line fb(1700000.,make_float3(-1.4e6,-20.0e6,-17.11e6),make_float3(400,1200,1400));		// y = 5*x + 10
	voltage_functor<float> fa(170000.,make_float3(-1.4e6,-20.0e6,-17.14e6),make_float3(400,1200,1400));
	// thrust::tuple<float,float> y1 = fa(20.0,3.0);		// y1 = 20 + 1
	thrust::tuple<float,float> y2 = fb(3.0);		// y2 = 5*3 + 10

	// cout << "y1 = " << thrust::get<0>(y1) << " y2 = " << thrust::get<1>(y1) << endl;
	cout << "y1 = " << thrust::get<0>(y2) << " y2 = " << thrust::get<1>(y2) << endl;
	double phi;
	cout << thrust::get<0>(fa(1.)) << endl;
	double energylost = 170000;

	double r = cbrt_deriv(energylost,make_float3(-1.4e6,-20.0e6,-17.14e6),make_float3(400,1200,1400),0.0,-1.0,1.0);
	double r2 = cbrt_deriv(energylost,make_float3(-1.4e6,-20.0e6,-17.14e6),make_float3(400,1200,1400),-6.0,-7.0,-5.0);
	// double r = boost::math::tools::newton_raphson_iterate(fa(phi),0.0,-0.1,0.2,12);
	cout << r << " , "<<r2 << endl;
	return 0;

}