#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include "random.h"
#include <iostream>

using std::cout;

// using namespace std;

// thrust particle
typedef thrust::tuple<float,float> vec2;
typedef thrust::tuple<float,float,float,float> vec4;
typedef thrust::tuple<float,float,float,float,float,float> vec6;

struct rand_vec4_bigaussian
{
  __host__ __device__
  vec4 operator()(float betax, float emix, float betay, float emiy, uint seed)
  {
    float ampx, ampy, amp, r1, r2, facc;
    float x,px, y, py;

    // 1 sigma rms beam sizes using average ring betas 
    ampx = sqrt(betax*emix);
    ampy = sqrt(betay*emiy);

    // generate bi-gaussian distribution in the x-px phase-space
    do 
    {
        r1 = 2*ran3(seed)-1;
        r2 = 2*ran3(seed)-1;
        amp = r1*r1+r2*r2;
    } 
    while ((amp >=1) || (amp<=3.e-6));

    facc = sqrt(-2*log(amp)/amp); // transforming [-1,1] uniform to gaussian - inverse transform

    x  = ampx * r1 * facc; // scaling the gaussian
    px = ampx * r2 * facc; // scaling the gaussian

    // generate bi-gaussian distribution in the y-py phase-space
    do 
    {
        r1 = 2*ran3(seed)-1;
        r2 = 2*ran3(seed)-1;
        amp = r1*r1+r2*r2;
    } 
    while ((amp >=1) || (amp<=3.e-6));

    facc = sqrt(-2*log(amp)/amp); // transforming [-1,1] uniform to gaussian - inverse transform

    y = ampy* r1 * facc;  // scaling the gaussian
    py = ampy* r2 * facc; // scaling the gaussian

    return vec4(x,px,y,py);
  }
};
// function for generating bigaussian in transverse plains
vec4 makeRandomTransverseBiGaussian(float betax, float emix, float betay, float emiy, uint seed)
{
	float ampx, ampy, amp, r1, r2, facc;
	float x,px, y, py;

	// 1 sigma rms beam sizes using average ring betas 
	ampx = sqrt(betax*emix);
  ampy = sqrt(betay*emiy);

  // generate bi-gaussian distribution in the x-px phase-space
  do 
	{
      r1 = 2*ran3(seed)-1;
      r2 = 2*ran3(seed)-1;
      amp = r1*r1+r2*r2;
  } 
  while ((amp >=1) || (amp<=3.e-6));

  facc = sqrt(-2*log(amp)/amp); // transforming [-1,1] uniform to gaussian - inverse transform

  x  = ampx * r1 * facc; // scaling the gaussian
  px = ampx * r2 * facc; // scaling the gaussian

  // generate bi-gaussian distribution in the y-py phase-space
  do 
	{
      r1 = 2*ran3(seed)-1;
      r2 = 2*ran3(seed)-1;
      amp = r1*r1+r2*r2;
  } 
  while ((amp >=1) || (amp<=3.e-6));

  facc = sqrt(-2*log(amp)/amp); // transforming [-1,1] uniform to gaussian - inverse transform

  y = ampy* r1 * facc;  // scaling the gaussian
  py = ampy* r2 * facc; // scaling the gaussian

  return vec4(x,px,y,py);
}


int main(int argc, char const *argv[])
{

  thrust::device_vector<vec4> d_particle(10);
  
  thrust::device_vector<int> seed(10);
  thrust::device_vector<float> betax(10);
  thrust::device_vector<float> betay(10);
  thrust::device_vector<float> emix(10);
  thrust::device_vector<float> emiy(10);

  thrust::fill(seed.begin(),seed.end(),12489);
  thrust::fill(betax.begin(),betax.end(),10.);
  thrust::fill(betay.begin(),betay.end(),20.);
  thrust::fill(emix.begin(),emix.end(),1e-9);
  thrust::fill(emiy.begin(),emiy.end(),2e-9);

  thrust::copy(seed.begin(), seed.end(), std::ostream_iterator<int>(std::cout,"\n"));

  // thrust::host_vector<vec4> h_vertical(10);
  // thrust::generate(h_vertical.begin(),h_vertical.end(),makeRandomTransverseBiGaussian(10, 1e-9, 20,2e-9,12489));
  // for (int i=0;i<10;i++)
  // {
  //   std::cout << thrust::get<0>(h_vertical[i]) << endl;
  // }

  return 0;
}