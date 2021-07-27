#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <cstdlib>

// #include "random.h"

#define MBIG 1000000000
#define MSEED 161803398
#define MZ 0
#define FAC (1.0/MBIG)

__host__ __device__ float ran3(int idum)
//int *idum;
{
	static int inext,inextp;
	static long ma[56];
	static int iff=0;
	long mj,mk;
	int i,ii,k;

	if (idum < 0 || iff == 0) {
		iff=1;
		mj=MSEED-(idum < 0 ? -idum : idum);
		mj %= MBIG;
		ma[55]=mj;
		mk=1;
		for (i=1;i<=54;i++) {
			ii=(21*i) % 55;
			ma[ii]=mk;
			mk=mj-mk;
			if (mk < MZ) mk += MBIG;
			mj=ma[ii];
		}
		for (k=1;k<=4;k++)
			for (i=1;i<=55;i++) {
				ma[i] -= ma[1+(i+30) % 55];
				if (ma[i] < MZ) ma[i] += MBIG;
			}
		inext=0;
		inextp=31;
		idum=1;
	}
	if (++inext == 56) inext=1;
	if (++inextp == 56) inextp=1;
	mj=ma[inext]-ma[inextp];
	if (mj < MZ) mj += MBIG;
	ma[inext]=mj;
	return mj*FAC;
}

#undef MBIG
#undef MSEED
#undef MZ
#undef FAC


struct rand_vec4_bigaussian
{
  __host__ __device__
  float4 operator()(float4 a)
  {
    float ampx, ampy, amp, r1, r2, facc;
    float x,px, y, py;
    float betax;
    float emix; 
    float betay; 
    float emiy; 
    uint seed;
    betax = 10.0;
    betay = 20.0;
    emix = 1e-9;
    emiy = 2e-9;
    seed=12489;
    // 1 sigma rms beam sizes using average ring betas 
    ampx = sqrt(betax*emix);
    ampy = sqrt(betay*emiy);
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

    a = make_float4(x,px,y,py);
    
    return a;
	};
};

__host__ std::ostream& operator<< (std::ostream& os, const float4& p)
{
	os << "["<< p.x << "," << p.y << "," << p.z <<"," << p.w <<"]";
	return os;
}

int main(int argc, char const *argv[])
{
	thrust::device_vector<float> Y(10);
	thrust::fill(Y.begin(),Y.end(),ran3(12489));

	rand_vec4_bigaussian func;
	thrust::device_vector<float4> v(1000000);

	thrust::transform(v.begin(),v.end(),v.begin(),func);

	thrust::copy(Y.begin(),Y.end(),std::ostream_iterator<float>(std::cout,"\n"));
	// std::copy(v.begin(), v.end(), std::ostream_iterator<float4>(std::cout, "  $  ") );
	// thrust::copy(v.begin(),v.end(),std::ostream_iterator<float>(std::cout,"\n"));
	std::cout << ran3(12489);
	return 0;
}