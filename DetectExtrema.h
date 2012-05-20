

#pragma once

#include "GPUBase.h"


class DetectExtrema :
	public GPUBase
{
private:

	cl_mem cmDevBufNumber;

	cl_mem cmDevBufNumberReject;

	int maxNumberKeys;

	cl_mem cmDevBufKeys;

	cl_mem cmDevBufCount;

	int counter;

	//Keys* keys;

	cl_kernel GPUKernelDesc;



public:

	/*!
	* Destructor.
	*/
	~DetectExtrema(void);

	/*!
	* Constructor.
	*/
	DetectExtrema(int _maxNumberKeys);
	
	
	bool Process( int* num, int* numRej, float prelim_contr_thr, int intvl, int octv, IplImage* img, Keys* keys);
};

