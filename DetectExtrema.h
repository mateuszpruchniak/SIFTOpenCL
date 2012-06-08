

#pragma once

#include "GPUBase.h"


class DetectExtrema :
	public GPUBase
{
private:

	cl_mem cmDevBufNumber;

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
	
	
	bool Process(cl_mem dogPyr, cl_mem gaussPyr, int imageWidth, int imageHeight, int OffsetPrev, int OffsetAct, int OffsetNext,int* numExtr, float prelim_contr_thr, int intvl, int octv, Keys* keys);
};

