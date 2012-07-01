

#pragma once

#include "PyramidProcess.h"


class GaussFilter :
	public PyramidProcess
{

	int GetGaussKernelSize(double sigma, double cut_off=0.001);

public:

	/*!
	* Destructor.
	*/
	~GaussFilter(void);

	/*!
	* Constructor.
	*/
	GaussFilter();

	bool Process(float sigma, int imageWidth, int imageHeight, int OffsetAct, int OffsetNext);

	

};

