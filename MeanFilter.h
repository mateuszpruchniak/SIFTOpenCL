

#pragma once

#include "PyramidProcess.h"


class MeanFilter :
	public PyramidProcess
{

public:

	/*!
	* Destructor.
	*/
	~MeanFilter(void);

	/*!
	* Constructor.
	*/
	MeanFilter();

	bool Process(float sigma, int imageWidth, int imageHeight, int OffsetAct, int OffsetNext);
};

