
#pragma once

#include "PyramidProcess.h"


class Subtract :
	public PyramidProcess
{
public:

	bool Process(cl_mem gaussPyr, int imageWidth, int imageHeight, int OffsetPrev, int OffsetAct);

	/*!
	* Destructor.
	*/
	~Subtract(void);

	/*!
	* Constructor.
	*/
	Subtract();
};
