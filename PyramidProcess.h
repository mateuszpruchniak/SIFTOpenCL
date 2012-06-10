#pragma once

#include "GPUBase.h"




class PyramidProcess :
	public GPUBase
{

public:

	cl_mem cmBufPyramid;

	bool CreateBufferForPyramid(float size);

	bool ReceiveImageFromPyramid( IplImage* img, int offset);

	bool SendImageToPyramid(IplImage* img, int offset);

	/*!
	* Destructor.
	*/
	~PyramidProcess(void);

	/*!
	* Constructor.
	*/
	PyramidProcess(char* source, char* KernelName);
};