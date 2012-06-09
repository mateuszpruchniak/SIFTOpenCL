
#pragma once

#include "GPUBase.h"


class Subtract :
	public GPUBase
{
public:

	cl_mem cmBufPyramid;

	bool CreateBuffer( float size );

	bool ReceiveImageToBufPyramid( IplImage* img, int offset);

	bool Process(cl_mem gaussPyr, int imageWidth, int imageHeight, int OffsetPrev, int OffsetAct);

	bool SendImageToBufPyramid( IplImage* img, int offset);

	/*!
	* Destructor.
	*/
	~Subtract(void);

	/*!
	* Constructor.
	*/
	Subtract();
	
	bool Process();
};
