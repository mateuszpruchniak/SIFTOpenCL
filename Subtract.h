
#pragma once

#include "GPUBase.h"


class Subtract :
	public GPUBase
{
public:

	cl_mem cmBufPyramid;

	bool CreateBuffer( float size );

	bool ReceiveImageToBufPyramid( IplImage* img, int offset, int* sizeOfImgOct);

	bool Process(cl_mem gaussPyr, int imageWidth, int imageHeight, int OffsetAct, int OffsetNext);

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
