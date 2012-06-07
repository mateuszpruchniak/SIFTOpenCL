

#pragma once

#include "GPUBase.h"


class MeanFilter :
	public GPUBase
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

	cl_mem cmBufPyramid;
	
	bool Process(float sigma, int imageWidth, int imageHeight, int OffsetAct, int OffsetNext);

	bool CreateBuffer( float size );

	bool SendImageToBufPyramid( IplImage* img, int offset);

	bool ReceiveImageToBufPyramid( IplImage* img, int offset);
};

