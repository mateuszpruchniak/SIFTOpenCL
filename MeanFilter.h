

#pragma once

#include "GPUBase.h"


class MeanFilter :
	public GPUBase
{

	cl_mem cmBufPyramid;


public:

	/*!
	* Destructor.
	*/
	~MeanFilter(void);

	/*!
	* Constructor.
	*/
	MeanFilter();
	
	bool Process(float sigma, int imageWidth, int imageHeight, int OffsetAct, int OffsetPrev);

	bool CreateBuffer( float size );

	bool SendImageToBufPyramid( IplImage* img, int offset, int* sizeOfImgOct);

	bool ReceiveImageToBufPyramid( IplImage* img, int offset, int* sizeOfImgOct);
};

