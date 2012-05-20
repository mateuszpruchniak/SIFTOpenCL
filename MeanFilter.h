

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
	
	bool Process(float sigma);

	bool Process2(IplImage*** gauss_pyr, int octvs, int intvls, float* sigma, float size );

	bool SendImageToBufPyramid( IplImage* img, int offset, int* sizeOfImgOct);
};

