

#pragma once

#include "GPUBase.h"


class MeanFilter :
	public GPUBase
{

	cl_mem cmBufPyramid;
	cl_mem cmBufPyramidOut;

	cl_mem cmBufSigma;

	cl_mem cmBufimageWidth;

	cl_mem cmBufimageHeight;

public:

	/*!
	* Destructor.
	*/
	~MeanFilter(void);

	/*!
	* Constructor.
	*/
	MeanFilter();
	
	bool Process(float* sigma, int* imageWidth, int* imageHeight,  int octvs, int intvlsSum, int maxWidth, int maxHeight, int nChannels);

	bool CreateBuffer(IplImage*** gauss_pyr, int octvs, int intvls, float* sigma, float size );

	bool SendImageToBufPyramid( IplImage* img, int offset, int* sizeOfImgOct);

	bool ReceiveImageToBufPyramid( IplImage* img, int offset, int* sizeOfImgOct);
};

