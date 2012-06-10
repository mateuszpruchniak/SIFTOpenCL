#include "PyramidProcess.h"



PyramidProcess::PyramidProcess(char* source, char* KernelName) : GPUBase(source,KernelName)
{

}


PyramidProcess::~PyramidProcess(void)
{

}

bool PyramidProcess::CreateBufferForPyramid( float size )
{
	cmBufPyramid = clCreateBuffer(GPUContext, CL_MEM_READ_WRITE, size, NULL, &GPUError);
	CheckError(GPUError);
	return true;
}


bool PyramidProcess::ReceiveImageFromPyramid( IplImage* img, int offset)
{
	clock_t start, finish;
	double duration = 0;
	start = clock();
	GPUError = clEnqueueReadBuffer(GPUCommandQueue, cmBufPyramid, CL_TRUE, offset, img->imageSize, (void*)img->imageData, 0, NULL, NULL);
	CheckError(GPUError);
	finish = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	RecvTime += duration;

	return true;
}

bool PyramidProcess::SendImageToPyramid( IplImage* img, int offset)
{
	clock_t start, finish;
	double duration = 0;
	start = clock();
	GPUError = clEnqueueWriteBuffer(GPUCommandQueue, cmBufPyramid, CL_TRUE, offset, img->imageSize, (void*)img->imageData, 0, NULL, NULL);
	CheckError(GPUError);
	finish = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	SendTime += duration;

	return true;
}