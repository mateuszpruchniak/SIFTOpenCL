

#include "MeanFilter.h"



MeanFilter::~MeanFilter(void)
{
}

MeanFilter::MeanFilter(): GPUBase("C:\\Users\\Mati\\Desktop\\Dropbox\\MGR\\SIFTOpenCL\\GPU\\OpenCL\\BlurGaussFilter.cl","ckConv")
{

}

bool MeanFilter::Process(float sigma, int imageWidth, int imageHeight, int OffsetAct, int OffsetPrev)
{

	OffsetAct = OffsetAct / 4;
	OffsetPrev = OffsetPrev / 4;

	size_t GPULocalWorkSize[2];
	GPULocalWorkSize[0] = iBlockDimX;
	GPULocalWorkSize[1] = iBlockDimY;
	GPUGlobalWorkSize[0] = shrRoundUp((int)GPULocalWorkSize[0], (int)imageWidth);
	GPUGlobalWorkSize[1] = shrRoundUp((int)GPULocalWorkSize[1], (int)imageHeight);
	
	int iLocalPixPitch = iBlockDimX + 2;
	GPUError = clSetKernelArg(GPUKernel, 0, sizeof(cl_mem), (void*)&cmBufPyramid);
	GPUError |= clSetKernelArg(GPUKernel, 1, sizeof(cl_uint), (void*)&OffsetAct);
	GPUError |= clSetKernelArg(GPUKernel, 2, sizeof(cl_uint), (void*)&OffsetPrev);
	GPUError |= clSetKernelArg(GPUKernel, 3, sizeof(cl_uint), (void*)&imageWidth);
	GPUError |= clSetKernelArg(GPUKernel, 4, sizeof(cl_uint), (void*)&imageHeight);
	GPUError |= clSetKernelArg(GPUKernel, 5, sizeof(cl_float), (void*)&sigma);

	if(GPUError) return false;

	if(clEnqueueNDRangeKernel( GPUCommandQueue, GPUKernel, 2, NULL, GPUGlobalWorkSize, GPULocalWorkSize, 0, NULL, NULL)) return false;
	return true;
}


bool MeanFilter::SendImageToBufPyramid( IplImage* img, int offset)
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

bool MeanFilter::CreateBuffer( float size )
{
	cmBufPyramid = clCreateBuffer(GPUContext, CL_MEM_READ_WRITE, size, NULL, &GPUError);
	CheckError(GPUError);
	return true;
}



bool MeanFilter::ReceiveImageToBufPyramid( IplImage* img, int offset)
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
