

#include "MeanFilter.h"



MeanFilter::~MeanFilter(void)
{
}

MeanFilter::MeanFilter(): GPUBase("C:\\Users\\Mati\\Desktop\\Dropbox\\MGR\\SIFTOpenCL\\GPU\\OpenCL\\BlurGaussFilter.cl","ckConv")
{

}

bool MeanFilter::Process(float* sigma, int* imageWidth, int* imageHeight,  int octvs, int intvlsSum, int maxWidth, int maxHeight, int nChannels)
{

	cmBufSigma = clCreateBuffer(GPUContext, CL_MEM_READ_WRITE, intvlsSum*sizeof(float), NULL, &GPUError);
	CheckError(GPUError);
	GPUError = clEnqueueWriteBuffer(GPUCommandQueue, cmBufSigma, CL_TRUE, 0, intvlsSum*sizeof(float), (void*)sigma, 0, NULL, NULL);
	CheckError(GPUError);

	cmBufimageWidth = clCreateBuffer(GPUContext, CL_MEM_READ_WRITE, octvs*sizeof(int), NULL, &GPUError);
	CheckError(GPUError);
	GPUError = clEnqueueWriteBuffer(GPUCommandQueue, cmBufimageWidth, CL_TRUE, 0, octvs*sizeof(int), (void*)imageWidth, 0, NULL, NULL);
	CheckError(GPUError);

	cmBufimageHeight = clCreateBuffer(GPUContext, CL_MEM_READ_WRITE, octvs*sizeof(int), NULL, &GPUError);
	CheckError(GPUError);
	GPUError = clEnqueueWriteBuffer(GPUCommandQueue, cmBufimageHeight, CL_TRUE, 0, octvs*sizeof(int), (void*)imageHeight, 0, NULL, NULL);
	CheckError(GPUError);



	size_t GPULocalWorkSize[2];
	GPULocalWorkSize[0] = iBlockDimX;
	GPULocalWorkSize[1] = iBlockDimY;
	GPUGlobalWorkSize[0] = shrRoundUp((int)GPULocalWorkSize[0], (int)maxWidth);
	GPUGlobalWorkSize[1] = shrRoundUp((int)GPULocalWorkSize[1], (int)maxHeight);
	
	int iLocalPixPitch = iBlockDimX + 2;
	GPUError = clSetKernelArg(GPUKernel, 0, sizeof(cl_mem), (void*)&cmBufPyramid);
	GPUError |= clSetKernelArg(GPUKernel, 1, sizeof(cl_mem), (void*)&cmBufPyramidOut);
	GPUError |= clSetKernelArg(GPUKernel, 2, sizeof(cl_uint), (void*)&octvs);
	GPUError |= clSetKernelArg(GPUKernel, 3, sizeof(cl_uint), (void*)&intvlsSum);
	GPUError |= clSetKernelArg(GPUKernel, 4, sizeof(cl_mem), (void*)&cmBufimageWidth);
	GPUError |= clSetKernelArg(GPUKernel, 5, sizeof(cl_mem), (void*)&cmBufimageHeight);
	GPUError |= clSetKernelArg(GPUKernel, 6, sizeof(cl_mem), (void*)&cmBufSigma);
	GPUError |= clSetKernelArg(GPUKernel, 7, sizeof(cl_mem), (void*)&nChannels);

	if(GPUError) return false;

	if(clEnqueueNDRangeKernel( GPUCommandQueue, GPUKernel, 2, NULL, GPUGlobalWorkSize, GPULocalWorkSize, 0, NULL, NULL)) return false;
	return true;
}


bool MeanFilter::SendImageToBufPyramid( IplImage* img, int offset, int* sizeOfImgOct)
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

bool MeanFilter::CreateBuffer(IplImage*** gauss_pyr, int octvs, int intvls, float* sigma, float size )
{
	cmBufPyramid = clCreateBuffer(GPUContext, CL_MEM_READ_WRITE, size, NULL, &GPUError);
	CheckError(GPUError);

	cmBufPyramidOut = clCreateBuffer(GPUContext, CL_MEM_READ_WRITE, size, NULL, &GPUError);
	CheckError(GPUError);
	return true;
}



bool MeanFilter::ReceiveImageToBufPyramid( IplImage* img, int offset, int* sizeOfImgOct)
{
	clock_t start, finish;
	double duration = 0;
	start = clock();
	GPUError = clEnqueueReadBuffer(GPUCommandQueue, cmBufPyramidOut, CL_TRUE, offset, img->imageSize, (void*)img->imageData, 0, NULL, NULL);
	CheckError(GPUError);
	finish = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	RecvTime += duration;

	return true;
}
