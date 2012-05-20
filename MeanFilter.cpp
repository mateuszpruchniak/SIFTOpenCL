

#include "MeanFilter.h"



MeanFilter::~MeanFilter(void)
{
}

MeanFilter::MeanFilter(): GPUBase("C:\\Users\\Mati\\Desktop\\Dropbox\\MGR\\SIFTOpenCL\\GPU\\OpenCL\\BlurGaussFilter.cl","ckConv")
{

}

bool MeanFilter::Process( float sigma )
{
	int maskSize = cvRound(sigma * 3 * 2 + 1) | 1;
	size_t GPULocalWorkSize[2];
	GPULocalWorkSize[0] = iBlockDimX;
	GPULocalWorkSize[1] = iBlockDimY;
	GPUGlobalWorkSize[0] = shrRoundUp((int)GPULocalWorkSize[0], (int)imageWidth);
	GPUGlobalWorkSize[1] = shrRoundUp((int)GPULocalWorkSize[1], (int)imageHeight);
	
	int iLocalPixPitch = iBlockDimX + 2;
	GPUError = clSetKernelArg(GPUKernel, 0, sizeof(cl_mem), (void*)&GPU::getInstance().buffersListIn[0]);
	GPUError = clSetKernelArg(GPUKernel, 1, sizeof(cl_mem), (void*)&GPU::getInstance().buffersListOut[0]);
	GPUError |= clSetKernelArg(GPUKernel, 2, sizeof(cl_uint), (void*)&imageWidth);
	GPUError |= clSetKernelArg(GPUKernel, 3, sizeof(cl_uint), (void*)&imageHeight);
	GPUError |= clSetKernelArg(GPUKernel, 4, sizeof(cl_float), (void*)&sigma);
	GPUError |= clSetKernelArg(GPUKernel, 5, sizeof(cl_int), (void*)&maskSize);
	if(GPUError) return false;

	if(clEnqueueNDRangeKernel( GPUCommandQueue, GPUKernel, 2, NULL, GPUGlobalWorkSize, GPULocalWorkSize, 0, NULL, NULL)) return false;
	return true;
}


bool MeanFilter::SendImageToBufPyramid( IplImage* img, int offset, int* sizeOfImgOct)
{


	GPUError = clEnqueueWriteBuffer(GPUCommandQueue, cmBufPyramid, CL_TRUE, offset, img->imageSize, (void*)img->imageData, 0, NULL, NULL);
	CheckError(GPUError);




	return true;
}

bool MeanFilter::Process2(IplImage*** gauss_pyr, int octvs, int intvls, float* sigma, float size )
{


	cmBufPyramid = clCreateBuffer(GPUContext, CL_MEM_READ_WRITE, size, NULL, &GPUError);
	CheckError(GPUError);




	/*int maskSize = cvRound(sigma * 3 * 2 + 1) | 1;
	size_t GPULocalWorkSize[2];
	GPULocalWorkSize[0] = iBlockDimX;
	GPULocalWorkSize[1] = iBlockDimY;
	GPUGlobalWorkSize[0] = shrRoundUp((int)GPULocalWorkSize[0], (int)imageWidth);
	GPUGlobalWorkSize[1] = shrRoundUp((int)GPULocalWorkSize[1], (int)imageHeight);
	
	int iLocalPixPitch = iBlockDimX + 2;
	GPUError = clSetKernelArg(GPUKernel, 0, sizeof(cl_mem), (void*)&GPU::getInstance().buffersListIn[0]);
	GPUError = clSetKernelArg(GPUKernel, 1, sizeof(cl_mem), (void*)&GPU::getInstance().buffersListOut[0]);
	GPUError |= clSetKernelArg(GPUKernel, 2, sizeof(cl_uint), (void*)&imageWidth);
	GPUError |= clSetKernelArg(GPUKernel, 3, sizeof(cl_uint), (void*)&imageHeight);
	GPUError |= clSetKernelArg(GPUKernel, 4, sizeof(cl_float), (void*)&sigma);
	GPUError |= clSetKernelArg(GPUKernel, 5, sizeof(cl_int), (void*)&maskSize);
	if(GPUError) return false;

	if(clEnqueueNDRangeKernel( GPUCommandQueue, GPUKernel, 2, NULL, GPUGlobalWorkSize, GPULocalWorkSize, 0, NULL, NULL)) return false;
	*/
	
	return true;
}