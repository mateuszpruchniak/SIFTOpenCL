

#include "MeanFilter.h"



MeanFilter::~MeanFilter(void)
{
}

MeanFilter::MeanFilter(): PyramidProcess("C:\\Users\\Mati\\Desktop\\Dropbox\\MGR\\SIFTOpenCL\\GPU\\OpenCL\\BlurGaussFilter.cl","ckConv")
{

}

bool MeanFilter::Process(float sigma, int imageWidth, int imageHeight, int OffsetAct, int OffsetNext)
{

	OffsetAct = OffsetAct / 4;
	OffsetNext = OffsetNext / 4;

	int maskSize = 0;
	maskSize = cvRound(sigma * 3.0 * 2.0 + 1.0) | 1;


	size_t GPULocalWorkSize[2];
	GPULocalWorkSize[0] = iBlockDimX;
	GPULocalWorkSize[1] = iBlockDimY;
	GPUGlobalWorkSize[0] = RoundUpGroupDim((int)GPULocalWorkSize[0], (int)imageWidth);
	GPUGlobalWorkSize[1] = RoundUpGroupDim((int)GPULocalWorkSize[1], (int)imageHeight);
	
	int iLocalPixPitch = iBlockDimX + 2;
	GPUError = clSetKernelArg(GPUKernel, 0, sizeof(cl_mem), (void*)&cmBufPyramid);
	GPUError |= clSetKernelArg(GPUKernel, 1, sizeof(cl_uint), (void*)&OffsetAct);
	GPUError |= clSetKernelArg(GPUKernel, 2, sizeof(cl_uint), (void*)&OffsetNext);
	GPUError |= clSetKernelArg(GPUKernel, 3, sizeof(cl_uint), (void*)&imageWidth);
	GPUError |= clSetKernelArg(GPUKernel, 4, sizeof(cl_uint), (void*)&imageHeight);
	GPUError |= clSetKernelArg(GPUKernel, 5, sizeof(cl_float), (void*)&sigma);
	GPUError |= clSetKernelArg(GPUKernel, 6, sizeof(cl_uint), (void*)&maskSize);

	if(GPUError) return false;

	if(clEnqueueNDRangeKernel( GPUCommandQueue, GPUKernel, 2, NULL, GPUGlobalWorkSize, GPULocalWorkSize, 0, NULL, NULL)) return false;
	return true;
}
