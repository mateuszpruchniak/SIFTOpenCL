

#include "Subtract.h"



Subtract::Subtract(): PyramidProcess("C:\\Users\\Mati\\Desktop\\Dropbox\\MGR\\SIFTOpenCL\\GPU\\OpenCL\\Subtract.cl","ckSub")
{

}


Subtract::~Subtract(void)
{
}


bool Subtract::Process(cl_mem gaussPyr, int imageWidth, int imageHeight, int OffsetPrev, int OffsetAct)
{

	OffsetAct = OffsetAct / 4;
	OffsetPrev = OffsetPrev / 4;

	size_t GPULocalWorkSize[2];
	GPULocalWorkSize[0] = iBlockDimX;
	GPULocalWorkSize[1] = iBlockDimY;
	GPUGlobalWorkSize[0] = RoundUpGroupDim((int)GPULocalWorkSize[0], (int)imageWidth);
	GPUGlobalWorkSize[1] = RoundUpGroupDim((int)GPULocalWorkSize[1], (int)imageHeight);
	
	int iLocalPixPitch = iBlockDimX + 2;
	GPUError = clSetKernelArg(GPUKernel, 0, sizeof(cl_mem), (void*)&gaussPyr);
	GPUError |= clSetKernelArg(GPUKernel, 1, sizeof(cl_mem), (void*)&cmBufPyramid);
	GPUError |= clSetKernelArg(GPUKernel, 2, sizeof(cl_uint), (void*)&OffsetPrev);
	GPUError |= clSetKernelArg(GPUKernel, 3, sizeof(cl_uint), (void*)&OffsetAct);
	GPUError |= clSetKernelArg(GPUKernel, 4, sizeof(cl_uint), (void*)&imageWidth);
	GPUError |= clSetKernelArg(GPUKernel, 5, sizeof(cl_uint), (void*)&imageHeight);

	if(GPUError) return false;

	if(clEnqueueNDRangeKernel( GPUCommandQueue, GPUKernel, 2, NULL, GPUGlobalWorkSize, GPULocalWorkSize, 0, NULL, NULL)) return false;
	return true;
}
