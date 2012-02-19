

#include "Subtract.h"



Subtract::Subtract(): GPUBase("C:\\Dropbox\\MGR\\GPUFeatureExtraction\\GPU\\OpenCL\\Subtract.cl","ckSub")
{

}


Subtract::~Subtract(void)
{
}



bool Subtract::Process()
{
	size_t GPULocalWorkSize[2];
	GPULocalWorkSize[0] = iBlockDimX;
	GPULocalWorkSize[1] = iBlockDimY;
	GPUGlobalWorkSize[0] = shrRoundUp((int)GPULocalWorkSize[0], (int)imageWidth);
	GPUGlobalWorkSize[1] = shrRoundUp((int)GPULocalWorkSize[1], (int)imageHeight);
	
	int iLocalPixPitch = iBlockDimX + 2;
	GPUError = clSetKernelArg(GPUKernel, 0, sizeof(cl_mem), (void*)&buffersListIn[0]);
	GPUError = clSetKernelArg(GPUKernel, 1, sizeof(cl_mem), (void*)&buffersListIn[1]);
	GPUError = clSetKernelArg(GPUKernel, 2, sizeof(cl_mem), (void*)&buffersListOut[0]);
	GPUError |= clSetKernelArg(GPUKernel, 3, sizeof(cl_uint), (void*)&imageWidth);
	GPUError |= clSetKernelArg(GPUKernel, 4, sizeof(cl_uint), (void*)&imageHeight);
	if(GPUError) return false;

	if(clEnqueueNDRangeKernel( GPUCommandQueue, GPUKernel, 2, NULL, GPUGlobalWorkSize, GPULocalWorkSize, 0, NULL, NULL)) return false;
	return true;
}