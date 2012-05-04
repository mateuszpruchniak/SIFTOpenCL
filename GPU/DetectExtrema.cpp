
#include "DetectExtrema.h"



DetectExtrema::~DetectExtrema(void)
{
}

DetectExtrema::DetectExtrema(int _maxNumberKeys, Keys* _keys): GPUBase("C:\\Dropbox\\MGR\\SIFTOpenCL\\GPU\\OpenCL\\DetectExtrema.cl","ckDetect")
{
	counter = 0;
	numberExtr = 0;
	numberExtrRej = 0;
	maxNumberKeys = _maxNumberKeys;
	counter = 0;
	keys = _keys;

	for (int i =0 ; i < maxNumberKeys ; i++)
	{
		keys[i].x = 0.0;
		keys[i].y = 0.0;
		keys[i].intvl = 0.0;
		keys[i].octv = 0.0;
		keys[i].subintvl = 0.0;
		keys[i].scx = 0.0;
		keys[i].scy = 0.0;
	}

	cmDevBufNumber = clCreateBuffer(GPUContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &GPUError);
	CheckError(GPUError);
	GPUError = clEnqueueWriteBuffer(GPUCommandQueue, cmDevBufNumber, CL_TRUE, 0, sizeof(int), (void*)&numberExtr, 0, NULL, NULL);
	CheckError(GPUError);

	cmDevBufNumberReject = clCreateBuffer(GPUContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &GPUError);
	CheckError(GPUError);
	GPUError = clEnqueueWriteBuffer(GPUCommandQueue, cmDevBufNumberReject, CL_TRUE, 0, sizeof(int), (void*)&numberExtrRej, 0, NULL, NULL);
	CheckError(GPUError);

	cmDevBufCount = clCreateBuffer(GPUContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &GPUError);
	CheckError(GPUError);
	GPUError = clEnqueueWriteBuffer(GPUCommandQueue, cmDevBufCount, CL_TRUE, 0, sizeof(int), (void*)&counter, 0, NULL, NULL);
	CheckError(GPUError);

	cmDevBufKeys = clCreateBuffer(GPUContext, CL_MEM_READ_WRITE, maxNumberKeys*sizeof(Keys), NULL, &GPUError);
	CheckError(GPUError);
	GPUError = clEnqueueWriteBuffer(GPUCommandQueue, cmDevBufKeys, CL_TRUE, 0, maxNumberKeys*sizeof(Keys), (void*)keys, 0, NULL, NULL);
	CheckError(GPUError);

	GPUKernelDesc = clCreateKernel(GPUProgram, "ckDesc", &GPUError);
	CheckError(GPUError);

}


bool DetectExtrema::Process( int* numExtr, int* numExtrRej, float prelim_contr_thr, int intvl, int octv, Keys* keys, IplImage* img )
{
	counter = 0;
	numberExtr = 0;
	numberExtrRej = 0;

	GPUError = clEnqueueWriteBuffer(GPUCommandQueue, cmDevBufNumber, CL_TRUE, 0, sizeof(int), (void*)&numberExtr, 0, NULL, NULL);
	CheckError(GPUError);
	GPUError = clEnqueueWriteBuffer(GPUCommandQueue, cmDevBufNumberReject, CL_TRUE, 0, sizeof(int), (void*)&numberExtrRej, 0, NULL, NULL);
	CheckError(GPUError);

	size_t GPULocalWorkSize[2];
	GPULocalWorkSize[0] = iBlockDimX;
	GPULocalWorkSize[1] = iBlockDimY;
	GPUGlobalWorkSize[0] = shrRoundUp((int)GPULocalWorkSize[0], (int)imageWidth);
	GPUGlobalWorkSize[1] = shrRoundUp((int)GPULocalWorkSize[1], (int)imageHeight);

	int iLocalPixPitch = iBlockDimX + 2;
	GPUError = clSetKernelArg(GPUKernel, 0, sizeof(cl_mem), (void*)&buffersListIn[0]);
	GPUError |= clSetKernelArg(GPUKernel, 1, sizeof(cl_mem), (void*)&buffersListIn[1]);
	GPUError |= clSetKernelArg(GPUKernel, 2, sizeof(cl_mem), (void*)&buffersListIn[2]);
	GPUError |= clSetKernelArg(GPUKernel, 3, sizeof(cl_mem), (void*)&buffersListOut[0]);
	GPUError |= clSetKernelArg(GPUKernel, 4, sizeof(cl_mem), (void*)&cmDevBufKeys);
	GPUError |= clSetKernelArg(GPUKernel, 5, sizeof(cl_uint), (void*)&imageWidth);
	GPUError |= clSetKernelArg(GPUKernel, 6, sizeof(cl_uint), (void*)&imageHeight);
	GPUError |= clSetKernelArg(GPUKernel, 7, sizeof(cl_float), (void*)&prelim_contr_thr);
	GPUError |= clSetKernelArg(GPUKernel, 8, sizeof(cl_uint), (void*)&intvl);
	GPUError |= clSetKernelArg(GPUKernel, 9, sizeof(cl_uint), (void*)&octv);
	GPUError |= clSetKernelArg(GPUKernel, 10, sizeof(cl_mem), (void*)&cmDevBufNumber);
	GPUError |= clSetKernelArg(GPUKernel, 11, sizeof(cl_mem), (void*)&cmDevBufNumberReject);
	if(GPUError) return false;

	if(clEnqueueNDRangeKernel( GPUCommandQueue, GPUKernel, 2, NULL, GPUGlobalWorkSize, GPULocalWorkSize, 0, NULL, NULL)) return false;

	GPUError = clEnqueueReadBuffer(GPUCommandQueue, cmDevBufNumber, CL_TRUE, 0, sizeof(int), (void*)&numberExtr, 0, NULL, NULL);
	CheckError(GPUError);

	GPUError = clEnqueueReadBuffer(GPUCommandQueue, cmDevBufNumberReject, CL_TRUE, 0, sizeof(int), (void*)&numberExtrRej, 0, NULL, NULL);
	CheckError(GPUError);

	*numExtr = numberExtr;
	*numExtrRej = numberExtrRej;

	GPULocalWorkSize[0] = iBlockDimX;
	GPULocalWorkSize[1] = iBlockDimY;
	GPUGlobalWorkSize[0] = shrRoundUp((int)GPULocalWorkSize[0], (int)numberExtr);
	GPUGlobalWorkSize[1] = shrRoundUp((int)GPULocalWorkSize[1], (int)1);

	iLocalPixPitch = iBlockDimX + 2;
	GPUError |= clSetKernelArg(GPUKernelDesc, 0, sizeof(cl_mem), (void*)&buffersListIn[1]);
	GPUError |= clSetKernelArg(GPUKernelDesc, 1, sizeof(cl_mem), (void*)&buffersListOut[0]);
	GPUError |= clSetKernelArg(GPUKernelDesc, 2, sizeof(cl_mem), (void*)&cmDevBufCount);
	GPUError |= clSetKernelArg(GPUKernelDesc, 3, sizeof(cl_mem), (void*)&cmDevBufKeys);
	GPUError |= clSetKernelArg(GPUKernelDesc, 4, sizeof(cl_uint), (void*)&imageWidth);
	GPUError |= clSetKernelArg(GPUKernelDesc, 5, sizeof(cl_uint), (void*)&imageHeight);
	GPUError |= clSetKernelArg(GPUKernelDesc, 6, sizeof(cl_float), (void*)&prelim_contr_thr);
	GPUError |= clSetKernelArg(GPUKernelDesc, 7, sizeof(cl_uint), (void*)&intvl);
	GPUError |= clSetKernelArg(GPUKernelDesc, 8, sizeof(cl_uint), (void*)&octv);
	GPUError |= clSetKernelArg(GPUKernelDesc, 9, sizeof(cl_mem), (void*)&cmDevBufNumber);
	if(GPUError) return false;

	if(clEnqueueNDRangeKernel( GPUCommandQueue, GPUKernelDesc, 2, NULL, GPUGlobalWorkSize, GPULocalWorkSize, 0, NULL, NULL)) 
		cout << "Error clEnqueueNDRangeKernel" << endl;

	GPUError = clEnqueueReadBuffer(GPUCommandQueue, cmDevBufCount, CL_TRUE, 0, sizeof(int), (void*)&counter, 0, NULL, NULL);
	CheckError(GPUError);

	GPUError = clEnqueueReadBuffer(GPUCommandQueue, cmDevBufKeys, CL_TRUE, 0, maxNumberKeys*sizeof(Keys), (void*)keys, 0, NULL, NULL);
	CheckError(GPUError);

	return true;
}