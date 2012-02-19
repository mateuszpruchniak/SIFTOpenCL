/*!
 * \file Feature.cpp
 * \brief 
 *
 * \author Mateusz Pruchniak
 * \date 2010-05-02
 */

#include "Feature.h"

Feature::Feature(void)
{

}

Feature::Feature(char* source, cl_context GPUContext ,GPUTransferManager* transfer,char* KernelName)
{
    GPUTransfer = transfer;

    iBlockDimX = 8;
    iBlockDimY = 8;
    size_t szKernelLength;	
	size_t szKernelLengthFilter;
	size_t szKernelLengthSum;

    // Load OpenCL kernel
	SourceOpenCL = oclLoadProgSource("./CL/GPUCode.cl", "// My comment\n", &szKernelLength);
    SourceOpenCLFilter = oclLoadProgSource(source, "// My comment\n", &szKernelLengthFilter);
	//strncat (SourceOpenCL, SourceOpenCLFilter,szKernelLengthFilter );
	szKernelLengthSum = szKernelLength + szKernelLengthFilter;
	char* sourceCL = new char[szKernelLengthSum];
	strcpy(sourceCL,SourceOpenCL);
	strcat (sourceCL, SourceOpenCLFilter);

    // creates a program object for a context, and loads the source code specified by the text strings in
    //the strings array into the program object. The devices associated with the program object are the
    //devices associated with context.
    GPUProgram = clCreateProgramWithSource( GPUContext , 1, (const char **)&sourceCL, &szKernelLengthSum, &GPUError);
    CheckError(GPUError);

    // Build the program with 'mad' Optimization option
    char *flags = "-cl-mad-enable";

    GPUError = clBuildProgram(GPUProgram, 0, NULL, flags, NULL, NULL);
    CheckError(GPUError);

    
}

Feature::~Feature()
{
    if(GPUProgram)clReleaseProgram(GPUProgram);
    if(GPUFilter)clReleaseKernel(GPUFilter);
}


void Feature::CheckError(int code)
{
    switch(code)
    {
    case CL_SUCCESS:
        return;
        break;
    default:
         cout << "OTHERS ERROR" << endl;
    }

    //getchar();
}

bool Feature::process(cl_command_queue GPUCommandQueue)
{
	
	/*
	----------------------------------------------------------------------------------------------

	*/

	GPUFilter = clCreateKernel(GPUProgram, "ckBuildPyramid", &GPUError);
    int iLocalPixPitch = iBlockDimX + 2;
	GPUError = clSetKernelArg(GPUFilter, 0, sizeof(cl_mem), (void*)&GPUTransfer->cmDevBuf);
    GPUError |= clSetKernelArg(GPUFilter, 1, sizeof(cl_uint), (void*)&GPUTransfer->ImageWidth);
    GPUError |= clSetKernelArg(GPUFilter, 2, sizeof(cl_uint), (void*)&GPUTransfer->ImageHeight);
	GPUError |= clSetKernelArg(GPUFilter, 3, sizeof(cl_int), (void*)&GPUTransfer->nChannels);
    if(GPUError) return false;

	size_t GPULocalWorkSize[2]; 
    GPULocalWorkSize[0] = iBlockDimX;
    GPULocalWorkSize[1] = iBlockDimY;
    GPUGlobalWorkSize[0] = shrRoundUp((int)GPULocalWorkSize[0], GPUTransfer->ImageWidth); 

    GPUGlobalWorkSize[1] = shrRoundUp((int)GPULocalWorkSize[1], (int)GPUTransfer->ImageHeight);

    if( clEnqueueNDRangeKernel( GPUCommandQueue, GPUFilter, 2, NULL, GPUGlobalWorkSize, GPULocalWorkSize, 0, NULL, NULL) ) return false;


	/*
	----------------------------------------------------------------------------------------------
	*/








    return true;
}


void Feature::BuildPyramid(IplImage *img)
{
	


}