

#pragma once

#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <string>
#include <stdio.h>
#include <ctype.h>
#include <time.h>
#include "cv.h"
#include "cxmisc.h"
#include "highgui.h"
#include <stdarg.h>
#include "Shared.h"
#include <math.h>

using namespace std;

#define MAX_KERNEL_SIZE	20

typedef struct
{
	float	scx;
	float	scy;
	float	x;			/**< x coord */
	float	y;			/**< y coord */
	float	subintvl;
	float	intvl;	
	float	octv;
	float	scl;
	float	scl_octv;	/**< scale of a Lowe-style feature */
	float	ori;		/**< orientation of a Lowe-style feature */
	float	mag;
	float	desc[128];
} Keys;




class GPU
{
  private:

	cl_device_id device;
	cl_uint platforms;
	cl_uint devices;
	cl_int GPUError;

	
	/*!
	 * Platforms are represented by a cl_platform_id, OpenCL framework allow an application to share resources and execute kernels on devices in the platform.
	 */
	cl_platform_id cpPlatform;

    GPU() 
	{
		printf("\n ----------- SINGLETON START --------------- \n");
		// Fetch the Platform and Device IDs; we only want one.
		GPUError = clGetPlatformIDs(1, &cpPlatform, &platforms);
		if (GPUError != CL_SUCCESS) {
				printf("\n Error number %d", GPUError);
		}

		GPUError = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device, &devices);

		cl_context_properties properties[]={
		CL_CONTEXT_PLATFORM, (cl_context_properties)cpPlatform,
		0};

		// Note that nVidia's OpenCL requires the platform property
		GPUContext = clCreateContext(properties, 1, &device, NULL, NULL, &GPUError);
		GPUCommandQueue = clCreateCommandQueue(GPUContext, device, 0, &GPUError);
		if (GPUError != CL_SUCCESS) {
				printf("\n Error number %d", GPUError);
		}
		
		maxNumberBufIn  = 0;
		maxNumberBufOut = 0;
		
		printf("\n ----------- SINGLETON END --------------- \n");
	}
    
	
	int maxNumberBufIn;
	
	int maxNumberBufOut;

    GPU(const GPU&);
    
    GPU& operator=(const GPU&);

  public:

	/*!
	 * OpenCL command-queue, is an object where OpenCL commands are enqueued to be executed by the device.
	 * "The command-queue is created on a specific device in a context [...] Having multiple command-queues allows applications to queue multiple independent commands without requiring synchronization." (OpenCL Specification).
	 */
	cl_command_queue GPUCommandQueue; 
	
	/*!
	 * Context defines the entire OpenCL environment, including OpenCL kernels, devices, memory management, command-queues, etc. Contexts in OpenCL are referenced by an cl_context object
	 */
	cl_context GPUContext; 
	
	cl_mem* buffersListIn;

	int numberOfBuffersIn;

	cl_mem* buffersListOut;

	int numberOfBuffersOut;
	
	bool CreateBuffersIn(int maxBufferSize, int numberOfBuffers);

	bool CreateBuffersOut(int maxBufferSize, int numberOfBuffers);
	
	static GPU& getInstance()
	{
	  static GPU instance;
	  return instance;
	}
};



class GPUBase
{
    public:
        
		
	/*!
	 * OpenCL command-queue, is an object where OpenCL commands are enqueued to be executed by the device.
	 * "The command-queue is created on a specific device in a context [...] Having multiple command-queues allows applications to queue multiple independent commands without requiring synchronization." (OpenCL Specification).
	 */
	cl_command_queue GPUCommandQueue; 

	/*!
	 * Context defines the entire OpenCL environment, including OpenCL kernels, devices, memory management, command-queues, etc. Contexts in OpenCL are referenced by an cl_context object
	 */
	cl_context GPUContext;  

	/*!
	 * Kernels are essentially functions that we can call from the host and that will run on the device
	 */
	cl_kernel GPUKernel;

	/*!
	 * Error code, only 0 is allowed.
	 */
	cl_int GPUError;	

	/*!
	 * Program is formed by a set of kernels, functions and declarations, and it's represented by an cl_program object.
	 */
	cl_program GPUProgram;
	  

	/*!
	 * Check error code.
	 */
	void CheckError(int code);

	

	/*!
	 * Work-group size - dim X.
	 */
	int iBlockDimX;                    

	/*!
	 * Work-group size - dim Y.
	 */
	int iBlockDimY;  

	/*!
	 * Image width.
	 */
	unsigned int imageWidth;   

	/*!
	 * Image height.
	 */
	unsigned int imageHeight; 

	/*!
	 * Global size of NDRange.
	 */
	size_t GPUGlobalWorkSize[2];

	char* kernelFuncName;
	

	GPUBase();

	~GPUBase();

	GPUBase(char* source, char* KernelName);

	bool CreateBuffersIn(int maxBufferSize, int numberOfBuffers);

	bool CreateBuffersOut(int maxBufferSize, int numberOfBuffers);

	bool SendImageToBuffers(int number, ... );

	bool ReceiveImageData(int number, ... );

	size_t shrRoundUp(int group_size, int global_size);

	char* oclLoadProgSource(const char* cFilename, const char* cPreamble, size_t* szFinalLength);

	int GetKernelSize(double sigma, double cut_off=0.001);
};


