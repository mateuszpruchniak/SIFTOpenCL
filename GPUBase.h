

#pragma once

#include <oclUtils.h>
#include <iostream>
#include <vector>
#include <string>
#include <stdio.h>
#include <ctype.h>
#include <time.h>
#include "cv.h"
#include "cxmisc.h"
#include "highgui.h"

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

/*feat = new_feature();
ddata = feat_detection_data( feat );
feat->img_pt.x = feat->x = ( c + xc ) * pow( 2.0, octv );
feat->img_pt.y = feat->y = ( r + xr ) * pow( 2.0, octv );
ddata->r = r;
ddata->c = c;
ddata->octv = octv;
ddata->intvl = intvl;
ddata->subintvl = xi;*/

class GPUBase
{
    public:
        

		/*!
		 * Error code, only 0 is allowed.
		 */
        cl_int GPUError;	

		/*!
		 * Platforms are represented by a cl_platform_id, OpenCL framework allow an application to share resources and execute kernels on devices in the platform.
		 */
		cl_platform_id cpPlatform;

		/*!
		 * Device (GPU,CPU) are represented by cl_device_id. Each device has id.
		 */
		cl_device_id* cdDevices;	

		/*!
		 * Total number of devices available to the platform.
		 */
		cl_uint uiDevCount;			
		
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
		 * Check error code.
		 */
        void CheckError(int code);

		/*!
		 * Program is formed by a set of kernels, functions and declarations, and it's represented by an cl_program object.
		 */
        cl_program GPUProgram;    

		cl_program GPUProgram2;    

		/*!
		 * Kernels are essentially functions that we can call from the host and that will run on the device
		 */
        cl_kernel GPUKernel;

		cl_kernel GPUKernel2;

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
		
		cl_mem* buffersListIn;

		int* sizeBuffersIn;

		int numberOfBuffersIn;

		cl_mem* buffersListOut;

		int* sizeBuffersOut;

		int numberOfBuffersOut;

		GPUBase();

		~GPUBase();

		GPUBase(char* source, char* KernelName);

		bool CreateBuffersIn(int maxBufferSize, int numberOfBuffers);

		bool CreateBuffersOut(int maxBufferSize, int numberOfBuffers);

		bool SendImageToBuffers(IplImage* img, ... );

		bool ReceiveImageData(IplImage* img, ... );

		size_t shrRoundUp(int group_size, int global_size);

		char* oclLoadProgSource(const char* cFilename, const char* cPreamble, size_t* szFinalLength);

		int GetKernelSize(double sigma, double cut_off=0.001);
};

