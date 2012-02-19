/*!
 * \file GPUTransferManager.h
 * \brief File contains class responsible for managing transfer to GPU.
 *
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

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

/*!
 * \class GPUTransferManager
 * \brief Class responsible for managing transfer between GPU and CPU.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class GPUTransferManager
{
	private:
                


        /*!
		 * Mapped Pointer to pinned Host output buffer for host processing.
		 */
		cl_uint* GPUOutput;


        /*!
		 * OpenCL host memory output buffer object:  pinned.
		 */
		cl_mem cmPinnedBufOutput;



		/*!
		 * Error code, only 0 is allowed.
		 */
        cl_int GPUError;

		/*!
		 * Image return from buffer.
		 */
        IplImage* image;


    public:
        /*!
		 * The size in bytes of the buffer memory object to be allocated.
		 */
		size_t szBuffBytes;

		/*!
		 * OpenCL command-queue, is an object where OpenCL commands are enqueued to be executed by the device.
		*/
        cl_command_queue GPUCommandQueue; 

		/*!
		 * Context defines the entire OpenCL environment, including OpenCL kernels, devices, memory management, command-queues, etc. Contexts in OpenCL are referenced by an cl_context object
		 */
        cl_context GPUContext;    

		/*!
		 * OpenCL device memory input buffer object.
		 */
        cl_mem cmDevBuf;

		cl_mem cmDevBuf2;

		cl_mem cmDevBuf3;

		cl_mem cmDevBuf4;


        /*!
		 * OpenCL device memory output buffer object.
		 */
		cl_mem cmDevBufOutput;

		cl_mem cmDevBufOutput2;

		/*!
		 * Image width.
		 */
        unsigned int ImageWidth;   

		/*!
		 * Image height.
		 */
        unsigned int ImageHeight;  

		/*!
		 * Number of color channels.
		 */
		int nChannels;

		/*!
		 * Destructor. Release buffers.
		 */
        ~GPUTransferManager();

        /*!
		 * Constructor. Allocate pinned and mapped memory for input and output host image buffers.
		 */
        GPUTransferManager( cl_context , cl_command_queue , unsigned int , unsigned int,  int nChannels );

		/*!
		 * Default Constructor.
		 */
        GPUTransferManager();

        /*!
		 * Send image to GPU memory.
		 */
        void SendImage( IplImage*  );

		void SendImageData(char* imageData, int height, int width);

        /*!
		 * Get image from GPU memory.
		 */
        IplImage* ReceiveImage();


		void ReceiveImageData(char* imageData);
		void ReceiveImageData(char* imageData, char* imageData2);

		
		/*!
		 * Checking size of image(if image size is larger than max size, this function increases buffers)
		 */
		bool CheckImage(IplImage*);
        
		/*!
		 * Release all buffers.
		 */
        void Cleanup();

        /*!
		 * Check error code.
		 */
        void CheckError(int );

        /*!
		 * Create buffers.
		 */
        void CreateBuffers();

};


