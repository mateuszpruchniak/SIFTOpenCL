/*!
 * \file Feature.h
 * \brief 
 *
 * \author Mateusz Pruchniak
 * \date 2011-05-02
 */

#pragma once

#include <oclUtils.h>
#include <iostream>
#include "cv.h"
#include "cxmisc.h"
#include "highgui.h"
#include <vector>
#include <string>
#include <algorithm>
#include <stdio.h>
#include <ctype.h>
#include <time.h>
#include "GPUTransferManager.h"

using namespace std;

/*!
 * \class Feature
 * \brief 
 * \author Mateusz Pruchniak
 * \date 2011-05-02
 */
class Feature
{
	protected:

		/*!
		 * Work-group size - dim X.
		 */
        int iBlockDimX;                    

		/*!
		 * Work-group size - dim Y.
		 */
        int iBlockDimY;                    

		/*!
		 * Error code, only 0 is allowed.
		 */
        cl_int GPUError;

		/*!
		 * Pointer to instance of class GPUTransferManager.
		 */
        GPUTransferManager* GPUTransfer;

		/*!
		 * Loaded .cl file, which contain code responsible for image processing.
		 */
        char* SourceOpenCLFilter;                 

		/*!
		 * Loaded .cl file, which contain support function.
		 */
		char* SourceOpenCL;              

		/*!
		 * Program is formed by a set of kernels, functions and declarations, and it's represented by an cl_program object.
		 */
        cl_program GPUProgram;              

		/*!
		 * Kernels are essentially functions that we can call from the host and that will run on the device
		 */
        cl_kernel GPUFilter;               

		/*!
		 * Global size of NDRange.
		 */
        size_t GPUGlobalWorkSize[2];        

    public:

		/*!
		* Default constructor. Nothing doing.
		*/
		Feature(void);

		/*!
		 * Constructor, creates a program object for a context, loads the source code (.cl files) and build the program.
		 */
        Feature(char* , cl_context GPUContext  ,GPUTransferManager*  ,char* );

		/*!
		 * Destructor.
		 */
        virtual ~Feature();

		/*!
		 * Virtual methods, processing image. Launching the Kernel.
		 */
		virtual bool process(cl_command_queue GPUCommandQueue);
        
		/*!
		 * Check error code.
		 */
        void CheckError(int);

		/*!
		 * Build pyramid
		 */
        void BuildPyramid(IplImage *img);
};

