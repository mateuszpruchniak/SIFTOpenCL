/*!
 * \file GPUTransferManager.cpp
 * \brief Class responsible for managing transfer to GPU.
 *
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#include "GPUTransferManager.h"

GPUTransferManager::~GPUTransferManager(void)
{
	 Cleanup();
}

GPUTransferManager::GPUTransferManager()
{
	cmDevBuf = NULL;
    cmPinnedBufOutput = NULL;
}

GPUTransferManager::GPUTransferManager( cl_context GPUContextArg, cl_command_queue GPUCommandQueueArg, unsigned int width, unsigned int height, int channels )
{
    //cout << "data transfer konstr" << endl;
	nChannels = channels;
    GPUContext = GPUContextArg;
    ImageHeight = height;
    ImageWidth = width;
    GPUCommandQueue = GPUCommandQueueArg;
    // Allocate pinned input and output host image buffers:  mem copy operations to/from pinned memory is much faster than paged memory
    szBuffBytes = ImageWidth * ImageHeight * nChannels * sizeof (char);
    CreateBuffers();
}


void GPUTransferManager::CheckError(int code)
{
    switch(code)
    {
    case CL_SUCCESS:
        return;
        break;
    case CL_INVALID_COMMAND_QUEUE:
        cout << "CL_INVALID_COMMAND_QUEUE" << endl;
        break;
    case CL_INVALID_CONTEXT:
        cout << "CL_INVALID_CONTEXT" << endl;
        break;
    case CL_INVALID_MEM_OBJECT:
        cout << "CL_INVALID_MEM_OBJECT" << endl;
        break;
    case CL_INVALID_VALUE:
        cout << "CL_INVALID_VALUE" << endl;
        break;
    case CL_INVALID_EVENT_WAIT_LIST:
        cout << "CL_INVALID_EVENT_WAIT_LIST" << endl;
        break;
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        cout << "CL_MEM_OBJECT_ALLOCATION_FAILURE" << endl;
        break;
    case CL_OUT_OF_HOST_MEMORY:
        cout << "CL_OUT_OF_HOST_MEMORY" << endl;
        break;
    default:
         cout << "OTHERS ERROR" << endl;
    }
}

void GPUTransferManager::Cleanup()
{
    // Cleanup allocated objects
    cout << "\nStarting Cleanup...\n\n";

    if(cmDevBuf)clReleaseMemObject(cmDevBuf);
}

IplImage* GPUTransferManager::ReceiveImage()
{
	int szBuffBytesLocal = ImageWidth * ImageHeight * nChannels * sizeof (char);
    GPUError = clEnqueueReadBuffer(GPUCommandQueue, cmDevBufOutput, CL_TRUE, 0, szBuffBytesLocal, (void*)image->imageData, 0, NULL, NULL);
    CheckError(GPUError);
    return image;
}

void GPUTransferManager::ReceiveImageData(char* imageData)
{
	int szBuffBytesLocal = ImageWidth * ImageHeight * nChannels * sizeof (char);
    GPUError = clEnqueueReadBuffer(GPUCommandQueue, cmDevBufOutput, CL_TRUE, 0, szBuffBytesLocal, (void*)imageData, 0, NULL, NULL);
    CheckError(GPUError);
}

void GPUTransferManager::ReceiveImageData(char* imageData, char* imageData2)
{
	int szBuffBytesLocal = ImageWidth * ImageHeight * nChannels * sizeof (char);
	GPUError = clEnqueueReadBuffer(GPUCommandQueue, cmDevBufOutput, CL_TRUE, 0, szBuffBytesLocal, (void*)imageData, 0, NULL, NULL);
	CheckError(GPUError);

	GPUError = clEnqueueReadBuffer(GPUCommandQueue, cmDevBufOutput2, CL_TRUE, 0, szBuffBytesLocal, (void*)imageData2, 0, NULL, NULL);
	CheckError(GPUError);

}


void GPUTransferManager::SendImageData(char* imageData, int height, int width)
{
	ImageHeight = height;
    ImageWidth = width;
    int szBuffBytesLocal = ImageWidth * ImageHeight * nChannels * sizeof (char);
    GPUError = clEnqueueWriteBuffer(GPUCommandQueue, cmDevBuf, CL_TRUE, 0, szBuffBytesLocal, (void*)imageData, 0, NULL, NULL);
    CheckError(GPUError);
}

void GPUTransferManager::SendImage( IplImage* imageToLoad )
{
	image = imageToLoad;
	ImageHeight = imageToLoad->height;
    ImageWidth = imageToLoad->width;
    int szBuffBytesLocal = ImageWidth * ImageHeight * nChannels * sizeof (char);
    GPUError = clEnqueueWriteBuffer(GPUCommandQueue, cmDevBuf, CL_TRUE, 0, szBuffBytesLocal, (void*)imageToLoad->imageData, 0, NULL, NULL);
    CheckError(GPUError);
}

bool GPUTransferManager::CheckImage(IplImage* img)
{
	//int ImageHeight = img->height;
 //   int ImageWidth = img->width;
 //   int size = ImageHeight * ImageWidth * nChannels * sizeof(char);
 //   if( size > szBuffBytes )
 //   {
 //   	// Allocate pinned input and output host image buffers:  mem copy operations to/from pinned memory is much faster than paged memory
	//	//szBuffBytes = size;
	//	//CreateBuffers();
 //   	return false;
 //   }
	return true;
}

void GPUTransferManager::CreateBuffers()
{
	//cmPinnedBufOutput = clCreateBuffer(GPUContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, szBuffBytes, NULL, &GPUError);
	//CheckError(GPUError);

	//// Enqueues a command to map a region of the buffer object given by buffer into the host address space and returns a pointer to this mapped region.
	//GPUOutput = (cl_uint*)clEnqueueMapBuffer(GPUCommandQueue, cmPinnedBufOutput, CL_TRUE, CL_MAP_WRITE, 0, szBuffBytes, 0, NULL, NULL, &GPUError);
	//CheckError(GPUError);

	// Create the device buffers in GMEM on each device, for now we have one device :)
	cmDevBuf = clCreateBuffer(GPUContext, CL_MEM_READ_WRITE, szBuffBytes, NULL, &GPUError);
	CheckError(GPUError);

	cmDevBuf2 = clCreateBuffer(GPUContext, CL_MEM_READ_WRITE, szBuffBytes, NULL, &GPUError);
	CheckError(GPUError);

	cmDevBuf3 = clCreateBuffer(GPUContext, CL_MEM_READ_WRITE, szBuffBytes, NULL, &GPUError);
	CheckError(GPUError);

	cmDevBuf4 = clCreateBuffer(GPUContext, CL_MEM_READ_WRITE, szBuffBytes, NULL, &GPUError);
	CheckError(GPUError);

	// Create the device buffers in GMEM on each device, for now we have one device :)
	cmDevBufOutput = clCreateBuffer(GPUContext, CL_MEM_READ_WRITE, szBuffBytes, NULL, &GPUError);
	CheckError(GPUError);

	cmDevBufOutput2 = clCreateBuffer(GPUContext, CL_MEM_READ_WRITE, szBuffBytes, NULL, &GPUError);
	CheckError(GPUError);
}


