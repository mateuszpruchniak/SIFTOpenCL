

#include "GPUBase.h"


GPUBase::GPUBase(char* source, char* KernelName)
{
	kernelFuncName = KernelName;
	size_t szKernelLength;
	size_t szKernelLengthFilter;
	size_t szKernelLengthSum;
	char* SourceOpenCLShared;
	char* SourceOpenCL;
	iBlockDimX = 16;
	iBlockDimY = 16;

	GPUError = oclGetPlatformID(&cpPlatform);
	CheckError(GPUError);

	cl_uint uiNumAllDevs = 0;

	// Get the number of GPU devices available to the platform
	GPUError = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &uiNumAllDevs);
	CheckError(GPUError);
	uiDevCount = uiNumAllDevs;

	// Create the device list
	cdDevices = new cl_device_id [uiDevCount];
	GPUError = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, uiDevCount, cdDevices, NULL);
	CheckError(GPUError);

	// Create the OpenCL context on a GPU device
	GPUContext = clCreateContext(0, uiNumAllDevs, cdDevices, NULL, NULL, &GPUError);
	CheckError(GPUError);

	//The command-queue can be used to queue a set of operations (referred to as commands) in order.
	GPUCommandQueue = clCreateCommandQueue(GPUContext, cdDevices[0], 0, &GPUError);
	CheckError(GPUError);

	oclPrintDevName(LOGBOTH, cdDevices[0]);

	// Load OpenCL kernel
	SourceOpenCLShared = oclLoadProgSource("C:\\Dropbox\\MGR\\GPUFeatureExtraction\\GPU\\OpenCL\\GPUCode.cl", "// My comment\n", &szKernelLength);

	SourceOpenCL = oclLoadProgSource(source, "// My comment\n", &szKernelLengthFilter);
	szKernelLengthSum = szKernelLength + szKernelLengthFilter;
	char* sourceCL = new char[szKernelLengthSum];
	strcpy(sourceCL,SourceOpenCLShared);
	strcat (sourceCL, SourceOpenCL);
	
	GPUProgram = clCreateProgramWithSource( GPUContext , 1, (const char **)&sourceCL, &szKernelLengthSum, &GPUError);
	CheckError(GPUError);

	// Build the program with 'mad' Optimization option
	char *flags = "-cl-unsafe-math-optimizations -cl-fast-relaxed-math -cl-mad-enable";

	GPUError = clBuildProgram(GPUProgram, 0, NULL, flags, NULL, NULL);
	//error checking code
	if(!GPUError)
	{
		//print kernel compilation error
		char programLog[1024];
		clGetProgramBuildInfo(GPUProgram, cdDevices[0], CL_PROGRAM_BUILD_LOG, 1024, programLog, 0);
		cout<<programLog<<endl;
	}


	cout << kernelFuncName << endl;

	GPUKernel = clCreateKernel(GPUProgram, kernelFuncName, &GPUError);
	CheckError(GPUError);

	

}

GPUBase::GPUBase()
{
	iBlockDimX = 16;
	iBlockDimY = 16;
}

bool GPUBase::CreateBuffersIn(int maxBufferSize, int numbOfBuffers)
{
	numberOfBuffersIn = numbOfBuffers;
	buffersListIn = new cl_mem[numberOfBuffersIn];
	sizeBuffersIn = new int[numberOfBuffersIn];

	for (int i = 0; i < numberOfBuffersIn ; i++)
	{
		buffersListIn[i] = clCreateBuffer(GPUContext, CL_MEM_READ_WRITE, maxBufferSize, NULL, &GPUError);
		CheckError(GPUError);
	}

	
	




	return true;
}

int GPUBase::GetKernelSize(double sigma, double cut_off)
{
	unsigned int i;
	for (i=0;i<MAX_KERNEL_SIZE;i++)
		if (exp(-((double)(i*i))/(2.0*sigma*sigma))<cut_off)
			break;
	unsigned int size = 2*i-1;
	return size;
}


bool GPUBase::CreateBuffersOut( int maxBufferSize, int numbOfBuffers)
{
	numberOfBuffersOut = numbOfBuffers;
	buffersListOut = new cl_mem[numberOfBuffersOut];
	sizeBuffersOut = new int[numberOfBuffersOut];

	for (int i = 0; i < numberOfBuffersOut ; i++)
	{
		buffersListOut[i] = clCreateBuffer(GPUContext, CL_MEM_READ_WRITE, maxBufferSize, NULL, &GPUError);
		CheckError(GPUError);
	}
	return true;
}


void GPUBase::CheckError( int code )
{
	switch(code)
	{
	case CL_SUCCESS:
		return;
		break;
	default:
		cout << "OTHERS ERROR" << endl;
	}
}


bool GPUBase::SendImageToBuffers(IplImage* img, ... )
{
	if(buffersListIn == NULL)
		return false;

	//clock_t start, finish;
	//double duration = 0;
	//start = clock();

		imageHeight = img->height;
		imageWidth = img->width;

		sizeBuffersIn[0] = img->width*img->height*sizeof(float);
		GPUError = clEnqueueWriteBuffer(GPUCommandQueue, buffersListIn[0], CL_TRUE, 0, img->width*img->height*sizeof(float) , (void*)img->imageData, 0, NULL, NULL);
		CheckError(GPUError);

		va_list arg_ptr;
		va_start(arg_ptr, img);
	
		for(int i = 1 ; i<numberOfBuffersIn ; i++)
		{
			IplImage* tmpImg = va_arg(arg_ptr, IplImage*);
			sizeBuffersIn[i] = tmpImg->width*tmpImg->height*sizeof(float);
			GPUError = clEnqueueWriteBuffer(GPUCommandQueue, buffersListIn[i], CL_TRUE, 0, tmpImg->width*tmpImg->height*sizeof(float) , (void*)tmpImg->imageData, 0, NULL, NULL);
			CheckError(GPUError);
		}
		va_end(arg_ptr);



	//finish = clock();
	//duration = (double)(finish - start) / CLOCKS_PER_SEC;
	//cout << "SEND IMG TO GPU: " << endl;
	//cout << duration << endl;
}


bool GPUBase::ReceiveImageData( IplImage* img, ... )
{
	if(buffersListOut == NULL)
		return false;

	sizeBuffersOut[0] = img->width*img->height*sizeof(float);
	GPUError = clEnqueueReadBuffer(GPUCommandQueue, buffersListOut[0], CL_TRUE, 0, img->width*img->height*sizeof(float) , (void*)img->imageData, 0, NULL, NULL);
	CheckError(GPUError);

	va_list arg_ptr;
	va_start(arg_ptr, img);

	for(int i = 1 ; i<numberOfBuffersOut ; i++)
	{
		IplImage* tmpImg = va_arg(arg_ptr, IplImage*);
		sizeBuffersOut[i] = tmpImg->width*tmpImg->height*sizeof(float);
		GPUError = clEnqueueReadBuffer(GPUCommandQueue, buffersListOut[i], CL_TRUE, 0, tmpImg->width*tmpImg->height*sizeof(float) , (void*)tmpImg->imageData, 0, NULL, NULL);
		CheckError(GPUError);
	}
	va_end(arg_ptr);
}


size_t GPUBase::shrRoundUp(int group_size, int global_size)
{
	if(global_size < 80)
		global_size = 80;
	int r = global_size % group_size;
	if(r == 0)
	{
		return global_size;
	} else
	{
		return global_size + group_size - r;
	}
}

char* GPUBase::oclLoadProgSource(const char* cFilename, const char* cPreamble, size_t* szFinalLength)
{
	// locals
	FILE* pFileStream = NULL;
	size_t szSourceLength;


	pFileStream = fopen(cFilename, "rb");
	if(pFileStream == 0)
	{
		return NULL;
	}
	size_t szPreambleLength = strlen(cPreamble);

	// get the length of the source code
	fseek(pFileStream, 0, SEEK_END);
	szSourceLength = ftell(pFileStream);
	fseek(pFileStream, 0, SEEK_SET);

	// allocate a buffer for the source code string and read it in
	char* cSourceString = (char *)malloc(szSourceLength + szPreambleLength + 1);
	memcpy(cSourceString, cPreamble, szPreambleLength);
	if (fread((cSourceString) + szPreambleLength, szSourceLength, 1, pFileStream) != 1)
	{
		fclose(pFileStream);
		free(cSourceString);
		return 0;
	}

	// close the file and return the total length of the combined (preamble + source) string
	fclose(pFileStream);
	if(szFinalLength != 0)
	{
		*szFinalLength = szSourceLength + szPreambleLength;
	}
	cSourceString[szSourceLength + szPreambleLength] = '\0';

	return cSourceString;
}

GPUBase::~GPUBase()
{
	if(GPUCommandQueue)clReleaseCommandQueue(GPUCommandQueue);
	if(GPUContext)clReleaseContext(GPUContext);

	for(int i = 1 ; i<numberOfBuffersOut ; i++)
	{
		if(buffersListOut[i])clReleaseMemObject(buffersListOut[i]);
	}

	for(int i = 1 ; i<numberOfBuffersIn ; i++)
	{
		if(buffersListIn[i])clReleaseMemObject(buffersListIn[i]);
	}
}

















