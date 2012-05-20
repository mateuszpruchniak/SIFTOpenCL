

#include "GPUBase.h"


GPUBase::GPUBase(char* source, char* KernelName)
{
	
	printf("\n ----------- GPUBase START --------------- \n");
	kernelFuncName = KernelName;
	size_t szKernelLength = 0;
	size_t szKernelLengthFilter = 0;
	size_t szKernelLengthSum = 0;
	char* SourceOpenCLShared;
	char* SourceOpenCL;
	iBlockDimX = 16;
	iBlockDimY = 16;

	
	GPUContext = GPU::getInstance().GPUContext;
	GPUCommandQueue = GPU::getInstance().GPUCommandQueue;
	
    	// Load OpenCL kernel
	SourceOpenCLShared = oclLoadProgSource("C:\\Users\\Mati\\Desktop\\Dropbox\\MGR\\SIFTOpenCL\\GPU\\OpenCL\\GPUCode.cl", "// My comment\n", &szKernelLength);

	SourceOpenCL = oclLoadProgSource(source, "// My comment\n", &szKernelLengthFilter);
	
	szKernelLengthSum = szKernelLength + szKernelLengthFilter + 100;
	char* sourceCL = new char[szKernelLengthSum];
	
	
	strcpy(sourceCL,SourceOpenCLShared);
	strcat (sourceCL, SourceOpenCL);
	
	
	GPUProgram = clCreateProgramWithSource( GPUContext , 1, (const char **)&sourceCL, &szKernelLengthSum, &GPUError);
	CheckError(GPUError);

	// Build the program with 'mad' Optimization option
	char *flags = "-cl-unsafe-math-optimizations";

	GPUError = clBuildProgram(GPUProgram, 0, NULL, flags, NULL, NULL);
	CheckError(GPUError);
	
	GPUKernel = clCreateKernel(GPUProgram, kernelFuncName, &GPUError);
	CheckError(GPUError);

	printf("\n ----------- GPUBase END --------------- \n");

}

GPUBase::GPUBase()
{
	iBlockDimX = 16;
	iBlockDimY = 16;
}

bool GPUBase::CreateBuffersIn(int maxBufferSize, int numbOfBuffers)
{
	GPU::getInstance().CreateBuffersIn(maxBufferSize,numbOfBuffers);

	return true;
}

bool GPUBase::CreateBuffersOut( int maxBufferSize, int numbOfBuffers)
{
	GPU::getInstance().CreateBuffersOut(maxBufferSize,numbOfBuffers);
	
	return true;
}



bool GPU::CreateBuffersIn(int maxBufferSize, int numbOfBuffers)
{
	if( maxBufferSize > maxNumberBufIn)
	{
		if(maxNumberBufIn > 0)
		{
			for(int i = 0 ; i < numberOfBuffersIn ; i++)
			{
				if(buffersListIn[i])clReleaseMemObject(buffersListIn[i]);
			}
		}
		
		cout << "Tworzenie bufora In" << endl;
		
		maxNumberBufIn = maxBufferSize;
		numberOfBuffersIn = numbOfBuffers;
		buffersListIn = new cl_mem[numberOfBuffersIn];
	
		for (int i = 0; i < numberOfBuffersIn ; i++)
		{
			buffersListIn[i] = clCreateBuffer(GPUContext, CL_MEM_READ_WRITE, maxBufferSize, NULL, &GPUError);
		}
	}
	return true;
}



bool GPU::CreateBuffersOut( int maxBufferSize, int numbOfBuffers)
{
	if( maxBufferSize > maxNumberBufOut)
	{
		if(maxNumberBufOut > 0)
		{
			for(int i = 0 ; i < numberOfBuffersOut ; i++)
			{
				if(buffersListOut[i])clReleaseMemObject(buffersListOut[i]);
			}
		}
		
		cout << "Tworzenie bufora Out" << endl;
		
		maxNumberBufOut = maxBufferSize;
		numberOfBuffersOut = numbOfBuffers;
		buffersListOut = new cl_mem[numberOfBuffersOut];
	
		for (int i = 0; i < numberOfBuffersOut ; i++)
		{
			buffersListOut[i] = clCreateBuffer(GPUContext, CL_MEM_READ_WRITE, maxBufferSize, NULL, &GPUError);
		}
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





void GPUBase::CheckError( int code )
{
	switch(code)
	{
	case CL_SUCCESS:
		return;
		break;
	default:
		cout << "ERROR : " << code << endl;
	}
}


bool GPUBase::SendImageToBuffers(int number, ... )
{
	if(GPU::getInstance().buffersListIn == NULL)
		return false;


	va_list arg_ptr;
	va_start(arg_ptr, number);
	

	for(int i = 0 ; i < number ; i++)
	{
		IplImage* tmpImg = va_arg(arg_ptr, IplImage*);
		imageHeight = tmpImg->height;
		imageWidth = tmpImg->width;
		GPUError = clEnqueueWriteBuffer(GPUCommandQueue, GPU::getInstance().buffersListIn[i], CL_TRUE, 0, tmpImg->width*tmpImg->height*sizeof(float) , (void*)tmpImg->imageData, 0, NULL, NULL);
		CheckError(GPUError);
	}
	va_end(arg_ptr);
}


bool GPUBase::ReceiveImageData(int number, ... )
{
	if(GPU::getInstance().buffersListOut == NULL)
		return false;

	va_list arg_ptr;
	va_start(arg_ptr, number);

	for(int i = 0 ; i < number ; i++)
	{
		IplImage* tmpImg = va_arg(arg_ptr, IplImage*);
		GPUError = clEnqueueReadBuffer(GPUCommandQueue, GPU::getInstance().buffersListOut[i], CL_TRUE, 0, tmpImg->width*tmpImg->height*sizeof(float) , (void*)tmpImg->imageData, 0, NULL, NULL);
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

	for(int i = 0 ; i<GPU::getInstance().numberOfBuffersOut ; i++)
	{
		if(GPU::getInstance().buffersListOut[i])clReleaseMemObject(GPU::getInstance().buffersListOut[i]);
	}

	for(int i = 0 ; i<GPU::getInstance().numberOfBuffersIn ; i++)
	{
		if(GPU::getInstance().buffersListIn[i])clReleaseMemObject(GPU::getInstance().buffersListIn[i]);
	}
}

















