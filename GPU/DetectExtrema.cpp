
#include "DetectExtrema.h"



DetectExtrema::~DetectExtrema(void)
{
}

DetectExtrema::DetectExtrema(): GPUBase("C:\\Dropbox\\MGR\\GPUFeatureExtraction\\GPU\\OpenCL\\DetectExtrema.cl","ckDetect")
{

}


bool DetectExtrema::Process( int* num, int* numRej, float prelim_contr_thr, int intvl, int octv, Keys* keys, IplImage* img )
{
	GPUKernel2 = clCreateKernel(GPUProgram, "ckDesc", &GPUError);
	CheckError(GPUError);


	int maxNumberKeys = 1000;



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


	cl_mem cmDevBufNumber = clCreateBuffer(GPUContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &GPUError);
	CheckError(GPUError);
	GPUError = clEnqueueWriteBuffer(GPUCommandQueue, cmDevBufNumber, CL_TRUE, 0, sizeof(int), (void*)num, 0, NULL, NULL);
	CheckError(GPUError);

	cl_mem cmDevBufNumberReject = clCreateBuffer(GPUContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &GPUError);
	CheckError(GPUError);
	GPUError = clEnqueueWriteBuffer(GPUCommandQueue, cmDevBufNumberReject, CL_TRUE, 0, sizeof(int), (void*)numRej, 0, NULL, NULL);
	CheckError(GPUError);

	int count = 0;
	cl_mem cmDevBufCount = clCreateBuffer(GPUContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &GPUError);
	CheckError(GPUError);
	GPUError = clEnqueueWriteBuffer(GPUCommandQueue, cmDevBufCount, CL_TRUE, 0, sizeof(int), (void*)&count, 0, NULL, NULL);
	CheckError(GPUError);

	cl_mem cmDevBufKeys = clCreateBuffer(GPUContext, CL_MEM_READ_WRITE, maxNumberKeys*sizeof(Keys), NULL, &GPUError);
	CheckError(GPUError);
	GPUError = clEnqueueWriteBuffer(GPUCommandQueue, cmDevBufKeys, CL_TRUE, 0,maxNumberKeys*sizeof(Keys), (void*)keys, 0, NULL, NULL);
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
	GPUError |= clSetKernelArg(GPUKernel, 4, sizeof(cl_mem), (void*)&cmDevBufCount);
	GPUError |= clSetKernelArg(GPUKernel, 5, sizeof(cl_mem), (void*)&cmDevBufKeys);
	GPUError |= clSetKernelArg(GPUKernel, 6, sizeof(cl_uint), (void*)&imageWidth);
	GPUError |= clSetKernelArg(GPUKernel, 7, sizeof(cl_uint), (void*)&imageHeight);
	GPUError |= clSetKernelArg(GPUKernel, 8, sizeof(cl_float), (void*)&prelim_contr_thr);
	GPUError |= clSetKernelArg(GPUKernel, 9, sizeof(cl_uint), (void*)&intvl);
	GPUError |= clSetKernelArg(GPUKernel, 10, sizeof(cl_uint), (void*)&octv);
	GPUError |= clSetKernelArg(GPUKernel, 11, sizeof(cl_mem), (void*)&cmDevBufNumber);
	GPUError |= clSetKernelArg(GPUKernel, 12, sizeof(cl_mem), (void*)&cmDevBufNumberReject);
	if(GPUError) return false;


	if(clEnqueueNDRangeKernel( GPUCommandQueue, GPUKernel, 2, NULL, GPUGlobalWorkSize, GPULocalWorkSize, 0, NULL, NULL)) return false;


	GPUError = clEnqueueReadBuffer(GPUCommandQueue, cmDevBufNumber, CL_TRUE, 0, sizeof(int), (void*)num, 0, NULL, NULL);
	CheckError(GPUError);

	//GPUError = clEnqueueReadBuffer(GPUCommandQueue, cmDevBufNumberReject, CL_TRUE, 0, sizeof(int), (void*)numRej, 0, NULL, NULL);
	//CheckError(GPUError);

	//GPUError = clEnqueueReadBuffer(GPUCommandQueue, cmDevBufKeys, CL_TRUE, 0, maxNumberKeys*sizeof(Keys), (void*)keys, 0, NULL, NULL);
	//CheckError(GPUError);


	*numRej = 0;


	GPUError = clEnqueueWriteBuffer(GPUCommandQueue, cmDevBufNumber, CL_TRUE, 0, sizeof(int), (void*)num, 0, NULL, NULL);
	CheckError(GPUError);

	GPUError = clEnqueueWriteBuffer(GPUCommandQueue, cmDevBufNumberReject, CL_TRUE, 0, sizeof(int), (void*)numRej, 0, NULL, NULL);
	CheckError(GPUError);


	//GPUError = clEnqueueWriteBuffer(GPUCommandQueue, cmDevBufKeys, CL_TRUE, 0,maxNumberKeys*sizeof(Keys), (void*)keys, 0, NULL, NULL);
	//CheckError(GPUError);



	cl_image_format volume_format;
	volume_format.image_channel_order = CL_R;
	volume_format.image_channel_data_type = CL_FLOAT;

	cl_mem d_volume = clCreateImage2D(GPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &volume_format, imageWidth, imageHeight, imageWidth*sizeof(float), (float*)img->imageData, &GPUError);
	CheckError(GPUError);




	GPULocalWorkSize[0] = iBlockDimX;
	GPULocalWorkSize[1] = iBlockDimY;
	GPUGlobalWorkSize[0] = shrRoundUp((int)GPULocalWorkSize[0], (int)*num);
	GPUGlobalWorkSize[1] = shrRoundUp((int)GPULocalWorkSize[1], (int)1);



	iLocalPixPitch = iBlockDimX + 2;
	GPUError |= clSetKernelArg(GPUKernel2, 0, sizeof(cl_mem), (void*)&d_volume);
	GPUError |= clSetKernelArg(GPUKernel2, 1, sizeof(cl_mem), (void*)&buffersListOut[0]);
	GPUError |= clSetKernelArg(GPUKernel2, 2, sizeof(cl_mem), (void*)&cmDevBufCount);
	GPUError |= clSetKernelArg(GPUKernel2, 3, sizeof(cl_mem), (void*)&cmDevBufKeys);
	GPUError |= clSetKernelArg(GPUKernel2, 4, sizeof(cl_uint), (void*)&imageWidth);
	GPUError |= clSetKernelArg(GPUKernel2, 5, sizeof(cl_uint), (void*)&imageHeight);
	GPUError |= clSetKernelArg(GPUKernel2, 6, sizeof(cl_float), (void*)&prelim_contr_thr);
	GPUError |= clSetKernelArg(GPUKernel2, 7, sizeof(cl_uint), (void*)&intvl);
	GPUError |= clSetKernelArg(GPUKernel2, 8, sizeof(cl_uint), (void*)&octv);
	GPUError |= clSetKernelArg(GPUKernel2, 9, sizeof(cl_mem), (void*)&cmDevBufNumber);
	GPUError |= clSetKernelArg(GPUKernel2, 10, sizeof(cl_mem), (void*)&cmDevBufNumberReject);
	if(GPUError) return false;

	if(clEnqueueNDRangeKernel( GPUCommandQueue, GPUKernel2, 2, NULL, GPUGlobalWorkSize, GPULocalWorkSize, 0, NULL, NULL)) 
		cout << "Error clEnqueueNDRangeKernel" << endl;


	//GPUError = clEnqueueReadBuffer(GPUCommandQueue, buffersListOut[0], CL_TRUE, 0, img->width*img->height*sizeof(float) , (void*)img->imageData, 0, NULL, NULL);
	//CheckError(GPUError);

	//cvNamedWindow( "ggg", 1 );
	//cvShowImage( "ggg", img );
	//cvWaitKey(0);



	GPUError = clEnqueueReadBuffer(GPUCommandQueue, cmDevBufNumber, CL_TRUE, 0, sizeof(int), (void*)num, 0, NULL, NULL);
	CheckError(GPUError);
	GPUError = clEnqueueReadBuffer(GPUCommandQueue, cmDevBufNumberReject, CL_TRUE, 0, sizeof(int), (void*)numRej, 0, NULL, NULL);
	CheckError(GPUError);

	GPUError = clEnqueueReadBuffer(GPUCommandQueue, cmDevBufCount, CL_TRUE, 0, sizeof(int), (void*)&count, 0, NULL, NULL);
	CheckError(GPUError);

	GPUError = clEnqueueReadBuffer(GPUCommandQueue, cmDevBufKeys, CL_TRUE, 0, maxNumberKeys*sizeof(Keys), (void*)keys, 0, NULL, NULL);
	CheckError(GPUError);

	//for(int i = 0; i < *num; i++)
	//{
	//	for(int j = 0; j < 128 ; j++)
	//	{
	//		if( keys[i].desc[j] != keys2[i].desc[j] )
	//		{
	//			cout << "dofferent" << endl;
	//		}
	//	}
	//	//cout << "ori GPU: " << keys[i].x << endl;
	//	//cout << "mag GPU: " << keys[i].y << endl;
	//	//cout << endl;
	//}
	//cout << "Number GPU: " << *num << endl;













	return true;
}