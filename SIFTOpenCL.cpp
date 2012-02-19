

#include "SIFTOpenCL.h"
#include "imgfeatures.h"
#include "utils.h"
#include "stdafx.h"
#include <cxcore.h>
#include <cv.h>





 SIFTOpenCL::SIFTOpenCL()
 {
	
	 intvls = SIFT_INTVLS;
	 sigma = SIFT_SIGMA;
	 contr_thr = SIFT_CONTR_THR;
	 curv_thr = SIFT_CURV_THR;
	 img_dbl = SIFT_IMG_DBL;
	 descr_width = SIFT_DESCR_WIDTH;
	 descr_hist_bins = SIFT_DESCR_HIST_BINS;


	 meanFilter = new MeanFilter();
	 subtract = new Subtract();
	 detectExt = new DetectExtrema();

 }




 bool SIFTOpenCL::DoSift( IplImage* img )
 {
	IplImage* init_img;

	/* check arguments */
	if( ! img )
		printf( "NULL pointer error, %s, line %d",  __FILE__, __LINE__ );

	if( ! feat )
		printf( "NULL pointer error, %s, line %d",  __FILE__, __LINE__ );

	/* build scale space pyramid; smallest dimension of top level is ~4 pixels */

	init_img = createInitialImg( img, img_dbl, sigma );

	//octvs = log( (float)MIN( init_img->width, init_img->height ) ) / log((float)2) - 2;
	//gauss_pyr = buildGaussPyr( init_img, octvs, intvls, sigma );
	//dog_pyr = buildDogPyr( gauss_pyr, octvs, intvls );
	//storage = cvCreateMemStorage( 0 );

	//features = scaleSpaceExtrema( dog_pyr, octvs, intvls, contr_thr,
	//	curv_thr, storage );

	
	//if(SIFTCPU)
	//{

	//	calcFeatureScales( features, sigma, intvls );

	//	if( img_dbl )
	//		adjustForImgDbl( features );

	//	CalcFeatureOris( features, gauss_pyr );
	//	

	//	clock_t start, finish;
	//	double duration = 0;
	//	start = clock();
	//		compute_descriptors( features, gauss_pyr, descr_width, descr_hist_bins );
	//	finish = clock();
	//	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	//	cout << endl;
	//	cout << "compute_descriptors " << SIFTCPU << ": ";;
	//	cout << duration << endl;
	//	cout << endl;
	//}
	//else
	//{
	//}


	///* sort features by decreasing scale and move from CvSeq to array */
	//cvSeqSort( features, (CvCmpFunc)feature_cmp, NULL );
	//n = features->total;
	//*feat = (feature*)calloc( n, sizeof(feature) );
	//*feat = (feature*)cvCvtSeqToArray( features, *feat, CV_WHOLE_SEQ );
	//for( i = 0; i < n; i++ )
	//{
	//	free( (*feat)[i].feature_data );
	//	(*feat)[i].feature_data = NULL;
	//}

	//cvReleaseMemStorage( &storage );
	//cvReleaseImage( &init_img );
	//release_pyr( &gauss_pyr, octvs, intvls + 3 );
	//release_pyr( &dog_pyr, octvs, intvls + 2 );
	//return n;


	return true;
 }



 /*
 Converts an image to 8-bit grayscale and Gaussian-smooths it.  The image is
 optionally doubled in size prior to smoothing.

 @param img input image
 @param img_dbl if true, image is doubled in size prior to smoothing
 @param sigma total std of Gaussian smoothing
 */
 IplImage* SIFTOpenCL::createInitialImg( IplImage* img, int img_dbl, float sigma )
 {
	 IplImage* gray, * dbl;
	 float sig_diff;

	 gray = convertToGray32( img );
	 if( img_dbl )
	 {
		 sig_diff = sqrt( sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4 );
		 dbl = cvCreateImage( cvSize( img->width*2, img->height*2 ),
			 32, 1 );
		 cvResize( gray, dbl, CV_INTER_CUBIC );

		 /************************ GPU **************************/
		 if(SIFTCPU)
			 cvSmooth( dbl, dbl, CV_GAUSSIAN, 0, 0, sig_diff, sig_diff );
		 else
		 {
			 meanFilter->CreateBuffersIn(dbl->width*dbl->height*sizeof(float),1);
			 meanFilter->CreateBuffersOut(dbl->width*dbl->height*sizeof(float),1);
			 meanFilter->SendImageToBuffers(dbl);
			 meanFilter->Process(sig_diff);
			 meanFilter->ReceiveImageData(dbl);
		 }
		 /************************ GPU **************************/

		 cvReleaseImage( &gray );
		 return dbl;
	 }
	 else
	 {
		 sig_diff = sqrt( sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA );

		 /************************ GPU **************************/
		 if(SIFTCPU)
			 cvSmooth( gray, gray, CV_GAUSSIAN, 0, 0, sig_diff, sig_diff );
		 else
		 {
			 meanFilter->CreateBuffersIn(gray->width*gray->height*sizeof(float),1);
			 meanFilter->CreateBuffersOut(gray->width*gray->height*sizeof(float),1);
			 meanFilter->SendImageToBuffers(gray);
			 meanFilter->Process(sig_diff);
			 meanFilter->ReceiveImageData(gray);
		 }
		 /************************ GPU **************************/

		 return gray;
	 }
 }


 /*
 Converts an image to 32-bit grayscale

 @param img a 3-channel 8-bit color (BGR) or 8-bit gray image

 @return Returns a 32-bit grayscale image
 */
 IplImage* SIFTOpenCL::convertToGray32( IplImage* img )
 {
	 IplImage* gray8, * gray32;

	 gray32 = cvCreateImage( cvGetSize(img), 32, 1 );
	 if( img->nChannels == 1 )
		 gray8 = (IplImage*)cvClone( img );
	 else
	 {
		 gray8 = cvCreateImage( cvGetSize(img), IPL_DEPTH_8U, 1 );
		 cvCvtColor( img, gray8, CV_BGR2GRAY );
	 }
	 cvConvertScale( gray8, gray32, 1.0 / 255.0, 0 );

	 cvReleaseImage( &gray8 );
	 return gray32;
 }