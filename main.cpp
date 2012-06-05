

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
#include "stdafx.h"
#include "SiftGPU.h"
#include "utils.h"
#include "kdtree.h"

using namespace std;

/* the maximum number of keypoint NN candidates to check during BBF search */
#define KDTREE_BBF_MAX_NN_CHKS 200

/* threshold on squared ratio of distances between NN and 2nd NN */
#define NN_SQ_DIST_RATIO_THR 0.49

double SendTime = 0;
double RecvTime = 0;

// The main function!
int main()
{

	int a[10];
	a[0] = 0;
	a[1] = 1;
	a[2] = 2;
	a[3] = 3;
	a[4] = 4;
	a[5] = 5;


	int* b;

	b = &a[2];

	cout << b[1] << endl;




	char* img1_file = "c:\\opel.jpg";
	char* img2_file = "c:\\opel.jpg";
	IplImage* img1, * img2, *stacked;
	char* img_file_name = "c:\\scene2.jpg";
	char* out_file_name  = "c:\\h1.sift";;
	char* out_img_name = "C:\\Users\\Mati\\Pictures\\sceneOut.jpg";
	int display = 1;
	int intvls = SIFT_INTVLS;
	double sigma = SIFT_SIGMA;
	float contr_thr = 0.04;
	int curv_thr = SIFT_CURV_THR;
	int img_dbl = SIFT_IMG_DBL;
	int descr_width = SIFT_DESCR_WIDTH;
	int descr_hist_bins = SIFT_DESCR_HIST_BINS;
	IplImage* img;
	feature* features1;
	feature* features2;
	feature* feat;
	int n1 = 0;
	int n2 = 0;
	feature** nbrs;
	struct kd_node* kd_root;
	int i, k, m = 0;
	double d0, d1;
	CvPoint pt1, pt2;



	SiftGPU* siftOpenCL = new SiftGPU();
	//SIFTOpenCL* siftOpenCL2 = new SIFTOpenCL();

	fprintf( stderr, "Finding SIFT features...\n" );
	
	img1 = cvLoadImage( img1_file, 1 );
	if( ! img1 )
		printf( "unable to load image from %s", img1_file );
	img2 = cvLoadImage( img2_file, 1 );
	if( ! img2 )
		printf( "unable to load image from %s", img2_file );

	stacked = stack_imgs( img1, img2 );

	clock_t start, finish;
	double duration = 0;
	start = clock();
		
		n1 = siftOpenCL->DoSift(img1);
		features1 = siftOpenCL->feat;

		//n2 = siftOpenCL2->DoSift(img2);
		//features2 = siftOpenCL2->feat;

	finish = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	cout << endl;
	cout << "SIFT " << SIFTCPU << ": ";;
	cout << duration << endl;
	cout << "Send time: " << SendTime << endl;
	cout << "Receive time: " << RecvTime << endl;
	cout << "Transfer time (%): " << (SendTime+RecvTime)*100 / duration  << endl;
	cout << endl;
	fprintf( stderr, "Found %d features1.\n", n1 );
	fprintf( stderr, "Found %d features2.\n", n2 );





	//kd_root = kdtree_build( features2, n2 );

	//for(i = 0; i < n1; i++ )
	//{
	//	feat = features1 + i;
	//	k = kdtree_bbf_knn( kd_root, feat, 2, &nbrs, KDTREE_BBF_MAX_NN_CHKS );
	//	if( k == 2 )
	//	{
	//		d0 = descr_dist_sq( feat, nbrs[0] );
	//		d1 = descr_dist_sq( feat, nbrs[1] );
	//		if( d0 < d1 * NN_SQ_DIST_RATIO_THR )
	//		{
	//			pt1 = cvPoint( cvRound( feat->x ), cvRound( feat->y ) );
	//			pt2 = cvPoint( cvRound( nbrs[0]->x ), cvRound( nbrs[0]->y ) );
	//			pt2.y += img1->height;
	//			cvLine( stacked, pt1, pt2, CV_RGB(255,0,255), 1, 8, 0 );
	//			m++;
	//			features1[i].fwd_match = nbrs[0];
	//		}
	//	}
	//	free( nbrs );
	//}

	//fprintf( stderr, "Found %d total matches\n", m );
	//cvNamedWindow( "Matches", 1 );
	//cvShowImage( "Matches", stacked );
	//cvWaitKey( 0 );


	/* 
	UNCOMMENT BELOW TO SEE HOW RANSAC FUNCTION WORKS

	Note that this line above:

	feat1[i].fwd_match = nbrs[0];

	is important for the RANSAC function to work.
	*/
	/*
	{
		CvMat* H;
		H = ransac_xform( feat1, n1, FEATURE_FWD_MATCH, lsq_homog, 4, 0.01,
			homog_xfer_err, 3.0, NULL, NULL );
		if( H )
		{
			IplImage* xformed;
			xformed = cvCreateImage( cvGetSize( img2 ), IPL_DEPTH_8U, 3 );
			cvWarpPerspective( img1, xformed, H, 
				CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS,
				cvScalarAll( 0 ) );
			cvNamedWindow( "Xformed", 1 );
			cvShowImage( "Xformed", xformed );
			cvWaitKey( 0 );
			cvReleaseImage( &xformed );
			cvReleaseMat( &H );
		}
	}
	*/

	//cvReleaseImage( &stacked );
	//cvReleaseImage( &img1 );
	//cvReleaseImage( &img2 );
	//kdtree_release( kd_root );
	//free( feat1 );
	//free( feat2 );
	//return 0;


	
	//SIFTGPU* siftGPU = new SIFTGPU();

	//int tmp = cvRound( 0.5 );

	//IplImage* img1, * img2, * stacked;
	//feature* feat1, * feat2, * feat;
	//feature** nbrs;
	//struct kd_node* kd_root;
	//CvPoint pt1, pt2;
	//double d0, d1;
	//int n1, n2, k, i, m = 0;


	

	//stacked = stack_imgs( img1, img2 );

	//clock_t start, finish;
	//double duration = 0;
	//start = clock();
	//for(int i = 0; i < 1 ; i++)
	//{
	//	fprintf( stderr, "Finding features in %s...\n", img1_file );

	//	n1 = siftGPU->_sift_features( img1, &feat1, intvls, sigma, contr_thr, curv_thr,
	//							img_dbl, descr_width, descr_hist_bins );

	//	//if( out_file_name != NULL )
	//	//	export_features( out_file_name, feat1, n1 );


	//	fprintf( stderr, "Finding features in %s...\n", img2_file );

	//	n2 = siftGPU->_sift_features( img2, &feat2, intvls, sigma, contr_thr, curv_thr,
	//							img_dbl, descr_width, descr_hist_bins );
	//}
	//finish = clock();
	//duration = (double)(finish - start) / CLOCKS_PER_SEC;
	//cout << "SIFT: " << endl;
	//cout << duration << endl;


	//kd_root = kdtree_build( feat2, n2 );

	//for( i = 0; i < n1; i++ )
	//{
	//	feat = feat1 + i;
	//	k = kdtree_bbf_knn( kd_root, feat, 2, &nbrs, KDTREE_BBF_MAX_NN_CHKS );
	//	if( k == 2 )
	//	{
	//		d0 = descr_dist_sq( feat, nbrs[0] );
	//		d1 = descr_dist_sq( feat, nbrs[1] );
	//		if( d0 < d1 * NN_SQ_DIST_RATIO_THR )
	//		{
	//			pt1 = cvPoint( cvRound( feat->x ), cvRound( feat->y ) );
	//			pt2 = cvPoint( cvRound( nbrs[0]->x ), cvRound( nbrs[0]->y ) );
	//			pt2.y += img1->height;
	//			cvLine( stacked, pt1, pt2, CV_RGB(255,0,255), 1, 8, 0 );
	//			m++;
	//			feat1[i].fwd_match = nbrs[0];
	//		}
	//	}
	//	free( nbrs );
	//}

	//fprintf( stderr, "Found %d total matches\n", m );
	//cvNamedWindow( "Matches", 1 );
	//cvShowImage( "Matches", stacked );
	//cvWaitKey( 0 );


	/* 
	UNCOMMENT BELOW TO SEE HOW RANSAC FUNCTION WORKS

	Note that this line above:

	feat1[i].fwd_match = nbrs[0];

	is important for the RANSAC function to work.
	*/
	/*
	{
		CvMat* H;
		H = ransac_xform( feat1, n1, FEATURE_FWD_MATCH, lsq_homog, 4, 0.01,
			homog_xfer_err, 3.0, NULL, NULL );
		if( H )
		{
			IplImage* xformed;
			xformed = cvCreateImage( cvGetSize( img2 ), IPL_DEPTH_8U, 3 );
			cvWarpPerspective( img1, xformed, H, 
				CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS,
				cvScalarAll( 0 ) );
			cvNamedWindow( "Xformed", 1 );
			cvShowImage( "Xformed", xformed );
			cvWaitKey( 0 );
			cvReleaseImage( &xformed );
			cvReleaseMat( &H );
		}
	}
	*/

	//cvReleaseImage( &stacked );
	//cvReleaseImage( &img1 );
	//cvReleaseImage( &img2 );
	//kdtree_release( kd_root );
	//free( feat1 );
	//free( feat2 );
	//return 0;





	return 0;
}



//
//
//int main(int argc, const char** argv)
//{
//
//
//	IplImage* img = cvLoadImage("./img/car.jpg");
//	
//
//	GPUImageProcessor* GPU = new GPUImageProcessor(img->width,img->height,img->nChannels);
//
//
//	
//	GPU->AddProcessing( new Feature("./CL/SIFT.cl",GPU->GPUContext,GPU->Transfer,"ckBuildPyramid") );
//
//
//	GPU->Transfer->CreateBuffers();
//	GPU->Transfer->SendImage(img);
//	GPU->Process();
//	img = GPU->Transfer->ReceiveImage();
//
//	cout << "-------------------------\n\n" << endl;
//
//    cvNamedWindow("sobel", CV_WINDOW_AUTOSIZE); 
//    cvShowImage("sobel", img );
//    cvWaitKey(2);
//
//
//
//	getchar();
//    return 0;
//
//}
//


