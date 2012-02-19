/*
Functions for detecting SIFT image features.

For more information, refer to:

Lowe, D.  Distinctive image features from scale-invariant keypoints.
<EM>International Journal of Computer Vision, 60</EM>, 2 (2004),
pp.91--110.

Copyright (C) 2006-2010  Rob Hess <hess@eecs.oregonstate.edu>

Note: The SIFT algorithm is patented in the United States and cannot be
used in commercial products without a license from the University of
British Columbia.  For more information, refer to the file LICENSE.ubc
that accompanied this distribution.

@version 1.1.2-20100521
*/

#include "sift2.h"
#include "imgfeatures.h"
#include "utils.h"

#include "stdafx.h"

#include <cxcore.h>
#include <cv.h>



/************************* Local Function Prototypes *************************/


 IplImage* convert_to_gray32( IplImage* );

 IplImage* downsample( IplImage* );


 int is_extremum( IplImage***, int, int, int, int );
 feature* interp_extremum( IplImage***, int, int, int, int, int, float);
 void interp_step( IplImage***, int, int, int, int, float*, float*, float* );
 CvMat* deriv_3D( IplImage***, int, int, int, int );
 void hessian_3D( IplImage***, int, int, int, int, float H[][3] );
 float interp_contr( IplImage***, int, int, int, int, float, float, float );
 feature* new_feature( void );
 int is_too_edge_like( IplImage*, int, int, int );



 void ori_hist( IplImage*, int, int, int, int, float, float * );
 int calc_grad_mag_ori( IplImage*, int, int, float*, float* );
 void smooth_ori_hist( float*, int );
 float dominant_ori( float*, int , int*);
 void add_good_ori_features( CvSeq*, float*, int, float, feature*, int );
 feature* clone_feature( feature* );
 void compute_descriptors( CvSeq*, IplImage***, int, int );
 float*** descr_hist( IplImage*, int, int, float, float, int, int );
 void interp_hist_entry( float***, float, float, float, float, int, int);
 void hist_to_descr( float***, int, int, feature* );
 void normalize_descr( feature* );
 int feature_cmp( void*, void*, void* );
 void release_descr_hist( float****, int );
 void release_pyr( IplImage****, int, int );


 void ckDetect( float* dataIn1,  float* dataIn2,  float* dataIn3,   float* gauss_pyr,  float* ucDest,
						 int* numberExtrema,  float* keys,
						int ImageWidth, int ImageHeight, float prelim_contr_thr, int intvl, int octv,  int* number,  int* numberRej, int pozX, int pozY);

/*********************** Functions prototyped in sift.h **********************/


/**
Finds SIFT features in an image using default parameter values.  All
detected features are stored in the array pointed to by \a feat.

@param img the image in which to detect features
@param feat a pointer to an array in which to store detected features

@return Returns the number of features stored in \a feat or -1 on failure
@see _sift_features()
*/
int SIFTGPU::sift_features( IplImage* img, feature** feat )
{
	return _sift_features( img, feat, SIFT_INTVLS, SIFT_SIGMA, SIFT_CONTR_THR,
							SIFT_CURV_THR, SIFT_IMG_DBL, SIFT_DESCR_WIDTH,
							SIFT_DESCR_HIST_BINS );
}



/**
Finds SIFT features in an image using user-specified parameter values.  All
detected features are stored in the array pointed to by \a feat.

@param img the image in which to detect features
@param fea a pointer to an array in which to store detected features
@param intvls the number of intervals sampled per octave of scale space
@param sigma the amount of Gaussian smoothing applied to each image level
	before building the scale space representation for an octave
@param cont_thr a threshold on the value of the scale space function
	\f$\left|D(\hat{x})\right|\f$, where \f$\hat{x}\f$ is a vector specifying
	feature location and scale, used to reject unstable features;  assumes
	pixel values in the range [0, 1]
@param curv_thr threshold on a feature's ratio of principle curvatures
	used to reject features that are too edge-like
@param img_dbl should be 1 if image doubling prior to scale space
	construction is desired or 0 if not
@param descr_width the width, \f$n\f$, of the \f$n \times n\f$ array of
	orientation histograms used to compute a feature's descriptor
@param descr_hist_bins the number of orientations in each of the
	histograms in the array used to compute a feature's descriptor

@return Returns the number of keypoints stored in \a feat or -1 on failure
@see sift_keypoints()
*/
int SIFTGPU::_sift_features( IplImage* img, feature** feat, int intvls,
				   float sigma, float contr_thr, int curv_thr,
				   int img_dbl, int descr_width, int descr_hist_bins )
{
	IplImage* init_img;
	IplImage*** dog_pyr;
	CvMemStorage* storage;
	CvSeq* features;
	int octvs, i, n = 0;

	/* check arguments */
	if( ! img )
		printf( "NULL pointer error, %s, line %d",  __FILE__, __LINE__ );

	if( ! feat )
		printf( "NULL pointer error, %s, line %d",  __FILE__, __LINE__ );

	/* build scale space pyramid; smallest dimension of top level is ~4 pixels */

	init_img = createInitImg( img, img_dbl, sigma );
	octvs = log( (float)MIN( init_img->width, init_img->height ) ) / log((float)2) - 2;
	gauss_pyr = buildGaussPyr( init_img, octvs, intvls, sigma );
	dog_pyr = buildDogPyr( gauss_pyr, octvs, intvls );
	storage = cvCreateMemStorage( 0 );

	features = scaleSpaceExtrema( dog_pyr, octvs, intvls, contr_thr,
		curv_thr, storage );

	
	if(SIFTCPU)
	{

		calcFeatureScales( features, sigma, intvls );

		if( img_dbl )
			adjustForImgDbl( features );

		calc_feature_oris( features, gauss_pyr );
		

		clock_t start, finish;
		double duration = 0;
		start = clock();
			compute_descriptors( features, gauss_pyr, descr_width, descr_hist_bins );
		finish = clock();
		duration = (double)(finish - start) / CLOCKS_PER_SEC;
		cout << endl;
		cout << "compute_descriptors " << SIFTCPU << ": ";;
		cout << duration << endl;
		cout << endl;
	}
	else
	{
	}

	//feature* feat2 = CV_GET_SEQ_ELEM(feature, features, 0 );

	//struct detection_data* ddata = feat_detection_data( feat2 );

	/*cout << "CPU total: " <<  features->total << endl;


	clock_t start, finish;
	double duration = 0;
	start = clock();
		compute_descriptors( features, gauss_pyr, descr_width, descr_hist_bins );
	finish = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	cout << endl;
	cout << "compute_descriptors " << SIFTCPU << ": ";;
	cout << duration << endl;
	cout << endl;
	
	*/



	/* sort features by decreasing scale and move from CvSeq to array */
	cvSeqSort( features, (CvCmpFunc)feature_cmp, NULL );
	n = features->total;
	*feat = (feature*)calloc( n, sizeof(feature) );
	*feat = (feature*)cvCvtSeqToArray( features, *feat, CV_WHOLE_SEQ );
	for( i = 0; i < n; i++ )
	{
		free( (*feat)[i].feature_data );
		(*feat)[i].feature_data = NULL;
	}

	cvReleaseMemStorage( &storage );
	cvReleaseImage( &init_img );
	release_pyr( &gauss_pyr, octvs, intvls + 3 );
	release_pyr( &dog_pyr, octvs, intvls + 2 );
	return n;
}


/************************ Functions prototyped here **************************/

/*
Converts an image to 8-bit grayscale and Gaussian-smooths it.  The image is
optionally doubled in size prior to smoothing.

@param img input image
@param img_dbl if true, image is doubled in size prior to smoothing
@param sigma total std of Gaussian smoothing
*/
 IplImage* SIFTGPU::createInitImg( IplImage* img, int img_dbl, float sigma )
{
	IplImage* gray, * dbl;
	float sig_diff;

	gray = convert_to_gray32( img );
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
 IplImage* convert_to_gray32( IplImage* img )
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



/*
Builds Gaussian scale space pyramid from an image

@param base base image of the pyramid
@param octvs number of octaves of scale space
@param intvls number of intervals per octave
@param sigma amount of Gaussian smoothing per octave

@return Returns a Gaussian scale space pyramid as an octvs x (intvls + 3) array
*/
 IplImage*** SIFTGPU::buildGaussPyr( IplImage* base, int octvs,
									int intvls, float sigma )
{
	float* sig = (float*)calloc( intvls + 3, sizeof(float));
	float sig_total, sig_prev, k;
	int i, o;

	gauss_pyr = (IplImage***)calloc( octvs, sizeof( IplImage** ) );
	for( i = 0; i < octvs; i++ )
		gauss_pyr[i] = (IplImage**)calloc( intvls + 3, sizeof( IplImage* ) );

	/*
		precompute Gaussian sigmas using the following formula:

		\sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
	*/

	sig[0] = sigma;
	k = pow( 2.0, 1.0 / intvls );
	for( i = 1; i < intvls + 3; i++ )
	{
		sig_prev = pow( k, i - 1 ) * sigma;
		sig_total = sig_prev * k;
		sig[i] = sqrt( sig_total * sig_total - sig_prev * sig_prev );
	}

	for( o = 0; o < octvs; o++ )
		for( i = 0; i < intvls + 3; i++ )
		{
			if( o == 0  &&  i == 0 )
				gauss_pyr[o][i] = cvCloneImage(base);

			/* base of new octvave is halved image from end of previous octave */
			else if( i == 0 )
			{
				
				gauss_pyr[o][i] = downsample( gauss_pyr[o-1][intvls] );

			}

			/* blur the current octave's last image to create the next one */
			else
			{
				gauss_pyr[o][i] = cvCreateImage( cvGetSize(gauss_pyr[o][i-1]),
					32, 1 );

				/************************ GPU **************************/
				if(SIFTCPU)
					cvSmooth( gauss_pyr[o][i-1], gauss_pyr[o][i],CV_GAUSSIAN, 0, 0, sig[i], sig[i] );
				else
				{
					meanFilter->CreateBuffersIn(gauss_pyr[o][i-1]->width*gauss_pyr[o][i-1]->height*sizeof(float),1);
					meanFilter->CreateBuffersOut(gauss_pyr[o][i]->width*gauss_pyr[o][i]->height*sizeof(float),1);
					meanFilter->SendImageToBuffers(gauss_pyr[o][i-1]);
					meanFilter->Process(sig[i]);
					meanFilter->ReceiveImageData(gauss_pyr[o][i]);
				}
				/************************ GPU **************************/
				
			}
		}

	free( sig );
	return gauss_pyr;
}



/*
Downsamples an image to a quarter of its size (half in each dimension)
using nearest-neighbor interpolation

@param img an image

@return Returns an image whose dimensions are half those of img
*/
 IplImage* downsample( IplImage* img )
{
	int width = img->width / 2;
	int height = img->height / 2;

	if( width < 50 || height < 50 )
	{
		width = width*2;
		height = height*2;
	}
	IplImage* smaller = cvCreateImage( cvSize( width, height),
		img->depth, img->nChannels );
	cvResize( img, smaller, CV_INTER_NN );

	return smaller;
}



/*
Builds a difference of Gaussians scale space pyramid by subtracting adjacent
intervals of a Gaussian pyramid

@param gauss_pyr Gaussian scale-space pyramid
@param octvs number of octaves of scale space
@param intvls number of intervals per octave

@return Returns a difference of Gaussians scale space pyramid as an
	octvs x (intvls + 2) array
*/
 IplImage*** SIFTGPU::buildDogPyr( IplImage*** gauss_pyr, int octvs, int intvls )
{
	IplImage*** dog_pyr;
	int i, o;

	dog_pyr = (IplImage***)calloc( octvs, sizeof( IplImage** ) );
	for( i = 0; i < octvs; i++ )
		dog_pyr[i] = (IplImage**)calloc( intvls + 2, sizeof(IplImage*) );

	for( o = 0; o < octvs; o++ )
		for( i = 0; i < intvls + 2; i++ )
		{
			/*cvNamedWindow( "sub1", 1 );
			cvShowImage( "sub1", gauss_pyr[o][i+1] );
			cvWaitKey( 0 );*/

			dog_pyr[o][i] = cvCreateImage( cvGetSize(gauss_pyr[o][i]),
				32, 1 );

			/************************ GPU **************************/
			if(SIFTCPU)
				cvSub( gauss_pyr[o][i+1], gauss_pyr[o][i], dog_pyr[o][i], NULL );
			else
			{
				subtract->CreateBuffersIn(gauss_pyr[o][i+1]->width*gauss_pyr[o][i+1]->height*sizeof(float),2);
				subtract->CreateBuffersOut(gauss_pyr[o][i]->width*gauss_pyr[o][i]->height*sizeof(float),1);
				subtract->SendImageToBuffers(gauss_pyr[o][i+1],gauss_pyr[o][i]);
				subtract->Process();
				subtract->ReceiveImageData(dog_pyr[o][i]);
			}
			/************************ GPU **************************/

			/*cvNamedWindow( "sub", 1 );
			cvShowImage( "sub", dog_pyr[o][i] );
			cvWaitKey( 0 );*/
			
		}

	return dog_pyr;
}


/*
Detects features at extrema in DoG scale space.  Bad features are discarded
based on contrast and ratio of principal curvatures.

@param dog_pyr DoG scale space pyramid
@param octvs octaves of scale space represented by dog_pyr
@param intvls intervals per octave
@param contr_thr low threshold on feature contrast
@param curv_thr high threshold on feature ratio of principal curvatures
@param storage memory storage in which to store detected features

@return Returns an array of detected features whose scales, orientations,
	and descriptors are yet to be determined.
*/
 CvSeq* SIFTGPU::scaleSpaceExtrema( IplImage*** dog_pyr, int octvs, int intvls,
								   float contr_thr, int curv_thr,
								   CvMemStorage* storage )
{
	CvSeq* features;
	float prelim_contr_thr = 0.5 * contr_thr / intvls;
	feature* feat;
	struct detection_data* ddata;
	int o, i, r, c;
	int num=0;				// Number of keypoins detected
	int numRemoved=0;		// The number of key points rejected because they failed a test

	Keys keys[1000];

	int numberExtrema = 0;
	int number = 0;
	int numberRej = 0;
	
	IplImage* img = cvCreateImage( cvGetSize(dog_pyr[0][0]), 32, 1 );

	cvZero(img);
	iteratorFGPU = 0;

	features = cvCreateSeq( 0, sizeof(CvSeq), sizeof(feature), storage );


	/************************ GPU **************************/
	detectExt->CreateBuffersIn(dog_pyr[0][0]->width*dog_pyr[0][0]->height*sizeof(float),4);
	detectExt->CreateBuffersOut(img->width*img->height*sizeof(float),1);
	/************************ GPU **************************/



	clock_t start, finish;
	double duration = 0;
	start = clock();

	for( o = 0; o < octvs; o++ )
		for( i = 1; i <= intvls; i++ )
		{
			

			/************************ GPU **************************/
			if(SIFTCPU)
			{

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
					keys[i].mag = 0.0;
					keys[i].ori = 0.0;
				}

				IplImage* img = cvCreateImage( cvGetSize(dog_pyr[o][i]), 32, 1 );
				cvZero(img);
				
				int numberExtrema = 0;
				int number = 0;
				int numberRej = 0;

				for(r = SIFT_IMG_BORDER; r < dog_pyr[o][0]->height-SIFT_IMG_BORDER; r++)
				for(c = SIFT_IMG_BORDER; c < dog_pyr[o][0]->width-SIFT_IMG_BORDER; c++)
					/* perform preliminary check on contrast */
				{
					
						if( ABS( pixval32f( dog_pyr[o][i], r, c ) ) > prelim_contr_thr )
						{
							if( is_extremum( dog_pyr, o, i, r, c ) )
							{

								feat = interp_extremum(dog_pyr, o, i, r, c, intvls, contr_thr);
								if( feat )
								{
									ddata = feat_detection_data( feat );

									if( ! is_too_edge_like( dog_pyr[ddata->octv][ddata->intvl],
										ddata->r, ddata->c, curv_thr ) )
									{
										num++;
										cvSeqPush( features, feat );
									}
									else
										free( ddata );
									free( feat );
								}
							}
						}
					
				}
			}
			else 
			{
				num = 0;
				//cvNamedWindow( "WWW", 1 );
				//cvShowImage( "WWW",  gauss_pyr[o][i] );
				//cvWaitKey(0);
				
				detectExt->SendImageToBuffers(dog_pyr[o][i-1],dog_pyr[o][i],dog_pyr[o][i+1], gauss_pyr[o][i]);
				detectExt->Process(&num, &numRemoved, prelim_contr_thr, i, o, keys, gauss_pyr[o][i]);
				//detectExt->ReceiveImageData(img);
				
				number = num;

				struct detection_data* ddata;

				for(int ik = 0; ik < number ; ik++)
				{ 
					feat = new_feature();
					ddata = feat_detection_data( feat );
					feat->img_pt.x = feat->x = keys[ik].scx;
					feat->img_pt.y = feat->y = keys[ik].scy;
					ddata->r = keys[ik].y;
					ddata->c = keys[ik].x;
					ddata->subintvl = keys[ik].subintvl;
					ddata->octv = keys[ik].octv;
					ddata->intvl = keys[ik].intvl;
					feat->scl = keys[ik].scl;
					ddata->scl_octv = keys[ik].scl_octv;
					feat->ori = keys[ik].ori;
					feat->d = 128;

					for(int i = 0; i < 128 ; i++ )
					{
						feat->descr[i] = keys[ik].desc[i];
					}

					cvSeqPush( features, feat );
					free( feat );
				}
				

			}
			/************************ GPU **************************/
		}

		finish = clock();
		duration = (double)(finish - start) / CLOCKS_PER_SEC;
		cout << endl;
		cout << "SIFT netto: " << endl;
		cout << duration << endl;
		cout << endl;
	return features;
}






/*
Determines whether a pixel is a scale-space extremum by comparing it to it's
3x3x3 pixel neighborhood.

@param dog_pyr DoG scale space pyramid
@param octv pixel's scale space octave
@param intvl pixel's within-octave interval
@param r pixel's image row
@param c pixel's image col

@return Returns 1 if the specified pixel is an extremum (max or min) among
	it's 3x3x3 pixel neighborhood.
*/
 int is_extremum( IplImage*** dog_pyr, int octv, int intvl, int r, int c )
{
	float val = pixval32f( dog_pyr[octv][intvl], r, c );
	int i, j, k;

	/* check for maximum */
	if( val > 0 )
	{
		for( i = -1; i <= 1; i++ )
			for( j = -1; j <= 1; j++ )
				for( k = -1; k <= 1; k++ )
					if( val < pixval32f( dog_pyr[octv][intvl+i], r + j, c + k ) )
						return 0;
	}

	/* check for minimum */
	else
	{
		for( i = -1; i <= 1; i++ )
			for( j = -1; j <= 1; j++ )
				for( k = -1; k <= 1; k++ )
					if( val > pixval32f( dog_pyr[octv][intvl+i], r + j, c + k ) )
						return 0;
	}



	return 1;
}



/*
Interpolates a scale-space extremum's location and scale to subpixel
accuracy to form an image feature.  Rejects features with low contrast.
Based on Section 4 of Lowe's paper.  

@param dog_pyr DoG scale space pyramid
@param octv feature's octave of scale space
@param intvl feature's within-octave interval
@param r feature's image row
@param c feature's image column
@param intvls total intervals per octave
@param contr_thr threshold on feature contrast

@return Returns the feature resulting from interpolation of the given
	parameters or NULL if the given location could not be interpolated or
	if contrast at the interpolated loation was too low.  If a feature is
	returned, its scale, orientation, and descriptor are yet to be determined.
*/
 feature* interp_extremum( IplImage*** dog_pyr, int octv, int intvl,
										int r, int c, int intvls, float contr_thr )
{
	feature* feat;
	struct detection_data* ddata;
	float xi, xr, xc, contr;
	int i = 0;

	if( c == 668 )
		i = 0;

	while( i < SIFT_MAX_INTERP_STEPS )
	{
		interp_step( dog_pyr, octv, intvl, r, c, &xi, &xr, &xc );
		if( ABS( xi ) < 0.5  &&  ABS( xr ) < 0.5  &&  ABS( xc ) < 0.5 )
			break;

		c += cvRound( xc );
		r += cvRound( xr );
		intvl += cvRound( xi );

		if( intvl < 1  ||
			intvl > intvls  ||
			c < SIFT_IMG_BORDER  ||
			r < SIFT_IMG_BORDER  ||
			c >= dog_pyr[octv][0]->width - SIFT_IMG_BORDER  ||
			r >= dog_pyr[octv][0]->height - SIFT_IMG_BORDER )
		{
			return NULL;
		}

		i++;
	}

	/* ensure convergence of interpolation */
	if( i >= SIFT_MAX_INTERP_STEPS )
		return NULL;

	contr = interp_contr( dog_pyr, octv, intvl, r, c, xi, xr, xc );
	if( ABS( contr ) < contr_thr / intvls )
		return NULL;

	feat = new_feature();
	ddata = feat_detection_data( feat );
	feat->img_pt.x = feat->x = ( c + xc ) * pow( 2.0, octv );
	feat->img_pt.y = feat->y = ( r + xr ) * pow( 2.0, octv );
	ddata->r = r;
	ddata->c = c;
	ddata->octv = octv;
	ddata->intvl = intvl;
	ddata->subintvl = xi;

	return feat;
}



/*
Performs one step of extremum interpolation.  Based on Eqn. (3) in Lowe's
paper.

@param dog_pyr difference of Gaussians scale space pyramid
@param octv octave of scale space
@param intvl interval being interpolated
@param r row being interpolated
@param c column being interpolated
@param xi output as interpolated subpixel increment to interval
@param xr output as interpolated subpixel increment to row
@param xc output as interpolated subpixel increment to col
*/

 void interp_step( IplImage*** dog_pyr, int octv, int intvl, int rr, int cc,
						 float* xi, float* xr, float* xc )
{
	CvMat* dD, X;
	float x[3] = { 0, 0 , 0 };
	float xx[3] = { 0, 0 , 0 };

	float H[3][3];
	float H_inv[3][3];

	dD = deriv_3D( dog_pyr, octv, intvl, rr, cc );
	hessian_3D( dog_pyr, octv, intvl, rr, cc, H);

	float a = H[0][0];
	float b = H[0][1];
	float c = H[0][2];
	float d = H[1][0];
	float e = H[1][1];
	float f = H[1][2];
	float g = H[2][0];
	float h = H[2][1];
	float k = H[2][2];

	float det = a*(e*k - f*h) + b*(f*g - k*d) + c*(d*h - e*g);
	float det_inv = 1.0 / det;

	H_inv[0][0] = (e*k - f*h)*det_inv;
	H_inv[0][1] = (c*h - b*k)*det_inv;
	H_inv[0][2] = (b*f - c*e)*det_inv;

	H_inv[1][0] = (f*g - d*k)*det_inv;
	H_inv[1][1] = (a*k - c*g)*det_inv;
	H_inv[1][2] = (c*d - a*f)*det_inv;

	H_inv[2][0] = (d*h - e*g)*det_inv;
	H_inv[2][1] = (g*b - a*h)*det_inv;
	H_inv[2][2] = (a*e - b*d)*det_inv;

	//cvInitMatHeader( &X, 3, 1, CV_64FC1, x, CV_AUTOSTEP );
	//cvGEMM( H_inv, dD, -1, NULL, 0, &X, 0 );

	*xc = (-1)*( H_inv[0][0]*cvGetReal2D(dD, 0, 0) + H_inv[1][0]*cvGetReal2D(dD, 1, 0) + H_inv[2][0]*cvGetReal2D(dD, 2, 0));
	*xr = (-1)*( H_inv[0][1]*cvGetReal2D(dD, 0, 0) + H_inv[1][1]*cvGetReal2D(dD, 1, 0) + H_inv[2][1]*cvGetReal2D(dD, 2, 0));
	*xi = (-1)*( H_inv[0][2]*cvGetReal2D(dD, 0, 0) + H_inv[1][2]*cvGetReal2D(dD, 1, 0) + H_inv[2][2]*cvGetReal2D(dD, 2, 0));

	cvReleaseMat( &dD );
}



/*
Computes the partial derivatives in x, y, and scale of a pixel in the DoG
scale space pyramid.

@param dog_pyr DoG scale space pyramid
@param octv pixel's octave in dog_pyr
@param intvl pixel's interval in octv
@param r pixel's image row
@param c pixel's image col

@return Returns the vector of partial derivatives for pixel I
	{ dI/dx, dI/dy, dI/ds }^T as a CvMat*
*/
 CvMat* deriv_3D( IplImage*** dog_pyr, int octv, int intvl, int r, int c )
{
	CvMat* dI;
	float dx, dy, ds;

	dx = ( pixval32f( dog_pyr[octv][intvl], r, c+1 ) -
		pixval32f( dog_pyr[octv][intvl], r, c-1 ) ) / 2.0;
	dy = ( pixval32f( dog_pyr[octv][intvl], r+1, c ) -
		pixval32f( dog_pyr[octv][intvl], r-1, c ) ) / 2.0;
	ds = ( pixval32f( dog_pyr[octv][intvl+1], r, c ) -
		pixval32f( dog_pyr[octv][intvl-1], r, c ) ) / 2.0;

	dI = cvCreateMat( 3, 1, CV_64FC1 );
	cvmSet( dI, 0, 0, dx );
	cvmSet( dI, 1, 0, dy );
	cvmSet( dI, 2, 0, ds );

	return dI;
}



/*
Computes the 3D Hessian matrix for a pixel in the DoG scale space pyramid.

@param dog_pyr DoG scale space pyramid
@param octv pixel's octave in dog_pyr
@param intvl pixel's interval in octv
@param r pixel's image row
@param c pixel's image col

@return Returns the Hessian matrix (below) for pixel I as a CvMat*

	/ Ixx  Ixy  Ixs \ <BR>
	| Ixy  Iyy  Iys | <BR>
	\ Ixs  Iys  Iss /
*/
void hessian_3D( IplImage*** dog_pyr, int octv, int intvl, int r, int c, float H[][3] )
{
	float v, dxx, dyy, dss, dxy, dxs, dys;

	v = pixval32f( dog_pyr[octv][intvl], r, c );
	dxx = ( pixval32f( dog_pyr[octv][intvl], r, c+1 ) + 
			pixval32f( dog_pyr[octv][intvl], r, c-1 ) - 2 * v );

	dyy = ( pixval32f( dog_pyr[octv][intvl], r+1, c ) +
			pixval32f( dog_pyr[octv][intvl], r-1, c ) - 2 * v );

	dss = ( pixval32f( dog_pyr[octv][intvl+1], r, c ) +
			pixval32f( dog_pyr[octv][intvl-1], r, c ) - 2 * v );

	dxy = ( pixval32f( dog_pyr[octv][intvl], r+1, c+1 ) -
			pixval32f( dog_pyr[octv][intvl], r+1, c-1 ) -
			pixval32f( dog_pyr[octv][intvl], r-1, c+1 ) +
			pixval32f( dog_pyr[octv][intvl], r-1, c-1 ) ) / 4.0;

	dxs = ( pixval32f( dog_pyr[octv][intvl+1], r, c+1 ) -
			pixval32f( dog_pyr[octv][intvl+1], r, c-1 ) -
			pixval32f( dog_pyr[octv][intvl-1], r, c+1 ) +
			pixval32f( dog_pyr[octv][intvl-1], r, c-1 ) ) / 4.0;

	dys = ( pixval32f( dog_pyr[octv][intvl+1], r+1, c ) -
			pixval32f( dog_pyr[octv][intvl+1], r-1, c ) -
			pixval32f( dog_pyr[octv][intvl-1], r+1, c ) +
			pixval32f( dog_pyr[octv][intvl-1], r-1, c ) ) / 4.0;


	H[0][0] = dxx;
	H[0][1] = dxy;
	H[0][2] = dxs;
	H[1][0] = dxy;
	H[1][1] = dyy;
	H[1][2] = dys;
	H[2][0] = dxs;
	H[2][1] = dys;
	H[2][2] = dss;
}



/*
Calculates interpolated pixel contrast.  Based on Eqn. (3) in Lowe's paper.

@param dog_pyr difference of Gaussians scale space pyramid
@param octv octave of scale space
@param intvl within-octave interval
@param r pixel row
@param c pixel column
@param xi interpolated subpixel increment to interval
@param xr interpolated subpixel increment to row
@param xc interpolated subpixel increment to col

@param Returns interpolated contrast.
*/
 float interp_contr( IplImage*** dog_pyr, int octv, int intvl, int r,
							int c, float xi, float xr, float xc )
{
	CvMat* dD, X, T;
	float t[1], x[3] = { xc, xr, xi };

	cvInitMatHeader( &X, 3, 1, CV_64FC1, x, CV_AUTOSTEP );
	cvInitMatHeader( &T, 1, 1, CV_64FC1, t, CV_AUTOSTEP );
	dD = deriv_3D( dog_pyr, octv, intvl, r, c );

	//cvGEMM( dD, &X, 1, NULL, 0, &T,  CV_GEMM_A_T );
	t[0] = cvGetReal2D(dD, 0, 0) * x[0] + cvGetReal2D(dD, 1, 0) * x[1] + cvGetReal2D(dD, 2, 0) * x[2];

	cvReleaseMat( &dD );

	return pixval32f( dog_pyr[octv][intvl], r, c ) + t[0] * 0.5;
}



/*
Allocates and initializes a new feature

@return Returns a pointer to the new feature
*/
 feature* new_feature( void )
{
	feature* feat;
	struct detection_data* ddata;

	feat = (feature*)malloc( sizeof( feature ) );
	memset( feat, 0, sizeof( feature ) );
	ddata = (detection_data*)malloc( sizeof( struct detection_data ) );
	memset( ddata, 0, sizeof( struct detection_data ) );
	feat->feature_data = ddata;
	feat->type = FEATURE_LOWE;

	return feat;
}



/*
Determines whether a feature is too edge like to be stable by computing the
ratio of principal curvatures at that feature.  Based on Section 4.1 of
Lowe's paper.

@param dog_img image from the DoG pyramid in which feature was detected
@param r feature row
@param c feature col
@param curv_thr high threshold on ratio of principal curvatures

@return Returns 0 if the feature at (r,c) in dog_img is sufficiently
	corner-like or 1 otherwise.
*/
 int is_too_edge_like( IplImage* dog_img, int r, int c, int curv_thr )
{
	float d, dxx, dyy, dxy, tr, det;

	/* principal curvatures are computed using the trace and det of Hessian */
	d = pixval32f(dog_img, r, c);
	dxx = pixval32f( dog_img, r, c+1 ) + pixval32f( dog_img, r, c-1 ) - 2 * d;
	dyy = pixval32f( dog_img, r+1, c ) + pixval32f( dog_img, r-1, c ) - 2 * d;
	dxy = ( pixval32f(dog_img, r+1, c+1) - pixval32f(dog_img, r+1, c-1) -
			pixval32f(dog_img, r-1, c+1) + pixval32f(dog_img, r-1, c-1) ) / 4.0;
	tr = dxx + dyy;
	det = dxx * dyy - dxy * dxy;

	/* negative determinant -> curvatures have different signs; reject feature */
	if( det <= 0 )
		return 1;

	if( tr * tr / det < ( curv_thr + 1.0 )*( curv_thr + 1.0 ) / curv_thr )
		return 0;
	return 1;
}



/*
Calculates characteristic scale for each feature in an array.

@param features array of features
@param sigma amount of Gaussian smoothing per octave of scale space
@param intvls intervals per octave of scale space
*/
 void SIFTGPU::calcFeatureScales( CvSeq* features, float sigma, int intvls )
{
	feature* feat;
	struct detection_data* ddata;
	float intvl;
	int i, n;

	n = features->total;
	for( i = 0; i < n; i++ )
	{
		//feature* a = &featureGPU[i];
		//struct detection_data* ddata2 = feat_detection_data( a );


		feat = CV_GET_SEQ_ELEM( feature, features, i );
		ddata = feat_detection_data( feat );
		intvl = ddata->intvl + ddata->subintvl;//
		feat->scl = sigma * pow( (float)2.0, ddata->octv + intvl / intvls );//
		ddata->scl_octv = sigma * pow((float) 2.0, intvl / intvls ); //
	}
}



/*
Halves feature coordinates and scale in case the input image was doubled
prior to scale space construction.

@param features array of features
*/
 void SIFTGPU::adjustForImgDbl( CvSeq* features )
{
	feature* feat;
	int i, n;

	n = features->total;
	for( i = 0; i < n; i++ )
	{
		feat = CV_GET_SEQ_ELEM( feature, features, i );
		feat->x /= 2.0;
		feat->y /= 2.0;
		feat->scl /= 2.0;
		feat->img_pt.x /= 2.0;
		feat->img_pt.y /= 2.0;
	}
}



/*
Computes a canonical orientation for each image feature in an array.  Based
on Section 5 of Lowe's paper.  This function adds features to the array when
there is more than one dominant orientation at a given feature location.

@param features an array of image features
@param gauss_pyr Gaussian scale space pyramid
*/
 void SIFTGPU::calc_feature_oris( CvSeq* features, IplImage*** gauss_pyr )
{
	feature* feat;
	struct detection_data* ddata;
	float* hist;
	float omax;
	int i, j, n = features->total;

	int tmp = iteratorFGPU;

	for( i = 0; i < n; i++ )
	{

		feat = (feature*)malloc( sizeof( feature ) );

		cvSeqPopFront( features, feat );

		ddata = feat_detection_data( feat );

		float hist[SIFT_ORI_HIST_BINS];
		for(int j = 0; j < SIFT_ORI_HIST_BINS; j++ )
							hist[j] = 0;

		ori_hist( gauss_pyr[ddata->octv][ddata->intvl],
						ddata->r, ddata->c, SIFT_ORI_HIST_BINS,
						ROUND( SIFT_ORI_RADIUS * ddata->scl_octv ),
						SIFT_ORI_SIG_FCTR * ddata->scl_octv, hist );

		
		for( j = 0; j < SIFT_ORI_SMOOTH_PASSES; j++ )
			smooth_ori_hist( hist, SIFT_ORI_HIST_BINS );

		int maxBin = 0;

		omax = dominant_ori( hist, SIFT_ORI_HIST_BINS, &maxBin );

		add_good_ori_features( features, hist, SIFT_ORI_HIST_BINS,
								omax * SIFT_ORI_PEAK_RATIO, feat, maxBin );

		free( ddata );
		free( feat );
		//free( hist );
	}
}



/*
Computes a gradient orientation histogram at a specified pixel.

@param img image
@param r pixel row
@param c pixel col
@param n number of histogram bins
@param rad radius of region over which histogram is computed
@param sigma std for Gaussian weighting of histogram entries

@return Returns an n-element array containing an orientation histogram
	representing orientations between 0 and 2 PI.
*/
void ori_hist( IplImage* img, int r, int c, int n, int rad, float sigma, float* hist)
{
	//float* hist;
	float mag, ori, w, exp_denom, PI2 = CV_PI * 2.0;
	int bin, i, j;

	//hist = (float*)calloc( n, sizeof( float ) );
	exp_denom = 2.0 * sigma * sigma;

	for( i = -rad; i <= rad; i++ )
		for( j = -rad; j <= rad; j++ )
			if( calc_grad_mag_ori( img, r + i, c + j, &mag, &ori ) )
			{	
				w = exp( -(float)( i*i + j*j ) / exp_denom );
				bin = ROUND( n * ( ori + CV_PI ) / PI2 );
				bin = ( bin < n )? bin : 0;
				hist[bin] += w * mag;
				
				//w = exp( -( i*i + j*j ) / exp_denom );
				//bin = cvRound( n * ( ori + CV_PI ) / PI2 );
				//bin = ( bin < n )? bin : 0;
				//hist[bin] += w * mag;
			}

}



/*
Calculates the gradient magnitude and orientation at a given pixel.

@param img image
@param r pixel row
@param c pixel col
@param mag output as gradient magnitude at pixel (r,c)
@param ori output as gradient orientation at pixel (r,c)

@return Returns 1 if the specified pixel is a valid one and sets mag and
	ori accordingly; otherwise returns 0
*/
 int calc_grad_mag_ori( IplImage* img, int r, int c, float* mag, float* ori )
{
	float dx, dy;

	if( r > 0  &&  r < img->height - 1  &&  c > 0  &&  c < img->width - 1 )
	{
		dx = pixval32f( img, r, c+1 ) - pixval32f( img, r, c-1 );
		dy = pixval32f( img, r-1, c ) - pixval32f( img, r+1, c );
		*mag = sqrt( dx*dx + dy*dy );
		*ori = atan2( dy, dx );
		return 1;
	}

	else
		return 0;
}



/*
Gaussian smooths an orientation histogram.

@param hist an orientation histogram
@param n number of bins
*/
 void smooth_ori_hist( float* hist, int n )
{
	float prev, tmp, h0 = hist[0];
	int i;

	prev = hist[n-1];
	for( i = 0; i < n; i++ )
	{
		tmp = hist[i];
		hist[i] = 0.25 * prev + 0.5 * hist[i] + 
			0.25 * ( ( i+1 == n )? h0 : hist[i+1] );
		prev = tmp;
	}
}



/*
Finds the magnitude of the dominant orientation in a histogram

@param hist an orientation histogram
@param n number of bins

@return Returns the value of the largest bin in hist
*/
 float dominant_ori( float* hist, int n, int* maxBin )
{
	float omax;
	int maxbin, i;

	omax = hist[0];
	maxbin = 0;
	for( i = 1; i < n; i++ )
		if( hist[i] > omax )
		{
			omax = hist[i];
			maxbin = i;
		}
	*maxBin = maxbin;
	return omax;
}



/*
Interpolates a histogram peak from left, center, and right values
*/
#define interp_hist_peak( l, c, r ) ( 0.5 * ((l)-(r)) / ((l) - 2.0*(c) + (r)) )



/*
Adds features to an array for every orientation in a histogram greater than
a specified threshold.

@param features new features are added to the end of this array
@param hist orientation histogram
@param n number of bins in hist
@param mag_thr new features are added for entries in hist greater than this
@param feat new features are clones of this with different orientations
*/
 void add_good_ori_features( CvSeq* features, float* hist, int n,
								   float mag_thr, feature* feat, int maxBin )
{
	feature* new_feat;
	float bin, PI2 = CV_PI * 2.0;
	int l, r, i;

	for( i = 0; i < n; i++ )
	{
		l = ( i == 0 )? n - 1 : i-1;
		r = ( i + 1 ) % n;

		if( hist[i] > hist[l]  &&  hist[i] > hist[r]  &&  hist[i] >= mag_thr )
		{
			bin = i + interp_hist_peak( hist[l], hist[i], hist[r] );
			bin = ( bin < 0 )? n + bin : ( bin >= n )? bin - n : bin;
			new_feat = clone_feature( feat );
			new_feat->ori = ( ( PI2 * bin ) / n ) - CV_PI;
			cvSeqPush( features, new_feat );
			free( new_feat );
		}
	}
}



/*
Makes a deep copy of a feature

@param feat feature to be cloned

@return Returns a deep copy of feat
*/
 feature* clone_feature( feature* feat )
{
	feature* new_feat;
	struct detection_data* ddata;

	new_feat = new_feature();
	ddata = feat_detection_data( new_feat );
	memcpy( new_feat, feat, sizeof( feature ) );
	memcpy( ddata, feat_detection_data(feat), sizeof( struct detection_data ) );
	new_feat->feature_data = ddata;

	return new_feat;
}



/*
Computes feature descriptors for features in an array.  Based on Section 6
of Lowe's paper.

@param features array of features
@param gauss_pyr Gaussian scale space pyramid
@param d width of 2D array of orientation histograms
@param n number of bins per orientation histogram
*/
 void compute_descriptors( CvSeq* features, IplImage*** gauss_pyr, int d, int n)
{
	feature* feat;
	struct detection_data* ddata;
	float*** hist;
	int i, k = features->total;

	for( i = 0; i < k; i++ )
	{
		feat = CV_GET_SEQ_ELEM(feature, features, i );
		ddata = feat_detection_data( feat );



		hist = descr_hist( gauss_pyr[ddata->octv][ddata->intvl], ddata->r,
			ddata->c, feat->ori, ddata->scl_octv, d, n );
		hist_to_descr( hist, d, n, feat );
		release_descr_hist( &hist, d );


	}
}



/*
Computes the 2D array of orientation histograms that form the feature
descriptor.  Based on Section 6.1 of Lowe's paper.

@param img image used in descriptor computation
@param r row coord of center of orientation histogram array
@param c column coord of center of orientation histogram array
@param ori canonical orientation of feature whose descr is being computed
@param scl scale relative to img of feature whose descr is being computed
@param d width of 2d array of orientation histograms
@param n bins per orientation histogram

@return Returns a d x d array of n-bin orientation histograms.
*/
 float*** descr_hist( IplImage* img, int r, int c, float ori,
							 float scl, int d, int n )
{
	float*** hist;
	float cos_t, sin_t, hist_width, exp_denom, r_rot, c_rot, grad_mag,
		grad_ori, w, rbin, cbin, obin, bins_per_rad, PI2 = 2.0 * CV_PI;
	int radius, i, j;

	hist = (float***)calloc( d, sizeof( float** ) );
	for( i = 0; i < d; i++ )
	{
		hist[i] = (float**)calloc( d, sizeof( float* ) );
		for( j = 0; j < d; j++ )
			hist[i][j] = (float*)calloc( n, sizeof( float ) );
	}

	cos_t = cos( ori );
	sin_t = sin( ori );
	bins_per_rad = n / PI2;
	exp_denom = d * d * 0.5;
	hist_width = SIFT_DESCR_SCL_FCTR * scl;
	radius = hist_width * sqrt(2.0) * ( d + 1.0 ) * 0.5 + 0.5;
	for( i = -radius; i <= radius; i++ )
		for( j = -radius; j <= radius; j++ )
		{
			/*
			Calculate sample's histogram array coords rotated relative to ori.
			Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
			r_rot = 1.5) have full weight placed in row 1 after interpolation.
			*/
			c_rot = ( j * cos_t - i * sin_t ) / hist_width;
			r_rot = ( j * sin_t + i * cos_t ) / hist_width;
			rbin = r_rot + d / 2 - 0.5;
			cbin = c_rot + d / 2 - 0.5;

			if( rbin > -1.0  &&  rbin < d  &&  cbin > -1.0  &&  cbin < d )
				if( calc_grad_mag_ori( img, r + i, c + j, &grad_mag, &grad_ori ))
				{
					grad_ori -= ori;
					while( grad_ori < 0.0 )
						grad_ori += PI2;
					while( grad_ori >= PI2 )
						grad_ori -= PI2;

					obin = grad_ori * bins_per_rad;
					w = exp( -(c_rot * c_rot + r_rot * r_rot) / exp_denom );
					interp_hist_entry( hist, rbin, cbin, obin, grad_mag * w, d, n );
				}
		}

	return hist;
}



/*
Interpolates an entry into the array of orientation histograms that form
the feature descriptor.

@param hist 2D array of orientation histograms
@param rbin sub-bin row coordinate of entry
@param cbin sub-bin column coordinate of entry
@param obin sub-bin orientation coordinate of entry
@param mag size of entry
@param d width of 2D array of orientation histograms
@param n number of bins per orientation histogram
*/
 void interp_hist_entry( float*** hist, float rbin, float cbin,
							   float obin, float mag, int d, int n )
{
	float d_r, d_c, d_o, v_r, v_c, v_o;
	float** row, * h;
	int r0, c0, o0, rb, cb, ob, r, c, o;

	r0 = cvFloor( rbin );
	c0 = cvFloor( cbin );
	o0 = cvFloor( obin );
	d_r = rbin - r0;
	d_c = cbin - c0;
	d_o = obin - o0;

	/*
	The entry is distributed into up to 8 bins.  Each entry into a bin
	is multiplied by a weight of 1 - d for each dimension, where d is the
	distance from the center value of the bin measured in bin units.
	*/
	for( r = 0; r <= 1; r++ )
	{
		rb = r0 + r;
		if( rb >= 0  &&  rb < d )
		{
			v_r = mag * ( ( r == 0 )? 1.0 - d_r : d_r );
			row = hist[rb];
			for( c = 0; c <= 1; c++ )
			{
				cb = c0 + c;
				if( cb >= 0  &&  cb < d )
				{
					v_c = v_r * ( ( c == 0 )? 1.0 - d_c : d_c );
					h = row[cb];
					for( o = 0; o <= 1; o++ )
					{
						ob = ( o0 + o ) % n;
						v_o = v_c * ( ( o == 0 )? 1.0 - d_o : d_o );
						h[ob] += v_o;
					}
				}
			}
		}
	}
}



/*
Converts the 2D array of orientation histograms into a feature's descriptor
vector.

@param hist 2D array of orientation histograms
@param d width of hist
@param n bins per histogram
@param feat feature into which to store descriptor
*/
 void hist_to_descr( float*** hist, int d, int n, feature* feat )
{
	int int_val, i, r, c, o, k = 0;

	for( r = 0; r < d; r++ )
		for( c = 0; c < d; c++ )
			for( o = 0; o < n; o++ )
				feat->descr[k++] = hist[r][c][o];

	feat->d = k;
	normalize_descr( feat );
	for( i = 0; i < k; i++ )
		if( feat->descr[i] > SIFT_DESCR_MAG_THR )
			feat->descr[i] = SIFT_DESCR_MAG_THR;
	normalize_descr( feat );

	/* convert floating-point descriptor to integer valued descriptor */
	for( i = 0; i < k; i++ )
	{
		int_val = SIFT_INT_DESCR_FCTR * feat->descr[i];
		feat->descr[i] = MIN( 255, int_val );
	}
}



/*
Normalizes a feature's descriptor vector to unitl length

@param feat feature
*/
 void normalize_descr( feature* feat )
{
	float cur, len_inv, len_sq = 0.0;
	int i, d = feat->d;

	for( i = 0; i < d; i++ )
	{
		cur = feat->descr[i];
		len_sq += cur*cur;
	}
	len_inv = 1.0 / sqrt( len_sq );
	for( i = 0; i < d; i++ )
		feat->descr[i] *= len_inv;
}



/*
Compares features for a decreasing-scale ordering.  Intended for use with
CvSeqSort

@param feat1 first feature
@param feat2 second feature
@param param unused

@return Returns 1 if feat1's scale is greater than feat2's, -1 if vice versa,
and 0 if their scales are equal
*/
 int feature_cmp( void* feat1, void* feat2, void* param )
{
	feature* f1 = (feature*) feat1;
	feature* f2 = (feature*) feat2;

	if( f1->scl < f2->scl )
		return 1;
	if( f1->scl > f2->scl )
		return -1;
	return 0;
}



/*
De-allocates memory held by a descriptor histogram

@param hist pointer to a 2D array of orientation histograms
@param d width of hist
*/
 void release_descr_hist( float**** hist, int d )
{
	int i, j;

	for( i = 0; i < d; i++)
	{
		for( j = 0; j < d; j++ )
			free( (*hist)[i][j] );
		free( (*hist)[i] );
	}
	free( *hist );
	*hist = NULL;
}


/*
De-allocates memory held by a scale space pyramid

@param pyr scale space pyramid
@param octvs number of octaves of scale space
@param n number of images per octave
*/
 void release_pyr( IplImage**** pyr, int octvs, int n )
{
	int i, j;
	for( i = 0; i < octvs; i++ )
	{
		for( j = 0; j < n; j++ )
			cvReleaseImage( &(*pyr)[i][j] );
		free( (*pyr)[i] );
	}
	free( *pyr );
	*pyr = NULL;
}






 SIFTGPU::SIFTGPU()
 {
	img_file_name = "C:\\scene2.jpg";
	out_file_name  = "C:\\Users\\Mati\\Pictures\\scene2.sift";;
	out_img_name = "C:\\Users\\Mati\\Pictures\\sceneOut2.jpg";
	display = 1;
	intvls = SIFT_INTVLS;
	sigma = SIFT_SIGMA;
	contr_thr = SIFT_CONTR_THR;
	curv_thr = SIFT_CURV_THR;
	img_dbl = SIFT_IMG_DBL;
	descr_width = SIFT_DESCR_WIDTH;
	descr_hist_bins = SIFT_DESCR_HIST_BINS;

	iteratorFGPU = 0;
	
	featureGPU = (feature*)malloc( 2000 * sizeof( feature ) );

	meanFilter = new MeanFilter();
	subtract = new Subtract();
	detectExt = new DetectExtrema();
	/*magOrient = new MagnitudeOrientation();
	assignOrient = new AssignOrientations();
	extractKeys = new ExtractKeypointDescriptors();*/


 }


void SIFTGPU::DoSift()
{
	

}



















/* width of border in which to ignore keypoints */
#define SIFT_IMG_BORDER 5



/** default number of sampled intervals per octave */
#define SIFT_INTVLS 3

/** default threshold on keypoint contrast |D(x)| */
#define SIFT_CONTR_THR 0.04

/** default threshold on keypoint ratio of principle curvatures */
#define SIFT_CURV_THR 10

/** default sigma for initial gaussian smoothing */
#define SIFT_SIGMA		1.6

/** default number of sampled intervals per octave */
#define SIFT_INTVLS		3

/* determines gaussian sigma for orientation assignment */
#define SIFT_ORI_SIG_FCTR 1.5

/* determines the radius of the region used in orientation assignment */
#define SIFT_ORI_RADIUS 3.0 * SIFT_ORI_SIG_FCTR

/* default number of bins in histogram for orientation assignment */
#define SIFT_ORI_HIST_BINS 36

/* number of passes of orientation histogram smoothing */
#define SIFT_ORI_SMOOTH_PASSES 2

/* orientation magnitude relative to max that results in new feature */
#define SIFT_ORI_PEAK_RATIO 1.0

#define CV_PI   3.1415926535897932384626433832795

/* absolute value */
#define ABS(x) ( ( (x) < 0 )? -(x) : (x) )

#define ROUND(x) ( ( x - (int)x ) <= 0.5 ? (int)x :  (int)x + 1 )

/*
Interpolates a histogram peak from left, center, and right values
*/
#define interp_hist_peak( l, c, r ) ( 0.5 * ((l)-(r)) / ((l) - 2.0*(c) + (r)) )




float GetPixel( float* dataIn, int x, int y, int ImageWidth, int ImageHeight )
{
	int X = x > ImageWidth  ? ImageWidth  : x;
	int Y = y > ImageHeight ? ImageHeight : y;
	int GMEMOffset = Y * ImageWidth + X;

	return dataIn[GMEMOffset];
}


/*
Determines whether a pixel is a scale-space extremum by comparing it to it's
3x3x3 pixel neighborhood.

@return Returns 1 if the specified pixel is an extremum (max or min) among
	it's 3x3x3 pixel neighborhood.
*/
int is_extremum( float* dataIn1,  float* dataIn2,  float* dataIn3, int pozX, int pozY, int ImageWidth, int ImageHeight )
{
	float val = GetPixel(dataIn2, pozX, pozY, ImageWidth, ImageHeight);

	int i, j, k;

	/* check for maximum */
	if( val > 0 )
	{
			for( j = -1; j <= 1; j++ )
				for( k = -1; k <= 1; k++ )
				{
					if( val < GetPixel(dataIn1, pozX+j, pozY+k, ImageWidth, ImageHeight) )
						return 0;
					if( val < GetPixel(dataIn2, pozX+j, pozY+k, ImageWidth, ImageHeight) )
						return 0;
					if( val < GetPixel(dataIn3, pozX+j, pozY+k, ImageWidth, ImageHeight) )
						return 0;
				}
	}

	/* check for minimum */
	else
	{
			for( j = -1; j <= 1; j++ )
				for( k = -1; k <= 1; k++ )
				{
					if( val > GetPixel(dataIn1, pozX+j, pozY+k, ImageWidth, ImageHeight) )
						return 0;
					if( val > GetPixel(dataIn2, pozX+j, pozY+k, ImageWidth, ImageHeight) )
						return 0;
					if( val > GetPixel(dataIn3, pozX+j, pozY+k, ImageWidth, ImageHeight) )
						return 0;
				}
	}



	return 1;
	
	/*if( val > 0.0 )
	{
		
				if( val < GetPixel(dataIn1, pozX-1, pozY-1, ImageWidth, ImageHeight) )
					return 0;
				if( val < GetPixel(dataIn2, pozX-1, pozY-1, ImageWidth, ImageHeight) )
					return 0;
				if( val < GetPixel(dataIn3, pozX-1, pozY-1, ImageWidth, ImageHeight) )
					return 0;
				if( val < GetPixel(dataIn1, pozX-1, pozY, ImageWidth, ImageHeight) )
					return 0;
				if( val < GetPixel(dataIn2, pozX-1, pozY, ImageWidth, ImageHeight) )
					return 0;
				if( val < GetPixel(dataIn3, pozX-1, pozY, ImageWidth, ImageHeight) )
					return 0;
				if( val < GetPixel(dataIn1, pozX-1, pozY+1, ImageWidth, ImageHeight) )
					return 0;
				if( val < GetPixel(dataIn2, pozX-1, pozY+1, ImageWidth, ImageHeight) )
					return 0;
				if( val < GetPixel(dataIn3, pozX-1, pozY+1, ImageWidth, ImageHeight) )
					return 0;

				if( val < GetPixel(dataIn1, pozX, pozY-1, ImageWidth, ImageHeight) )
					return 0;
				if( val < GetPixel(dataIn2, pozX, pozY-1, ImageWidth, ImageHeight) )
					return 0;
				if( val < GetPixel(dataIn3, pozX, pozY-1, ImageWidth, ImageHeight) )
					return 0;
				if( val < GetPixel(dataIn1, pozX, pozY, ImageWidth, ImageHeight) )
					return 0;
				if( val < GetPixel(dataIn2, pozX, pozY, ImageWidth, ImageHeight) )
					return 0;
				if( val < GetPixel(dataIn3, pozX, pozY, ImageWidth, ImageHeight) )
					return 0;
				if( val < GetPixel(dataIn1, pozX, pozY+1, ImageWidth, ImageHeight) )
					return 0;
				if( val < GetPixel(dataIn2, pozX, pozY+1, ImageWidth, ImageHeight) )
					return 0;
				if( val < GetPixel(dataIn3, pozX, pozY+1, ImageWidth, ImageHeight) )
					return 0;

				if( val < GetPixel(dataIn1, pozX+1, pozY-1, ImageWidth, ImageHeight) )
					return 0;
				if( val < GetPixel(dataIn2, pozX+1, pozY-1, ImageWidth, ImageHeight) )
					return 0;
				if( val < GetPixel(dataIn3, pozX+1, pozY-1, ImageWidth, ImageHeight) )
					return 0;
				if( val < GetPixel(dataIn1, pozX+1, pozY, ImageWidth, ImageHeight) )
					return 0;
				if( val < GetPixel(dataIn2, pozX+1, pozY, ImageWidth, ImageHeight) )
					return 0;
				if( val < GetPixel(dataIn3, pozX+1, pozY, ImageWidth, ImageHeight) )
					return 0;
				if( val < GetPixel(dataIn1, pozX+1, pozY+1, ImageWidth, ImageHeight) )
					return 0;
				if( val < GetPixel(dataIn2, pozX+1, pozY+1, ImageWidth, ImageHeight) )
					return 0;
				if( val < GetPixel(dataIn3, pozX+1, pozY+1, ImageWidth, ImageHeight) )
					return 0;
	}
	else 
	{
				if( val > GetPixel(dataIn1, pozX-1, pozY-1, ImageWidth, ImageHeight) )
					return 0;
				if( val > GetPixel(dataIn2, pozX-1, pozY-1, ImageWidth, ImageHeight) )
					return 0;
				if( val > GetPixel(dataIn3, pozX-1, pozY-1, ImageWidth, ImageHeight) )
					return 0;
				if( val > GetPixel(dataIn1, pozX-1, pozY, ImageWidth, ImageHeight) )
					return 0;
				if( val > GetPixel(dataIn2, pozX-1, pozY, ImageWidth, ImageHeight) )
					return 0;
				if( val > GetPixel(dataIn3, pozX-1, pozY, ImageWidth, ImageHeight) )
					return 0;
				if( val > GetPixel(dataIn1, pozX-1, pozY+1, ImageWidth, ImageHeight) )
					return 0;
				if( val > GetPixel(dataIn2, pozX-1, pozY+1, ImageWidth, ImageHeight) )
					return 0;
				if( val > GetPixel(dataIn3, pozX-1, pozY+1, ImageWidth, ImageHeight) )
					return 0;

				if( val > GetPixel(dataIn1, pozX, pozY-1, ImageWidth, ImageHeight) )
					return 0;
				if( val > GetPixel(dataIn2, pozX, pozY-1, ImageWidth, ImageHeight) )
					return 0;
				if( val > GetPixel(dataIn3, pozX, pozY-1, ImageWidth, ImageHeight) )
					return 0;
				if( val > GetPixel(dataIn1, pozX, pozY, ImageWidth, ImageHeight) )
					return 0;
				if( val > GetPixel(dataIn2, pozX, pozY, ImageWidth, ImageHeight) )
					return 0;
				if( val > GetPixel(dataIn3, pozX, pozY, ImageWidth, ImageHeight) )
					return 0;
				if( val > GetPixel(dataIn1, pozX, pozY+1, ImageWidth, ImageHeight) )
					return 0;
				if( val > GetPixel(dataIn2, pozX, pozY+1, ImageWidth, ImageHeight) )
					return 0;
				if( val > GetPixel(dataIn3, pozX, pozY+1, ImageWidth, ImageHeight) )
					return 0;

				if( val > GetPixel(dataIn1, pozX+1, pozY-1, ImageWidth, ImageHeight) )
					return 0;
				if( val > GetPixel(dataIn2, pozX+1, pozY-1, ImageWidth, ImageHeight) )
					return 0;
				if( val > GetPixel(dataIn3, pozX+1, pozY-1, ImageWidth, ImageHeight) )
					return 0;
				if( val > GetPixel(dataIn1, pozX+1, pozY, ImageWidth, ImageHeight) )
					return 0;
				if( val > GetPixel(dataIn2, pozX+1, pozY, ImageWidth, ImageHeight) )
					return 0;
				if( val > GetPixel(dataIn3, pozX+1, pozY, ImageWidth, ImageHeight) )
					return 0;
				if( val > GetPixel(dataIn1, pozX+1, pozY+1, ImageWidth, ImageHeight) )
					return 0;
				if( val > GetPixel(dataIn2, pozX+1, pozY+1, ImageWidth, ImageHeight) )
					return 0;
				if( val > GetPixel(dataIn3, pozX+1, pozY+1, ImageWidth, ImageHeight) )
					return 0;
	}

	
	
	return 1;*/
}

/*
Computes the partial derivatives in x, y, and scale of a pixel in the DoG
scale space pyramid
*/
void deriv_3D(  float* dataIn1,  float* dataIn2,  float* dataIn3, int pozX, int pozY, int ImageWidth, int ImageHeight, float* dI )
{
	float dx, dy, ds;
	dx = ( GetPixel(dataIn2, pozX+1, pozY, ImageWidth, ImageHeight) - GetPixel(dataIn2, pozX-1, pozY, ImageWidth, ImageHeight) ) / 2.0;
	dy = ( GetPixel(dataIn2, pozX, pozY+1, ImageWidth, ImageHeight) - GetPixel(dataIn2, pozX, pozY-1, ImageWidth, ImageHeight) ) / 2.0;
	ds = ( GetPixel(dataIn3, pozX, pozY, ImageWidth, ImageHeight) - GetPixel(dataIn1, pozX, pozY, ImageWidth, ImageHeight) ) / 2.0;
	dI[0] = dx;
	dI[1] = dy;
	dI[2] = ds;
}

/*
Computes the 3D Hessian matrix for a pixel in the DoG scale space pyramid.
	/ Ixx  Ixy  Ixs \ <BR>
	| Ixy  Iyy  Iys | <BR>
	\ Ixs  Iys  Iss /
*/
void hessian_3D(  float* dataIn1,  float* dataIn2,  float* dataIn3, int pozX, int pozY, int ImageWidth, int ImageHeight, float H[][3] )
{
	float v, dxx, dyy, dss, dxy, dxs, dys;

	v = GetPixel(dataIn2, pozX, pozY, ImageWidth, ImageHeight);

	dxx = ( GetPixel(dataIn2, pozX+1, pozY, ImageWidth, ImageHeight) + 
			GetPixel(dataIn2, pozX-1, pozY, ImageWidth, ImageHeight) - 2 * v );

	dyy = ( GetPixel(dataIn2, pozX, pozY+1, ImageWidth, ImageHeight) +
			GetPixel(dataIn2, pozX, pozY-1, ImageWidth, ImageHeight) - 2 * v );

	dss = ( GetPixel(dataIn3, pozX, pozY, ImageWidth, ImageHeight) +
			GetPixel(dataIn1, pozX, pozY, ImageWidth, ImageHeight) - 2 * v );

	dxy = ( GetPixel(dataIn2, pozX+1, pozY+1, ImageWidth, ImageHeight) -
			GetPixel(dataIn2, pozX-1, pozY+1, ImageWidth, ImageHeight) -
			GetPixel(dataIn2, pozX+1, pozY-1, ImageWidth, ImageHeight) +
			GetPixel(dataIn2, pozX-1, pozY-1, ImageWidth, ImageHeight) ) / 4.0;

	dxs = ( GetPixel(dataIn3, pozX+1, pozY, ImageWidth, ImageHeight) -
			GetPixel(dataIn3, pozX-1, pozY, ImageWidth, ImageHeight) -
			GetPixel(dataIn1, pozX+1, pozY, ImageWidth, ImageHeight) +
			GetPixel(dataIn1, pozX-1, pozY, ImageWidth, ImageHeight) ) / 4.0;

	dys = ( GetPixel(dataIn3, pozX, pozY+1, ImageWidth, ImageHeight) -
			GetPixel(dataIn3, pozX, pozY-1, ImageWidth, ImageHeight) -
			GetPixel(dataIn1, pozX, pozY+1, ImageWidth, ImageHeight) +
			GetPixel(dataIn1, pozX, pozY-1, ImageWidth, ImageHeight) ) / 4.0;



	H[0][0] = dxx;
	H[0][1] = dxy;
	H[0][2] = dxs;
	H[1][0] = dxy;
	H[1][1] = dyy;
	H[1][2] = dys;
	H[2][0] = dxs;
	H[2][1] = dys;
	H[2][2] = dss;
}





/*
Performs one step of extremum interpolation.  Based on Eqn. (3) in Lowe's
paper.
*/
void interp_step( float* dataIn1,  float* dataIn2,  float* dataIn3, int pozX, int pozY, int ImageWidth, int ImageHeight,
						 float* xi, float* xr, float* xc )
{
	
	float dD[3] = { 0, 0 , 0 };
	float H[3][3];
	float H_inv[3][3];

	deriv_3D(dataIn1, dataIn2, dataIn3, pozX, pozY, ImageWidth, ImageHeight, dD);
	hessian_3D(dataIn1, dataIn2, dataIn3, pozX, pozY, ImageWidth, ImageHeight, H);

	float a = H[0][0];
	float b = H[0][1];
	float c = H[0][2];
	float d = H[1][0];
	float e = H[1][1];
	float f = H[1][2];
	float g = H[2][0];
	float h = H[2][1];
	float k = H[2][2];

	float det = a*(e*k - f*h) + b*(f*g - k*d) + c*(d*h - e*g);
	float det_inv = 1.0 / det;

	H_inv[0][0] = (e*k - f*h)*det_inv;
	H_inv[0][1] = (c*h - b*k)*det_inv;
	H_inv[0][2] = (b*f - c*e)*det_inv;

	H_inv[1][0] = (f*g - d*k)*det_inv;
	H_inv[1][1] = (a*k - c*g)*det_inv;
	H_inv[1][2] = (c*d - a*f)*det_inv;

	H_inv[2][0] = (d*h - e*g)*det_inv;
	H_inv[2][1] = (g*b - a*h)*det_inv;
	H_inv[2][2] = (a*e - b*d)*det_inv;

	*xc = (-1)*( H_inv[0][0]*dD[0] + H_inv[1][0]*dD[1] + H_inv[2][0]*dD[2]);
	*xr = (-1)*( H_inv[0][1]*dD[0] + H_inv[1][1]*dD[1] + H_inv[2][1]*dD[2]);
	*xi = (-1)*( H_inv[0][2]*dD[0] + H_inv[1][2]*dD[1] + H_inv[2][2]*dD[2]);
}



/*
Calculates interpolated pixel contrast.  Based on Eqn. (3) in Lowe's paper.

@param Returns interpolated contrast.
*/
float interp_contr( float* dataIn1,  float* dataIn2,  float* dataIn3, int pozX, int pozY, int ImageWidth, int ImageHeight, float xi, float xr, float xc )
{
	float dD[3] = { 0, 0, 0 };
	deriv_3D(dataIn1, dataIn2, dataIn3, pozX, pozY, ImageWidth, ImageHeight, dD);
	float res = xc*dD[0] + xr*dD[1] + xi*dD[2];

	return GetPixel(dataIn2, pozX, pozY, ImageWidth, ImageHeight) + res * 0.5;
}


float interp_extremum( float* dataIn1,  float* dataIn2,  float* dataIn3, int pozX, int pozY, int ImageWidth, int ImageHeight, 
	int intvls, float contr_thr, int intvl, float* xi, float* xr, float* xc )
{
	
	float contr;

	int i = 0;
	int siftMaxInterpSteps = 5;

	if( pozX == 668 )
		i = 0;

	while( i < siftMaxInterpSteps )
	{
		interp_step(dataIn1, dataIn2, dataIn3, pozX, pozY, ImageWidth, ImageHeight, xi, xr, xc );
		
		if( ABS(*xi) <= 0.5 && ABS(*xr) <= 0.5 && ABS(*xc) <= 0.5 )
			break;
		
		pozX += ROUND( *xc);
		pozY += ROUND( *xr );
		intvl += ROUND( *xc );

		if( intvl < 1  ||
			intvl > intvls  ||
			pozX < SIFT_IMG_BORDER  ||
			pozY < SIFT_IMG_BORDER  ||
			pozX >= ImageWidth - SIFT_IMG_BORDER  ||
			pozY >= ImageHeight - SIFT_IMG_BORDER )
		{
			return 0;
		}
		i++;
	}

	/* ensure convergence of interpolation */
	if( i >= siftMaxInterpSteps )
		return 0;

	contr = interp_contr(dataIn1, dataIn2, dataIn3, pozX, pozY, ImageWidth, ImageHeight, *xi, *xr, *xc );
	if( (float)ABS( contr ) < (float)contr_thr / (float)intvls )
		return 0;

	return 1;
}

/*
Determines whether a feature is too edge like to be stable by computing the
ratio of principal curvatures at that feature.  Based on Section 4.1 of
Lowe's paper.

@return Returns 0 if the feature at (r,H[0][2]) in dog_img is sufficiently
	corner-like or 1 otherwise.
*/
 int is_too_edge_like( float* dataIn2, int pozX, int pozY, int ImageWidth, int ImageHeight, int curv_thr )
{
	float d, dxx, dyy, dxy, tr, det;

	/* principal curvatures are computed using the trace and det of Hessian */
	d = GetPixel(dataIn2, pozX, pozY, ImageWidth, ImageHeight);
	dxx = GetPixel(dataIn2, pozX+1, pozY, ImageWidth, ImageHeight)  + GetPixel(dataIn2, pozX-1, pozY, ImageWidth, ImageHeight) - 2 * d;
	dyy = GetPixel(dataIn2, pozX, pozY+1, ImageWidth, ImageHeight) + GetPixel(dataIn2, pozX, pozY-1, ImageWidth, ImageHeight) - 2 * d;
	dxy = ( GetPixel(dataIn2, pozX+1, pozY+1, ImageWidth, ImageHeight) - GetPixel(dataIn2, pozX-1, pozY+1, ImageWidth, ImageHeight) -
			GetPixel(dataIn2, pozX+1, pozY-1, ImageWidth, ImageHeight) + GetPixel(dataIn2, pozX-1, pozY-1, ImageWidth, ImageHeight) ) / 4.0;
	tr = dxx + dyy;
	det = dxx * dyy - dxy * dxy;

	/* negative determinant -> curvatures have different signs; reject feature */
	if( det <= 0 )
		return 1;

	if( tr * tr / det < ( curv_thr + 1.0 )*( curv_thr + 1.0 ) / curv_thr )
		return 0;
	return 1;
}


 /*
Calculates the gradient magnitude and orientation at a given pixel.


@return Returns 1 if the specified pixel is a valid one and sets mag and
	ori accordingly; otherwise returns 0
*/
 int calc_grad_mag_ori( float* gauss_pyr, int pozX, int pozY, int ImageWidth, int ImageHeight, float* mag, float* ori )
{
	float dx, dy;

	if( pozX > 0  &&  pozX < ImageWidth - 1  &&  pozY > 0  &&  pozY < ImageHeight - 1 )
	{
		dx = GetPixel(gauss_pyr, pozX+1, pozY, ImageWidth, ImageHeight) - GetPixel(gauss_pyr, pozX-1, pozY, ImageWidth, ImageHeight);
		dy =  GetPixel(gauss_pyr, pozX, pozY-1, ImageWidth, ImageHeight) - GetPixel(gauss_pyr, pozX, pozY+1, ImageWidth, ImageHeight);
		*mag = sqrt( dx*dx + dy*dy );
		*ori = atan2( dy, dx );
		return 1;
	}

	else
		return 0;
}


 /*
Computes a gradient orientation histogram at a specified pixel.


@return Returns an n-element array containing an orientation histogram
	representing orientations between 0 and 2 PI.
*/
void ori_hist( float* gauss_pyr, int pozX, int pozY, int ImageWidth, int ImageHeight, float* hist, int n, int rad, float sigma)
{
	float mag, ori, w, exp_denom, PI2 = CV_PI * 2.0;
	int bin, i, j;

	exp_denom = 2.0 * sigma * sigma;

	for( i = -rad; i <= rad; i++ )
		for( j = -rad; j <= rad; j++ )
			if( calc_grad_mag_ori( gauss_pyr, pozX + i, pozY + j, ImageWidth, ImageHeight, &mag, &ori ) )
			{	
				w = exp( -(float)( i*i + j*j ) / exp_denom );
				bin = ROUND( n * ( ori + CV_PI ) / PI2 );
				bin = ( bin < n )? bin : 0;
				hist[bin] += w * mag;
			}


}



/*
Adds features to an array for every orientation in a histogram greater than
a specified threshold.

*/
void add_good_ori_features(float* hist, int n, float mag_thr, float* orients, int* numberOrient )
{

	float bin, PI2 = CV_PI * 2.0;
	int l, r, i;

	for( i = 0; i < n; i++ )
	{
		l = ( i == 0 )? n - 1 : i-1;
		r = ( i + 1 ) % n;

		if( hist[i] > hist[l]  &&  hist[i] > hist[r]  &&  hist[i] >= mag_thr )
		{

			bin = i + interp_hist_peak( hist[l], hist[i], hist[r] );
			bin = ( bin < 0 )? n + bin : ( bin >= n )? bin - n : bin;

			orients[*numberOrient] = ( ( PI2 * bin ) / n ) - CV_PI;

			++(*numberOrient);
		}
	}
}


/*
Interpolates an entry into the array of orientation histograms that form
the feature descriptor.

*/
 void interp_hist_entryGPU( float hist[SIFT_DESCR_WIDTH][SIFT_DESCR_WIDTH][SIFT_DESCR_HIST_BINS] , float rbin, float cbin,
							   float obin, float mag, int d, int n )
{
	float d_r, d_c, d_o, v_r, v_c, v_o;
	float** row, * h;
	int r0, c0, o0, rb, cb, ob, r, c, o;

	r0 = cvFloor( rbin );  // floor()
	c0 = cvFloor( cbin );
	o0 = cvFloor( obin );
	d_r = rbin - r0;
	d_c = cbin - c0;
	d_o = obin - o0;

	/*
	The entry is distributed into up to 8 bins.  Each entry into a bin
	is multiplied by a weight of 1 - d for each dimension, where d is the
	distance from the center value of the bin measured in bin units.
	*/
	for( r = 0; r <= 1; r++ )
	{
		rb = r0 + r;

		if( rb >= 0  &&  rb < d )
		{
			v_r = mag * ( ( r == 0 )? 1.0 - d_r : d_r );
			
			for( c = 0; c <= 1; c++ )
			{

				cb = c0 + c;
				if( cb >= 0  &&  cb < d )
				{

					v_c = v_r * ( ( c == 0 )? 1.0 - d_c : d_c );
					
					for( o = 0; o <= 1; o++ )
					{
						ob = ( o0 + o ) % n;
						v_o = v_c * ( ( o == 0 )? 1.0 - d_o : d_o );
						hist[rb][cb][ob] += v_o;
					}
				}
			}
		}
	}
}


/*
Computes the 2D array of orientation histograms that form the feature
descriptor.  Based on Section 6.1 of Lowe's paper.

*/
void descr_hist(float* gauss_pyr,  int pozX, int pozY, int ImageWidth, int ImageHeight, float ori, float scl, float hist[SIFT_DESCR_WIDTH][SIFT_DESCR_WIDTH][SIFT_DESCR_HIST_BINS], int d, int n )
{
	
	float cos_t, sin_t, hist_width, exp_denom, r_rot, c_rot, grad_mag,
		grad_ori, w, rbin, cbin, obin, bins_per_rad, PI2 = 2.0 * CV_PI;
	int radius, i, j;

	
	cos_t = cos( ori );
	sin_t = sin( ori );
	bins_per_rad = n / PI2;
	exp_denom = d * d * 0.5;
	hist_width = SIFT_DESCR_SCL_FCTR * scl;
	radius = hist_width * sqrt(2.0) * ( d + 1.0 ) * 0.5 + 0.5;



	for( i = -radius; i <= radius; i++ )
		for( j = -radius; j <= radius; j++ )
		{
			/*
			Calculate sample's histogram array coords rotated relative to ori.
			Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
			r_rot = 1.5) have full weight placed in row 1 after interpolation.
			*/
			c_rot = ( j * cos_t - i * sin_t ) / hist_width;
			r_rot = ( j * sin_t + i * cos_t ) / hist_width;
			rbin = r_rot + d / 2 - 0.5;
			cbin = c_rot + d / 2 - 0.5;

			if( rbin > -1.0  &&  rbin < d  &&  cbin > -1.0  &&  cbin < d )
				if( calc_grad_mag_ori( gauss_pyr, pozX + i, pozY + j, ImageWidth, ImageHeight, &grad_mag, &grad_ori ) )
				{
					grad_ori -= ori;
					while( grad_ori < 0.0 )
						grad_ori += PI2;
					while( grad_ori >= PI2 )
						grad_ori -= PI2;

					obin = grad_ori * bins_per_rad;
					w = exp( -(c_rot * c_rot + r_rot * r_rot) / exp_denom );

					interp_hist_entryGPU( hist, rbin, cbin, obin, grad_mag * w, d, n );
				}
		}


}

/*
Normalizes a feature's descriptor vector to unitl length

@param feat feature
*/
 void normalize_descr( float* desc )
{
	float cur, len_inv, len_sq = 0.0;
	int i;

	for( i = 0; i < 128; i++ )
	{
		cur = desc[i];
		len_sq += cur*cur;
	}
	len_inv = 1.0 / sqrt( len_sq );
	for( i = 0; i < 128; i++ )
		desc[i] *= len_inv;
}


void ckDetect( float* dataIn1,  float* dataIn2,  float* dataIn3,   float* gauss_pyr,  float* ucDest,
						 int* numberExtrema,  float* keys,
						int ImageWidth, int ImageHeight, float prelim_contr_thr, int intvl, int octv,  int* number,  int* numberRej, int pozX, int pozY)
{
	//int pozX = get_global_id(0);
	//int pozY = get_global_id(1);
	int GMEMOffset = pozY *ImageWidth + pozX;
	
	float xc;
	float xr;
	float xi;

	int numberExt = 0;

	if( pozX < ImageWidth-SIFT_IMG_BORDER && pozY < ImageHeight-SIFT_IMG_BORDER && pozX > SIFT_IMG_BORDER && pozY > SIFT_IMG_BORDER )
	{
		
		float pixel = GetPixel(dataIn2, pozX, pozY, ImageWidth, ImageHeight);
		
		if( ABS(pixel) > prelim_contr_thr )
		{
			

			if( is_extremum( dataIn1, dataIn2, dataIn2, pozX, pozY, ImageWidth, ImageHeight) == 1 )
			{

				float feat = interp_extremum( dataIn1, dataIn2, dataIn3, pozX, pozY, ImageWidth, ImageHeight, SIFT_INTVLS, SIFT_CONTR_THR, intvl, &xi, &xr, &xc);
				if( feat )
				{
					if( is_too_edge_like( dataIn2, pozX, pozY, ImageWidth, ImageHeight, SIFT_CURV_THR ) != 1 )
					{

						float intvl2 = intvl + xi;  //intvl = ddata->intvl + ddata->subintvl;//

						float	scx = (float)(( pozX + xc ) * pow( (float)2.0, (float)octv ) / 2.0);
						float	scy = (float)(( pozY + xr ) * pow( (float)2.0, (float)octv ) / 2.0);
						float	x = pozX;
						float	y = pozY;
						float	subintvl = xi;
						float	intvlRes = intvl;
						float	octvRes = octv;
						float	scl = (SIFT_SIGMA * pow( (float)2.0, (octv + intvl2 / (float)SIFT_INTVLS) )) / 2.0;  //sigma * pow( (float)2.0, ddata->octv + intvl / intvls );//
						float	scl_octv = SIFT_SIGMA * pow( (float)2.0, (float)(intvl2 / SIFT_INTVLS) );
						float	ori = 0;
						float	mag = 0;

						float hist[SIFT_ORI_HIST_BINS];
						for(int j = 0; j < SIFT_ORI_HIST_BINS; j++ )
							hist[j] = 0;

						ori_hist( gauss_pyr, pozX, pozY, ImageWidth, ImageHeight, hist, SIFT_ORI_HIST_BINS,
										ROUND( SIFT_ORI_RADIUS * scl_octv ),	SIFT_ORI_SIG_FCTR * scl_octv );



						for(int j = 0; j < SIFT_ORI_SMOOTH_PASSES; j++ )
							smooth_ori_hist( hist, SIFT_ORI_HIST_BINS );

						int maxBin = 0;

						float omax = dominant_ori( hist, SIFT_ORI_HIST_BINS, &maxBin );

						float orients[SIFT_ORI_HIST_BINS];
						for(int j = 0; j < SIFT_ORI_HIST_BINS; j++ )
							orients[j] = 0;

						int numberOrient = 0;

						add_good_ori_features(hist, SIFT_ORI_HIST_BINS,	omax * SIFT_ORI_PEAK_RATIO, orients, &numberOrient);

						int iteratorOrient = 0;
						for(iteratorOrient = 0; iteratorOrient < numberOrient; iteratorOrient++ )
						{

							float hist2[SIFT_DESCR_WIDTH][SIFT_DESCR_WIDTH][SIFT_DESCR_HIST_BINS];

							for(int ii = 0; ii < SIFT_DESCR_WIDTH; ii++)
								for(int iii = 0; iii < SIFT_DESCR_WIDTH; iii++)
									for(int iiii = 0; iiii < SIFT_DESCR_HIST_BINS; iiii++)
										hist2[ii][iii][iiii] = 0.0;

							descr_hist( gauss_pyr, pozX, pozY, ImageWidth, ImageHeight, orients[iteratorOrient], scl_octv, hist2, SIFT_DESCR_WIDTH, SIFT_DESCR_HIST_BINS );


							int k = 0;
							float desc[128];
							
							for(int ii = 0; ii < SIFT_DESCR_WIDTH; ii++)
								for(int iii = 0; iii < SIFT_DESCR_WIDTH; iii++)
									for(int iiii = 0; iiii < SIFT_DESCR_HIST_BINS; iiii++)
										desc[k++] = hist2[ii][iii][iiii];
							
							normalize_descr( desc );


							for(int i = 0; i < k; i++ )
							{
								if( desc[i] > SIFT_DESCR_MAG_THR )
									desc[i] = SIFT_DESCR_MAG_THR;
							}

							normalize_descr( desc );



							/* convert floating-point descriptor to integer valued descriptor */
							for(int i = 0; i < k; i++ )
							{
								desc[i] = MIN( 255, (int)(SIFT_INT_DESCR_FCTR * desc[i]) );
							}

							int offset = 139;

							numberExt = (*number)++;

							keys[numberExt*offset] = scx;
							keys[numberExt*offset + 1] = scy;
							keys[numberExt*offset + 2] = x;
							keys[numberExt*offset + 3] = y;
							keys[numberExt*offset + 4] = subintvl;
							keys[numberExt*offset + 5] = intvlRes;
							keys[numberExt*offset + 6] = octvRes;
							keys[numberExt*offset + 7] = scl;
							keys[numberExt*offset + 8] = scl_octv;
							keys[numberExt*offset + 9] = orients[iteratorOrient];
							keys[numberExt*offset + 10] = omax;

							for(int i = 0; i < k; i++ )
								keys[numberExt*offset + 11 + i] = desc[i];

						}


					}
				}
				
			} else {
				//ucDest[GMEMOffset] = 0.5;
				//atomic_add(numberRej, (int)1);
			}
		}
	} else {
		
	}

}