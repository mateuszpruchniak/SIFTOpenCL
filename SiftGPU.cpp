#include "SiftGPU.h"
#include "imgfeatures.h"
#include "utils.h"
#include <cxcore.h>
#include <cv.h>



 SiftGPU::SiftGPU()
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
	 detectExt = new DetectExtrema(SIFT_MAX_NUMBER_KEYS);
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
int FeatureCmp( void* feat1, void* feat2, void* param )
 {
	 feature* f1 = (feature*) feat1;
	 feature* f2 = (feature*) feat2;

	 if( f1->scl < f2->scl )
		 return 1;
	 if( f1->scl > f2->scl )
		 return -1;
	 return 0;
 }


 int SiftGPU::DoSift( IplImage* img )
 {
	printf("\n ----------- DoSift START --------------- \n");
	IplImage* init_img;
	IplImage*** dog_pyr;
	CvMemStorage* storage;
	CvSeq* features;
	int octvs, i, n = 0;
	
	meanFilter->CreateBuffersIn(4*img->width*img->height*sizeof(float),3);
	meanFilter->CreateBuffersOut(4*img->width*img->height*sizeof(float),1);

	/* check arguments */
	if( ! img )
		printf( "NULL pointer error, %s, line %d",  __FILE__, __LINE__ );

	

	/* build scale space pyramid; smallest dimension of top level is ~4 pixels */

	init_img = CreateInitialImg( img, img_dbl, sigma );

	
	octvs = log( (float)MIN( init_img->width, init_img->height ) ) / log((float)2) - 2;
	
	cl_mem cmBufPyramid = BuildGaussPyr( init_img, octvs, intvls, sigma );
	
	
	dog_pyr = BuildDogPyr( cmBufPyramid, octvs, intvls );
	
	storage = cvCreateMemStorage( 0 );
	

	features = ScaleSpaceExtrema( dog_pyr, octvs, intvls, contr_thr, curv_thr, storage );


	/* sort features by decreasing scale and move from CvSeq to array */
	
	
	cvSeqSort( features, (CvCmpFunc)FeatureCmp, NULL );
	n = features->total;
	feat = (feature*)calloc( n, sizeof(feature) );
	feat = (feature*)cvCvtSeqToArray( features, feat, CV_WHOLE_SEQ );
	for( i = 0; i < n; i++ )
	{
		free( feat[i].feature_data );
		feat[i].feature_data = NULL;
	}

	cvReleaseMemStorage( &storage );
	cvReleaseImage( &init_img );
	ReleasePyr( &gauss_pyr, octvs, intvls + 3 );
	ReleasePyr( &dog_pyr, octvs, intvls + 2 );
	
	printf("Found: %d \n", n);
	printf("\n ----------- DoSift End --------------- \n");
	
	return n;
 }




 /*
 De-allocates memory held by a scale space pyramid

 @param pyr scale space pyramid
 @param octvs number of octaves of scale space
 @param n number of images per octave
 */
 void SiftGPU::ReleasePyr( IplImage**** pyr, int octvs, int n )
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
 CvSeq* SiftGPU::ScaleSpaceExtrema( IplImage*** dog_pyr, int octvs, int intvls,
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


	int numberExtrema = 0;
	int number = 0;
	int numberRej = 0;
	
	IplImage* img = cvCreateImage( cvGetSize(dog_pyr[0][0]), 32, 1 );

	cvZero(img);

	features = cvCreateSeq( 0, sizeof(CvSeq), sizeof(feature), storage );

	int total = features->total;

	/************************ GPU **************************/
	//detectExt->CreateBuffersIn(dog_pyr[0][0]->width*dog_pyr[0][0]->height*sizeof(float),4);
	//detectExt->CreateBuffersOut(img->width*img->height*sizeof(float),1);
	/************************ GPU **************************/

	for( o = 0; o < octvs; o++ )
		for( i = 1; i <= intvls; i++ )
		{
			
			if(SIFTCPU)
			{
				
				Keys keys[1000];

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
					
						if( abs( pixval32f( dog_pyr[o][i], r, c ) ) > prelim_contr_thr )
						{
							if( IsExtremum( dog_pyr, o, i, r, c ) )
							{

								feat = InterpExtremum(dog_pyr, o, i, r, c, intvls, contr_thr);
								if( feat )
								{
									ddata = FeatDetectionData( feat );

									if( ! IsTooEdgeLike( dog_pyr[ddata->octv][ddata->intvl],
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
				/************************ GPU **************************/
				
				num = 0;
				numRemoved = 0;

				int total = features->total;

				Keys keysArray[SIFT_MAX_NUMBER_KEYS];
				for (int i =0 ; i < SIFT_MAX_NUMBER_KEYS ; i++)
				{
					keysArray[i].x = 0.0;
					keysArray[i].y = 0.0;
					keysArray[i].intvl = 0.0;
					keysArray[i].octv = 0.0;
					keysArray[i].subintvl = 0.0;
					keysArray[i].scx = 0.0;
					keysArray[i].scy = 0.0;
				}
				detectExt->SendImageToBuffers(3,dog_pyr[o][i-1],dog_pyr[o][i],dog_pyr[o][i+1], gauss_pyr[o][i]);
				detectExt->Process(&num, &numRemoved, prelim_contr_thr, i, o, gauss_pyr[o][i], keysArray);
				//detectExt->ReceiveImageData(img);
				
				total = features->total;

				number = num;

				struct detection_data* ddata;

				

				for(int ik = 0; ik < number ; ik++)
				{ 
					feat = NewFeature();
					ddata = FeatDetectionData( feat );
					feat->img_pt.x = feat->x = keysArray[ik].scx;
					feat->img_pt.y = feat->y = keysArray[ik].scy;
					ddata->r = keysArray[ik].y;
					ddata->c = keysArray[ik].x;
					ddata->subintvl = keysArray[ik].subintvl;
					ddata->octv = keysArray[ik].octv;
					ddata->intvl = keysArray[ik].intvl;
					feat->scl = keysArray[ik].scl;
					ddata->scl_octv = keysArray[ik].scl_octv;
					feat->ori = keysArray[ik].ori;
					feat->d = 128;

					for(int i = 0; i < 128 ; i++ )
					{
						feat->descr[i] = keysArray[ik].desc[i];
					}

					cvSeqPush( features, feat );
					free( feat );
				}
				
			}
			/************************ GPU **************************/
		}

	return features;
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
int SiftGPU::IsTooEdgeLike( IplImage* dog_img, int r, int c, int curv_thr )
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
 int SiftGPU::IsExtremum( IplImage*** dog_pyr, int octv, int intvl, int r, int c )
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
 feature* SiftGPU::InterpExtremum( IplImage*** dog_pyr, int octv, int intvl,
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
		InterpStep( dog_pyr, octv, intvl, r, c, &xi, &xr, &xc );
		if( abs( xi ) < 0.5  &&  abs( xr ) < 0.5  &&  abs( xc ) < 0.5 )
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

	contr = InterpContr( dog_pyr, octv, intvl, r, c, xi, xr, xc );
	if( abs( contr ) < contr_thr / intvls )
		return NULL;

	feat = NewFeature();
	ddata = FeatDetectionData( feat );
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

 void SiftGPU::InterpStep( IplImage*** dog_pyr, int octv, int intvl, int rr, int cc,
						 float* xi, float* xr, float* xc )
{
	CvMat* dD, X;
	float x[3] = { 0, 0 , 0 };
	float xx[3] = { 0, 0 , 0 };

	float H[3][3];
	float H_inv[3][3];

	dD = Deriv3D( dog_pyr, octv, intvl, rr, cc );
	Hessian3D( dog_pyr, octv, intvl, rr, cc, H);

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
 CvMat* SiftGPU::Deriv3D( IplImage*** dog_pyr, int octv, int intvl, int r, int c )
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
void SiftGPU::Hessian3D( IplImage*** dog_pyr, int octv, int intvl, int r, int c, float H[][3] )
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
 float SiftGPU::InterpContr( IplImage*** dog_pyr, int octv, int intvl, int r,
							int c, float xi, float xr, float xc )
{
	CvMat* dD, X, T;
	float t[1], x[3] = { xc, xr, xi };

	cvInitMatHeader( &X, 3, 1, CV_64FC1, x, CV_AUTOSTEP );
	cvInitMatHeader( &T, 1, 1, CV_64FC1, t, CV_AUTOSTEP );
	dD = Deriv3D( dog_pyr, octv, intvl, r, c );

	//cvGEMM( dD, &X, 1, NULL, 0, &T,  CV_GEMM_A_T );
	t[0] = cvGetReal2D(dD, 0, 0) * x[0] + cvGetReal2D(dD, 1, 0) * x[1] + cvGetReal2D(dD, 2, 0) * x[2];

	cvReleaseMat( &dD );

	return pixval32f( dog_pyr[octv][intvl], r, c ) + t[0] * 0.5;
}



/*
Allocates and initializes a new feature

@return Returns a pointer to the new feature
*/
 feature* SiftGPU::NewFeature( void )
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
Builds a difference of Gaussians scale space pyramid by subtracting adjacent
intervals of a Gaussian pyramid

@param gauss_pyr Gaussian scale-space pyramid
@param octvs number of octaves of scale space
@param intvls number of intervals per octave

@return Returns a difference of Gaussians scale space pyramid as an
	octvs x (intvls + 2) array
*/
 IplImage*** SiftGPU::BuildDogPyr( cl_mem cmBufPyramid, int octvs, int intvls )
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
				subtract->SendImageToBuffers(2,gauss_pyr[o][i+1],gauss_pyr[o][i]);
				subtract->Process();
				subtract->ReceiveImageData(1,dog_pyr[o][i]);
			}
			/************************ GPU **************************/

			/*cvNamedWindow( "sub", 1 );
			cvShowImage( "sub", dog_pyr[o][i] );
			cvWaitKey( 0 );*/
			
		}

	return dog_pyr;
}



 /*
Builds Gaussian scale space pyramid from an image

@param base base image of the pyramid
@param octvs number of octaves of scale space
@param intvls number of intervals per octave
@param sigma amount of Gaussian smoothing per octave

@return Returns a Gaussian scale space pyramid as an octvs x (intvls + 3) array
*/
 cl_mem SiftGPU::BuildGaussPyr( IplImage* base, int octvs,
									int intvls, float sigma )
{
	float* sig = (float*)calloc( intvls + 3, sizeof(float));
	float sig_total, sig_prev, k;
	int i, o;
	int intvlsSum = intvls + 3;
	int* sizeOfImages = new int[octvs];
	int* imageWidth = new int[octvs];
	int* imageHeight = new int[octvs];
	float SumOfPyramid = 0;

	gauss_pyr = (IplImage***)calloc( octvs, sizeof( IplImage** ) );
	for( i = 0; i < octvs; i++ )
		gauss_pyr[i] = (IplImage**)calloc( intvlsSum, sizeof( IplImage* ) );

	/*
		precompute Gaussian sigmas using the following formula:

		\sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
	*/

	sig[0] = sigma;
	k = pow( 2.0, 1.0 / intvls );
	for( i = 1; i < intvlsSum; i++ )
	{
		sig_prev = pow( k, i - 1 ) * sigma;
		sig_total = sig_prev * k;
		sig[i] = sqrt( sig_total * sig_total - sig_prev * sig_prev );
	}

	
	gauss_pyr[0][0] = cvCloneImage(base);
	

	sizeOfImages[0] = gauss_pyr[0][0]->imageSize;
	SumOfPyramid += gauss_pyr[0][0]->imageSize * intvlsSum;
	imageHeight[0] = gauss_pyr[0][0]->height;
	imageWidth[0] = gauss_pyr[0][0]->width;

	for( o = 1; o < octvs; o++ )
	{
		gauss_pyr[o][0] = Downsample( gauss_pyr[o-1][0] );
		SumOfPyramid += gauss_pyr[o][0]->imageSize * intvlsSum;
		sizeOfImages[o] = gauss_pyr[o][0]->imageSize;
		imageHeight[o] = gauss_pyr[o][0]->height;
		imageWidth[o] = gauss_pyr[o][0]->width;
	}



	meanFilter->CreateBuffer(SumOfPyramid);

	int offset = 0;

	for( o = 0; o < octvs; o++ )
	{
		for( i = 0; i < intvlsSum; i++ )
		{
			if( o == 0  &&  i == 0 )
			{

			} else if(i == 0)
			{
				gauss_pyr[o][i] = Downsample( gauss_pyr[o-1][intvls] );
			} else {
				gauss_pyr[o][i] = gauss_pyr[o][i-1];
			}

			meanFilter->SendImageToBufPyramid(gauss_pyr[o][i], offset, sizeOfImages);
			
			offset += sizeOfImages[o];

		}
	}


	

	offset = 0;
	int OffsetAct = 0;
	int OffsetPrev = 0;

	for( o = 0; o < octvs; o++ )
	{
		for( i = 0; i < intvlsSum; i++ )
		{
			if(i > 0)
				meanFilter->Process( sig[i], gauss_pyr[o][i]->width, gauss_pyr[o][i]->height, OffsetAct, OffsetPrev);

			//meanFilter->ReceiveImageToBufPyramid(gauss_pyr[o][i], OffsetAct, sizeOfImages);

			OffsetPrev = OffsetAct;
			OffsetAct += sizeOfImages[o];
		}
	}

	

	cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << endl;

	free( sig );	
	return meanFilter->cmBufPyramid;
}


 /*
 Downsamples an image to a quarter of its size (half in each dimension)
 using nearest-neighbor interpolation

 @param img an image

 @return Returns an image whose dimensions are half those of img
 */
 IplImage* SiftGPU::Downsample( IplImage* img )
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
 Converts an image to 8-bit grayscale and Gaussian-smooths it.  The image is
 optionally doubled in size prior to smoothing.

 @param img input image
 @param img_dbl if true, image is doubled in size prior to smoothing
 @param sigma total std of Gaussian smoothing
 */
 IplImage* SiftGPU::CreateInitialImg( IplImage* img, int img_dbl, float sigma )
 {
	 IplImage* gray, * dbl;
	 float sig_diff;

	 gray = ConvertToGray32( img );
	 if( img_dbl )
	 {
		 sig_diff = sqrt( sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4 );
		 dbl = cvCreateImage( cvSize( img->width*2, img->height*2 ),
			 32, 1 );
		 cvResize( gray, dbl, CV_INTER_CUBIC );

		 /************************ GPU **************************/
		 //if(SIFTCPU)
			 cvSmooth( dbl, dbl, CV_GAUSSIAN, 0, 0, sig_diff, sig_diff );
		 //else
		 //{
			// //meanFilter->CreateBuffersIn(dbl->width*dbl->height*sizeof(float),1);
			// //meanFilter->CreateBuffersOut(dbl->width*dbl->height*sizeof(float),1);
			// meanFilter->SendImageToBuffers(1,dbl);
			// 
			// 
			// 
			// 
			// 
			// 
			// 
			// 
			// //meanFilter->Process(sig_diff);
			// 
			// 
			// 
			// 
			// 
			// 
			// 
			// 
			// 
			// 
			// 
			// meanFilter->ReceiveImageData(1,dbl);
		 //}
		 /************************ GPU **************************/

		 cvReleaseImage( &gray );
		 return dbl;
	 }
	 else
	 {
		 sig_diff = sqrt( sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA );

		 /************************ GPU **************************/
		 //if(SIFTCPU)
			 cvSmooth( gray, gray, CV_GAUSSIAN, 0, 0, sig_diff, sig_diff );
		 //else
		 //{
			// //meanFilter->CreateBuffersIn(gray->width*gray->height*sizeof(float),1);
			// //meanFilter->CreateBuffersOut(gray->width*gray->height*sizeof(float),1);
			// meanFilter->SendImageToBuffers(1,gray);
			// 
			// 



			// 
			// //meanFilter->Process(sig_diff);
			// 
			// 
			// 
			// 
			// 
			// 
			// 
			// 
			// 
			// 
			// meanFilter->ReceiveImageData(1,gray);
		 //}
		 /************************ GPU **************************/

		 return gray;
	 }
 }


 /*
 Converts an image to 32-bit grayscale

 @param img a 3-channel 8-bit color (BGR) or 8-bit gray image

 @return Returns a 32-bit grayscale image
 */
 IplImage* SiftGPU::ConvertToGray32( IplImage* img )
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