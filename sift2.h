/**@file
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

#ifndef SIFT2_H
#define SIFT2_H

#include "GPU\MeanFilter.h"
#include "GPU\Subtract.h"
#include "GPU\DetectExtrema.h"
#include "cxcore.h"
#include "imgfeatures.h"


using namespace std;



/******************************** Structures *********************************/

/** holds feature data relevant to detection */
typedef struct detection_data
{
	int r;
	int c;
	int octv;
	int intvl;
	float subintvl;
	float scl_octv;
} detection_data;



/******************************* Defs and macros *****************************/

/** default number of sampled intervals per octave */
#define SIFT_INTVLS		3

/** default sigma for initial gaussian smoothing */
#define SIFT_SIGMA		1.6

/** default threshold on keypoint contrast |D(x)| */
#define SIFT_CONTR_THR	0.04

/** default threshold on keypoint ratio of principle curvatures */
#define SIFT_CURV_THR	10

/** float image size before pyramid construction? */
#define SIFT_IMG_DBL	1

/** default width of descriptor histogram array */
#define SIFT_DESCR_WIDTH 4

/** default number of bins per histogram in descriptor array */
#define SIFT_DESCR_HIST_BINS 8

/* assumed gaussian blur for input image */
#define SIFT_INIT_SIGMA 0.5

/* width of border in which to ignore keypoints */
#define SIFT_IMG_BORDER 5

/* maximum steps of keypoint interpolation before failure */
#define SIFT_MAX_INTERP_STEPS 5

/* default number of bins in histogram for orientation assignment */
#define SIFT_ORI_HIST_BINS 36

/* determines gaussian sigma for orientation assignment */
#define SIFT_ORI_SIG_FCTR 1.5

/* determines the radius of the region used in orientation assignment */
#define SIFT_ORI_RADIUS 3.0 * SIFT_ORI_SIG_FCTR

/* number of passes of orientation histogram smoothing */
#define SIFT_ORI_SMOOTH_PASSES 2

/* orientation magnitude relative to max that results in new feature */
#define SIFT_ORI_PEAK_RATIO 0.8

/* determines the size of a single descriptor orientation histogram */
#define SIFT_DESCR_SCL_FCTR 3.0

/* threshold on magnitude of elements of descriptor vector */
#define SIFT_DESCR_MAG_THR 0.2

/* factor used to convert floating-point descriptor to unsigned char */
#define SIFT_INT_DESCR_FCTR 512.0

/* returns a feature's detection data */
#define feat_detection_data(f) ( (struct detection_data*)(f->feature_data) )

#define ROUND(x) ( ( x - (int)x ) <= 0.5 ? (int)x :  (int)x + 1 )

#define	SIFTCPU		0


class SIFTGPU
{
	public:

		char* img_file_name;
		char* out_file_name;
		char* out_img_name;
		int display;
		int intvls;
		float sigma;
		float contr_thr;
		int curv_thr;
		int img_dbl;
		int descr_width;
		int descr_hist_bins;

		feature* featureGPU;
		int iteratorFGPU;

		MeanFilter* meanFilter;
		Subtract* subtract;
		DetectExtrema* detectExt;
		

		IplImage*** gauss_pyr;


		SIFTGPU();

		void DoSift();

		int _sift_features( IplImage* img, feature** feat, int intvls,
						  float sigma, float contr_thr, int curv_thr,
						  int img_dbl, int descr_width, int descr_hist_bins );

		int sift_features( IplImage* img, feature** feat );

		IplImage* createInitImg( IplImage*, int, float );

		IplImage*** buildGaussPyr( IplImage*, int, int, float );

		IplImage*** buildDogPyr( IplImage***, int, int );

		CvSeq* scaleSpaceExtrema( IplImage***, int, int, float, int, CvMemStorage*);

		void calcFeatureScales( CvSeq*, float, int );

		void adjustForImgDbl( CvSeq* );

		void CalcFeatureOris( CvSeq*, IplImage*** );
};



#endif



