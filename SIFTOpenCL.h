
#ifndef SIFTOPENCL_H
#define SIFTOPENCL_H



#include "GPU\MeanFilter.h"
#include "GPU\Subtract.h"
#include "GPU\DetectExtrema.h"
#include "cxcore.h"
#include "imgfeatures.h"


using namespace std;



/******************************** Structures *********************************/

///** holds feature data relevant to detection */
struct detection_data
{
	int r;
	int c;
	int octv;
	int intvl;
	float subintvl;
	float scl_octv;
};


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
#define FeatDetectionData(f) ( (struct detection_data*)(f->feature_data) )

#define ROUND(x) ( ( x - (int)x ) <= 0.5 ? (int)x :  (int)x + 1 )

#define	SIFTCPU		0




class SIFTOpenCL
{
private:

	char* img_file_name;
	char* out_file_name;
	char* out_img_name;
	int intvls;
	float sigma;
	float contr_thr;
	int curv_thr;
	int img_dbl;
	int descr_width;
	int descr_hist_bins;
	MeanFilter* meanFilter;
	Subtract* subtract;
	DetectExtrema* detectExt;
	
	
	IplImage*** gauss_pyr;

	IplImage* CreateInitialImg( IplImage* img, int img_dbl, float sigma );
	IplImage* ConvertToGray32( IplImage* img );
	IplImage*** BuildGaussPyr( IplImage* base, int octvs, int intvls, float sigma );
	IplImage* Downsample( IplImage* img );
	IplImage*** BuildDogPyr( IplImage*** gauss_pyr, int octvs, int intvls );
	CvSeq* ScaleSpaceExtrema( IplImage*** dog_pyr, int octvs, int intvls, float contr_thr, int curv_thr, CvMemStorage* storage );
	feature* NewFeature( void );
	float InterpContr( IplImage*** dog_pyr, int octv, int intvl, int r, int c, float xi, float xr, float xc );
	void Hessian3D( IplImage*** dog_pyr, int octv, int intvl, int r, int c, float H[][3] );
	CvMat* Deriv3D( IplImage*** dog_pyr, int octv, int intvl, int r, int c );
	void InterpStep( IplImage*** dog_pyr, int octv, int intvl, int rr, int cc, float* xi, float* xr, float* xc );
	feature* InterpExtremum( IplImage*** dog_pyr, int octv, int intvl, int r, int c, int intvls, float contr_thr );
	int IsExtremum( IplImage*** dog_pyr, int octv, int intvl, int r, int c );
	int IsTooEdgeLike( IplImage* dog_img, int r, int c, int curv_thr );
	//int FeatureCmp( void* feat1, void* feat2, void* param );
	void ReleasePyr( IplImage**** pyr, int octvs, int n );

public:

	feature* feat;

	SIFTOpenCL();

	int DoSift(IplImage* img);
	









};



#endif