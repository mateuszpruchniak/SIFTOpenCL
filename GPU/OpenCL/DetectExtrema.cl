
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
#define SIFT_ORI_PEAK_RATIO 0.98

/* determines the size of a single descriptor orientation histogram */
#define SIFT_DESCR_SCL_FCTR 3.0

/* threshold on magnitude of elements of descriptor vector */
#define SIFT_DESCR_MAG_THR 0.2

/* factor used to convert floating-point descriptor to unsigned char */
#define SIFT_INT_DESCR_FCTR 512.0

/* returns a feature's detection data */
#define feat_detection_data(f) ( (struct detection_data*)(f->feature_data) )

#define CV_PI   3.1415926535897932384626433832795

/* absolute value */
#define ABS(x) ( ( (x) < 0 )? -(x) : (x) )

#define ROUND(x) ( ( x - (int)x ) <= 0.5 ? (int)x :  (int)x + 1 )

/*
Interpolates a histogram peak from left, center, and right values
*/
#define interp_hist_peak( l, c, r ) ( 0.5 * ((l)-(r)) / ((l) - 2.0*(c) + (r)) )




float GetPixel(__global float* dataIn, int x, int y, int ImageWidth, int ImageHeight )
{
	int X = x > ImageWidth  ? ImageWidth  : x;
	int Y = y > ImageHeight ? ImageHeight : y;
	int GMEMOffset = mul24(Y, ImageWidth) + X;

	return dataIn[GMEMOffset];
}

float GetPixelTEX( __read_only image2d_t dataIn, int x, int y, int ImageWidth, int ImageHeight )
{
	int2 pos = {x , y};
	sampler_t mySampler = CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE;

	return read_imagef(dataIn, mySampler, pos).x;
}


/*
Determines whether a pixel is a scale-space extremum by comparing it to it's
3x3x3 pixel neighborhood.

@return Returns 1 if the specified pixel is an extremum (max or min) among
	it's 3x3x3 pixel neighborhood.
*/
int is_extremum(__global float* dataIn1, __global float* dataIn2, __global float* dataIn3, int pozX, int pozY, int ImageWidth, int ImageHeight )
{
	float val = GetPixel(dataIn2, pozX, pozY, ImageWidth, ImageHeight);
	
	if( val > 0.0 )
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

	
	
	return 1;
}

/*
Computes the partial derivatives in x, y, and scale of a pixel in the DoG
scale space pyramid
*/
void deriv_3D( __global float* dataIn1, __global float* dataIn2, __global float* dataIn3, int pozX, int pozY, int ImageWidth, int ImageHeight, float* dI )
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
void hessian_3D( __global float* dataIn1, __global float* dataIn2, __global float* dataIn3, int pozX, int pozY, int ImageWidth, int ImageHeight, float H[][3] )
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
void interp_step(__global float* dataIn1, __global float* dataIn2, __global float* dataIn3, int pozX, int pozY, int ImageWidth, int ImageHeight,
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
float interp_contr(__global float* dataIn1, __global float* dataIn2, __global float* dataIn3, int pozX, int pozY, int ImageWidth, int ImageHeight, float xi, float xr, float xc )
{
	float dD[3] = { 0, 0, 0 };
	deriv_3D(dataIn1, dataIn2, dataIn3, pozX, pozY, ImageWidth, ImageHeight, dD);
	float res = xc*dD[0] + xr*dD[1] + xi*dD[2];

	return GetPixel(dataIn2, pozX, pozY, ImageWidth, ImageHeight) + res * 0.5;
}


float interp_extremum(__global float* dataIn1, __global float* dataIn2, __global float* dataIn3, int pozX, int pozY, int ImageWidth, int ImageHeight, 
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
 int is_too_edge_like(__global float* dataIn2, int pozX, int pozY, int ImageWidth, int ImageHeight, int curv_thr )
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
 int calc_grad_mag_ori(__global float* gauss_pyr, int pozX, int pozY, int ImageWidth, int ImageHeight, float* mag, float* ori )
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

int calc_grad_mag_oriTEX(__read_only image2d_t gauss_pyr, int pozX, int pozY, int ImageWidth, int ImageHeight, float* mag, float* ori )
{
	float dx, dy;

	if( pozX > 0  &&  pozX < ImageWidth - 1  &&  pozY > 0  &&  pozY < ImageHeight - 1 )
	{
		dx = GetPixelTEX(gauss_pyr, pozX+1, pozY, ImageWidth, ImageHeight) - GetPixelTEX(gauss_pyr, pozX-1, pozY, ImageWidth, ImageHeight);
		dy =  GetPixelTEX(gauss_pyr, pozX, pozY-1, ImageWidth, ImageHeight) - GetPixelTEX(gauss_pyr, pozX, pozY+1, ImageWidth, ImageHeight);
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
void ori_hist( __read_only image2d_t gauss_pyr, int pozX, int pozY, int ImageWidth, int ImageHeight, float* hist, int n, int rad, float sigma)
{
	float mag, ori, w, exp_denom, PI2 = CV_PI * 2.0;
	int bin, i, j;

	exp_denom = 2.0 * sigma * sigma;

	
	for( i = -rad; i <= rad; i++ )
		for( j = -rad; j <= rad; j++ )
			if( calc_grad_mag_oriTEX( gauss_pyr, pozX + i, pozY + j, ImageWidth, ImageHeight, &mag, &ori ) )
			{	
				w = exp( -(float)( i*i + j*j ) / exp_denom );
				bin = ROUND( n * ( ori + CV_PI ) / PI2 );
				bin = ( bin < n )? bin : 0;
				hist[bin] += w * mag;
			}


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

	r0 = floor( rbin );  // floor()
	c0 = floor( cbin );
	o0 = floor( obin );
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
void descr_hist( __read_only image2d_t gauss_pyr,  int pozX, int pozY, int ImageWidth, int ImageHeight, float ori, float scl, float hist[SIFT_DESCR_WIDTH][SIFT_DESCR_WIDTH][SIFT_DESCR_HIST_BINS], int d, int n )
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
				if( calc_grad_mag_oriTEX( gauss_pyr, pozX + j, pozY + i, ImageWidth, ImageHeight, &grad_mag, &grad_ori ) )
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



__kernel void ckDetect(__global float* dataIn1, __global float* dataIn2, __global float* dataIn3, __global float* ucDest,
						__global int* numberExtrema, __global float* keys,
						int ImageWidth, int ImageHeight, float prelim_contr_thr, int intvl, int octv, __global int* number, __global int* numberRej)
{
	int pozX = get_global_id(0);
	int pozY = get_global_id(1);
	int GMEMOffset = mul24(pozY, ImageWidth) + pozX;
	
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
						float intvl2 = intvl + xi; 

						float	scx = (float)(( pozX + xc ) * pow( (float)2.0, (float)octv ) / 2.0);
						float	scy = (float)(( pozY + xr ) * pow( (float)2.0, (float)octv ) / 2.0);
						float	x = pozX;
						float	y = pozY;
						float	subintvl = xi;
						float	intvlRes = intvl;
						float	octvRes = octv;
						float	scl = (SIFT_SIGMA * pow( (float)2.0, (octv + intvl2 / (float)SIFT_INTVLS) )) / 2.0;  
						float	scl_octv = SIFT_SIGMA * pow( (float)2.0, (float)(intvl2 / SIFT_INTVLS) );
						float	ori = 0;
						float	mag = 0;

						int offset = 139;
						numberExt = atomic_add(number, (int)1);

						keys[numberExt*offset] = scx;
						keys[numberExt*offset + 1] = scy;
						keys[numberExt*offset + 2] = x;
						keys[numberExt*offset + 3] = y;
						keys[numberExt*offset + 4] = subintvl;
						keys[numberExt*offset + 5] = intvlRes;
						keys[numberExt*offset + 6] = octvRes;
						keys[numberExt*offset + 7] = scl;
						keys[numberExt*offset + 8] = scl_octv;
						keys[numberExt*offset + 9] = 0;//orients[iteratorOrient];
						keys[numberExt*offset + 10] = 0;//omax;
					}
				}
			} 
		}
	} 
}



__kernel void ckDesc( __read_only image2d_t gauss_pyr, __global float* ucDest,
						__global int* numberExtrema, __global float* keys,
						int ImageWidth, int ImageHeight, float prelim_contr_thr, int intvl, int octv, __global int* number, __global int* numberAct)
 {

	sampler_t mySampler = CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE;

	int numberExt = atomic_add(numberAct, (int)1);
	int offset = 139;
	
	int pozX = get_global_id(0);
	int pozY = get_global_id(1);
	int GMEMOffset = mul24(pozY, ImageWidth) + pozX;

	
	if( numberExt < *number)
	{
		float	scx = keys[numberExt*offset];
		float	scy = keys[numberExt*offset + 1];
		float	x = keys[numberExt*offset + 2];
		float	y = keys[numberExt*offset + 3];
		float	subintvl = keys[numberExt*offset + 4];
		float	intvlRes = keys[numberExt*offset + 5];
		float	octvRes = keys[numberExt*offset + 6];
		float	scl = keys[numberExt*offset + 7];  
		float	scl_octv = keys[numberExt*offset + 8];
		float	ori = keys[numberExt*offset + 9];
		

		float hist[SIFT_ORI_HIST_BINS];
						
		for(int j = 0; j < SIFT_ORI_HIST_BINS; j++ )
			hist[j] = 0;

		ori_hist( gauss_pyr, x, y, ImageWidth, ImageHeight, hist, SIFT_ORI_HIST_BINS,
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


		ori = orients[0];
		keys[numberExt*offset + 9] = ori;


		float hist2[SIFT_DESCR_WIDTH][SIFT_DESCR_WIDTH][SIFT_DESCR_HIST_BINS];

		
		for(int ii = 0; ii < SIFT_DESCR_WIDTH; ii++)
			for(int iii = 0; iii < SIFT_DESCR_WIDTH; iii++)
				for(int iiii = 0; iiii < SIFT_DESCR_HIST_BINS; iiii++)
					hist2[ii][iii][iiii] = 0.0;


		descr_hist( gauss_pyr, keys[numberExt*offset + 2], keys[numberExt*offset + 3], ImageWidth, ImageHeight, keys[numberExt*offset + 9], keys[numberExt*offset + 8], hist2, SIFT_DESCR_WIDTH, SIFT_DESCR_HIST_BINS );
		

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

		// convert floating-point descriptor to integer valued descriptor */
		for(int i = 0; i < k; i++ )
		{
			desc[i] = min( 255, (int)(SIFT_INT_DESCR_FCTR * desc[i]) );
		}


		for(int i = 0; i < k; i++ )
			keys[numberExt*offset + 11 + i] = desc[i];

	}


	//if( numberExt < *number)
	//{
	//	float	scx = keys[numberExt*offset];
	//	float	scy = keys[numberExt*offset + 1];
	//	float	x = keys[numberExt*offset + 2];
	//	float	y = keys[numberExt*offset + 3];
	//	float	subintvl = keys[numberExt*offset + 4];
	//	float	intvlRes = keys[numberExt*offset + 5];
	//	float	octvRes = keys[numberExt*offset + 6];
	//	float	scl = keys[numberExt*offset + 7];  
	//	float	scl_octv = keys[numberExt*offset + 8];
	//	float	ori = keys[numberExt*offset + 9];
	//	

	//	float hist[SIFT_ORI_HIST_BINS];
	//					
	//	for(int j = 0; j < SIFT_ORI_HIST_BINS; j++ )
	//		hist[j] = 0;

	//	ori_hist( gauss_pyr, x, y, ImageWidth, ImageHeight, hist, SIFT_ORI_HIST_BINS,
	//					ROUND( SIFT_ORI_RADIUS * scl_octv ),	SIFT_ORI_SIG_FCTR * scl_octv );

	//					
	//	for(int j = 0; j < SIFT_ORI_SMOOTH_PASSES; j++ )
	//		smooth_ori_hist( hist, SIFT_ORI_HIST_BINS );

	//	int maxBin = 0;

	//	float omax = dominant_ori( hist, SIFT_ORI_HIST_BINS, &maxBin );

	//	float orients[SIFT_ORI_HIST_BINS];
	//	for(int j = 0; j < SIFT_ORI_HIST_BINS; j++ )
	//		orients[j] = 0;

	//	int numberOrient = 0;

	//	add_good_ori_features(hist, SIFT_ORI_HIST_BINS,	omax * SIFT_ORI_PEAK_RATIO, orients, &numberOrient);


	//	ori = orients[0];
	//	keys[numberExt*offset + 9] = ori;


	//	float hist2[SIFT_DESCR_WIDTH][SIFT_DESCR_WIDTH][SIFT_DESCR_HIST_BINS];

	//	
	//	for(int ii = 0; ii < SIFT_DESCR_WIDTH; ii++)
	//		for(int iii = 0; iii < SIFT_DESCR_WIDTH; iii++)
	//			for(int iiii = 0; iiii < SIFT_DESCR_HIST_BINS; iiii++)
	//				hist2[ii][iii][iiii] = 0.0;


	//	descr_hist( gauss_pyr, keys[numberExt*offset + 2], keys[numberExt*offset + 3], ImageWidth, ImageHeight, keys[numberExt*offset + 9], keys[numberExt*offset + 8], hist2, SIFT_DESCR_WIDTH, SIFT_DESCR_HIST_BINS );
	//	

	//	int k = 0;
	//	float desc[128];
	//						
	//	for(int ii = 0; ii < SIFT_DESCR_WIDTH; ii++)
	//		for(int iii = 0; iii < SIFT_DESCR_WIDTH; iii++)
	//			for(int iiii = 0; iiii < SIFT_DESCR_HIST_BINS; iiii++)
	//				desc[k++] = hist2[ii][iii][iiii];
	//						
	//	normalize_descr( desc );


	//	for(int i = 0; i < k; i++ )
	//	{
	//		if( desc[i] > SIFT_DESCR_MAG_THR )
	//			desc[i] = SIFT_DESCR_MAG_THR;
	//	}

	//	normalize_descr( desc );

	//	// convert floating-point descriptor to integer valued descriptor */
	//	for(int i = 0; i < k; i++ )
	//	{
	//		desc[i] = min( 255, (int)(SIFT_INT_DESCR_FCTR * desc[i]) );
	//	}


	//	for(int i = 0; i < k; i++ )
	//		keys[numberExt*offset + 11 + i] = desc[i];

	//}
 }