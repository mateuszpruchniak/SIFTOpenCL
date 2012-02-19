
int Offset(int x, int y, int ImageWidth)
{
	int offset = mul24( (int)y, ImageWidth ) + x;
	return offset;
}


__kernel void ckConv(__global float* ucSource,__global float* ucDest,
                      int ImageWidth, int ImageHeight, float sigma, int sizeMask)
{
	    int iImagePosX = get_global_id(0) > ImageWidth  ? ImageWidth  : get_global_id(0);
	    int iDevYPrime = get_global_id(1) > ImageHeight ? ImageHeight : get_global_id(1);
		
		int iDevGMEMOffset = mul24(iDevYPrime, ImageWidth) + iImagePosX;
		
		float pi = 3.1415926535897932384626433832795;

	    int r = (int)floor( (float)sizeMask/2 );
		
		float sum = 0.0;
		float sumG = 0.0;
		float G = 0.0;

		for(int j = -r ; j <= r; j++ ) //y
		{
			for(int i = -r ; i <= r; i++ ) //x
			{
				G = exp((float)((-1.0) * (float)(i * i + j * j) / (2.0 * sigma * sigma) ));
				sumG += G;
				int x = iImagePosX + i >= 0 && iImagePosX + i <= ImageWidth  ? iImagePosX + i : 0;
				int y = iDevYPrime + j >= 0 && iDevYPrime + j <= ImageHeight ? iDevYPrime + j : 0;
				int localOffest = Offset(x,y,ImageWidth);
				sum += G * ucSource[localOffest];
			}
		}
		

		// Write out to GMEM with restored offset
		if((iDevYPrime <= ImageHeight) && (iImagePosX <= ImageWidth))
		{
			ucDest[iDevGMEMOffset] = sum / (2.0 * pi * sigma * sigma);
		}
}
