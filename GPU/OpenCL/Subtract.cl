

__kernel void ckSub(__global float* ucSource, __global float* ucSource2, __global float* ucDest,
                      int ImageWidth, int ImageHeight)
{
	    int iImagePosX = get_global_id(0) > ImageWidth  ? ImageWidth  : get_global_id(0);
	    int iDevYPrime = get_global_id(1) > ImageHeight ? ImageHeight : get_global_id(1);
		
		int iDevGMEMOffset = mul24(iDevYPrime, ImageWidth) + iImagePosX;
		
		
		float res = ucSource2[iDevGMEMOffset] - ucSource[iDevGMEMOffset];
		

		// Write out to GMEM with restored offset
		if((iDevYPrime <= ImageHeight) && (iImagePosX <= ImageWidth))
		{
			ucDest[iDevGMEMOffset] = res;
		}
}
