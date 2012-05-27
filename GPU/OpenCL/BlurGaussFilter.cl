
#define ROUND(x) ( ( x - (int)x ) <= 0.5 ? (int)x :  (int)x + 1 )



__kernel void ckConv(__global float* ucSource, int Offset, int OffsetPrev,
					   int ImageWidth, int ImageHeight, float sigma)
{
		int pozX = 0;	
		int pozY = 0;
		int maskSize = 0;
		float pi = 3.1415926535897932384626433832795;
		int r = 0;
		float sum = 0.0;
		float sumG = 0.0;
		float G = 0.0;
		int punktOffset = 0;


		pozX = get_global_id(0) > ImageWidth  ? ImageWidth  : get_global_id(0);
		pozY = get_global_id(1) > ImageHeight ? ImageHeight : get_global_id(1);
		
		punktOffset = Offset + mul24(pozY, ImageWidth) + pozX;

		maskSize = ROUND(sigma * 3 * 2 + 1) | 1;



		r = (int)floor( (float)maskSize / 2 );

		for(int j = -r ; j <= r; j++ ) //y
		{
			for(int ii = -r ; ii <= r; ii++ ) //x
			{
				G = exp((float)((-1.0) * (float)(ii * ii + j * j) / (2.0 * sigma * sigma) ));
				sumG += G;
				int x = pozX + ii >= 0 && pozX + ii <= ImageWidth  ? pozX + ii : 0;
				int y = pozY + j >= 0 && pozY + j <= ImageHeight ? pozY + j : 0;

				int offset = OffsetPrev + mul24(y, ImageWidth) + x;
				sum += G * ucSource[offset];
			}
		}


		if((pozY <= ImageHeight) && (pozX <= ImageWidth))
		{
			ucSource[punktOffset] = sum / (2.0 * pi * sigma * sigma);
		}

}