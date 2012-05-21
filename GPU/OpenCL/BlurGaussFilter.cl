

#define ROUND(x) ( ( x - (int)x ) <= 0.5 ? (int)x :  (int)x + 1 )



__kernel void ckConv(__global float* ucSource, __global float* ucDest, int octvs, int intvlsSum,
					  __global int* ImageWidth, __global int* ImageHeight, __global float* sigma, int nChannels)
{
		int pozX = 0;	
		int pozY = 0;
		int maskSize = 0;
		int Offset = 0;
		int offset = 0;
		float pi = 3.1415926535897932384626433832795;
		int r = 0;
		float sum = 0.0;
		float sumG = 0.0;
		float G = 0.0;
		int punktOffset = 0;

		// rozmywanie na podstawie poprzedniego!!!!!!!!!!!!

		for(int o = 0; o < octvs; o++ )
		{
			for(int i = 0; i < intvlsSum; i++ )
			{
				pozX = get_global_id(0) > ImageWidth[o]  ? ImageWidth[o]  : get_global_id(0);
				pozY = get_global_id(1) > ImageHeight[o] ? ImageHeight[o] : get_global_id(1);
				maskSize = ROUND(sigma[i] * 3 * 2 + 1) | 1;

				r = (int)floor( (float)maskSize / 2 );
				punktOffset = Offset + mul24(pozY, ImageWidth[o]) + pozX;

				for(int j = -r ; j <= r; j++ ) //y
				{
					for(int ii = -r ; ii <= r; ii++ ) //x
					{
						G = exp((float)((-1.0) * (float)(ii * ii + j * j) / (2.0 * sigma[i] * sigma[i]) ));
						sumG += G;
						int x = pozX + ii >= 0 && pozX + ii <= ImageWidth[o]  ? pozX + ii : 0;
						int y = pozY + j >= 0 && pozY + j <= ImageHeight[o] ? pozY + j : 0;

						offset = Offset + mul24(y, ImageWidth[o]) + x;
						sum += G * ucSource[offset];
					}
				}

				Offset += ImageWidth[o] * ImageHeight[o];


				if((pozY <= ImageHeight[o]) && (pozX <= ImageWidth[o]))
				{
					ucDest[punktOffset] = sum / (2.0 * pi * sigma[i] * sigma[i]);
				}

			}
		}
		


		
		
		

		
		

}
