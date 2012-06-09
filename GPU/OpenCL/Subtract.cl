

__kernel void ckSub(__global float* ucSource, __global float* ucDest, int OffsetPrev, int OffsetAct,
					   int ImageWidth, int ImageHeight)
{
		int pozX = 0;	
		int pozY = 0;
		int punktOffset = 0;
		int punktOffsetNext = 0;

		pozX = get_global_id(0) > ImageWidth  ? ImageWidth  : get_global_id(0);
		pozY = get_global_id(1) > ImageHeight ? ImageHeight : get_global_id(1);
		
		punktOffset = OffsetPrev + mul24(pozY, ImageWidth) + pozX;
		punktOffsetNext = OffsetAct + mul24(pozY, ImageWidth) + pozX;

		float res = ucSource[punktOffsetNext] - ucSource[punktOffset];

		if((pozY <= ImageHeight) && (pozX <= ImageWidth))
		{
			ucDest[punktOffset] = res;
		}

}