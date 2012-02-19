

int Offset(int x, int y, int ImageWidth)
{
	int offset = mul24( (int)y, ImageWidth ) + x;
	return offset;
}