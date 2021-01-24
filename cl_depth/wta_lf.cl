//WTA for LF
__kernel void winnerTakesAll_LF(__global float * datacost, __global uchar4 * dispFinal, int dispRange, int dispScale)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int width = get_global_size(0);
	int height = get_global_size(1);

	float minCost = FLT_MAX;
	uchar4 disparity = 0;
	for (int d = 0; d < dispRange; d++)
	{
		float cost = datacost[y + x * height + width * height * (d)];
		if (minCost > cost)
		{
			minCost = cost;
			disparity = (d)+1;
		}
	}
	disparity.w = -255;
	//dispFinal[y + x * height] =  (disparity)* (uchar4) dispScale ;
	dispFinal[y + x * height] = (uchar4)(256, 256, 256, 0) - (disparity);
}