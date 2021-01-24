#include "ocl.h"
#include <opencv2/opencv.hpp>
#include <chrono>
using namespace cv;

#define VISUALIZE false

double executionTime(cl::Event event);

void processLightField(Mat input, Mat &output)
{
	/*
		input, output: RGBA 4 channel PNG
	*/
	
	int width = input.cols;
	int height = input.rows;

	//Light field parameter
	int depth_resolution = 75;
	float delta = 0.008f;   //0.0214 MONA || 0.0316 PAP || 0.0324 B2 || 0.0416 B1 || 0.0732 LIFE
	// 0.0518 Medieval //
	int UV_diameter = 5;
	int UV_radius = 2;
	int w_spatial = width / UV_diameter;
	int h_spatial = height / UV_diameter;
	int totalPixels = width * height;
	int totalPixels_spatial = w_spatial * h_spatial;
	float sigma = 10.0f;
	int scale = 256 / (depth_resolution);
	float alpha;
	static const cl::ImageFormat format = { CL_RGBA, CL_UNSIGNED_INT8 };

	// ouput [4(ch) * h_spatial * w_spatial]
	output = Mat(h_spatial, w_spatial, CV_8UC4);

	vector<double> elapsed;
	elapsed.reserve(depth_resolution + 1);

	cl::Image2D imageLF = cl::Image2D(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, format, width, height, 0, input.data);
	//cl::Buffer bufferInLF = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, totalPixels * sizeof(uint32_t), bitmapLF, NULL);
	cl::Buffer bufferRemap = cl::Buffer(context, CL_MEM_READ_WRITE, (totalPixels) * sizeof(cl_float4));
	//cl::Image2D imageRemap = cl::Image2D(context, CL_MEM_READ_WRITE, format, width, height, 0, bitmapLF);
	cl::Buffer bufferResponse = cl::Buffer(context, CL_MEM_READ_WRITE, (totalPixels_spatial)* depth_resolution * sizeof(cl_float));
	cl::Buffer bufferDepth = cl::Buffer(context, CL_MEM_READ_WRITE, (totalPixels_spatial) * sizeof(cl_uchar4));

	try {
		cl::Event	gWTA_LF_evt;
		for (int index = 1; index <= depth_resolution; index++) {
			cl::Event	gRemap_Lytro_evt, gCAE_evt;

			alpha = -((index - (depth_resolution + 1) / 2) * (delta));
			//Shear the LF with alpha value
			gRemap_Lytro.setArg(0, imageLF);
			gRemap_Lytro.setArg(1, bufferRemap);
			gRemap_Lytro.setArg(2, delta);
			gRemap_Lytro.setArg(3, UV_diameter);
			gRemap_Lytro.setArg(4, UV_radius);
			gRemap_Lytro.setArg(5, alpha);
			commandQueue.enqueueNDRangeKernel(gRemap_Lytro,
				cl::NullRange, // offset
				cl::NDRange(w_spatial, h_spatial), // global size
				cl::NDRange(16, 16), // local size
				NULL,
				&gRemap_Lytro_evt);
			gRemap_Lytro_evt.wait();

			//Calculate the response using CAE
			gCAE.setArg(0, bufferRemap);
			gCAE.setArg(1, bufferResponse);
			gCAE.setArg(2, UV_diameter);
			gCAE.setArg(3, sigma);
			gCAE.setArg(4, index);
			gCAE.setArg(5, depth_resolution);
			commandQueue.enqueueNDRangeKernel(gCAE,
				cl::NullRange,
				cl::NDRange(w_spatial, h_spatial),
				cl::NDRange(16, 16),
				NULL,
				&gCAE_evt);
			gCAE_evt.wait();

			elapsed[index - 1] = executionTime(gRemap_Lytro_evt) + executionTime(gCAE_evt);
			printf("Loop#%d || Alpha: %.5f || Elapsed Time: %.4lf ms\n", index, alpha, elapsed[index - 1]);
		}

		gWTA_LF.setArg(0, bufferResponse);
		gWTA_LF.setArg(1, bufferDepth);
		gWTA_LF.setArg(2, depth_resolution);
		gWTA_LF.setArg(3, scale);
		commandQueue.enqueueNDRangeKernel(gWTA_LF,
			cl::NullRange,
			cl::NDRange(w_spatial, h_spatial),
			cl::NDRange(16, 16),
			NULL,
			&gWTA_LF_evt);
		gWTA_LF_evt.wait();
		elapsed[depth_resolution] = executionTime(gWTA_LF_evt);

		double sum_elapsed = 0;
		for (int it=0; it<=depth_resolution; it++)
			sum_elapsed += elapsed[it];
		printf("Elapsed Time: %.4lf ms\n", sum_elapsed);
		printf("Transfer buffer back from GPU.....\n");
		commandQueue.enqueueReadBuffer(bufferDepth, CL_TRUE, 0, (totalPixels_spatial) * sizeof(cl_uchar4), output.data);
		//commandQueue.enqueueReadBuffer(bufferRemap, CL_TRUE, 0, (totalPixels) * sizeof(cl_uchar4), bitmapResult);
		//LOGI("%d %d %d %d", bitmapLF[0], bitmapLF[1], bitmapLF[2], bitmapLF[3]);
	}
	catch (cl::Error err) {
		printf("Error(%d): %s\n", err.err(), err.what());
	}
}

int main(int argc, char **argv)
{
	// Init opencl variable
	ocl_init();

	std::string img_name = "LightField/infer_output";
	std::string img_type = "png";

	if (argc > 1) {
		img_name = argv[1];
	}

	std::string depth_name = img_name + "_depth";

	// Read img
	Mat img = imread(img_name + "." + img_type);
#if VISUALIZE
	imshow("orgin", img);
#endif
	cvtColor(img, img, CV_BGR2RGBA); // 3 channels -> 4 channels : + @

	Mat output;
	processLightField(img, output);
#if VISUALIZE
	imshow("output", output);
	waitKey(0);
#else
	imwrite(depth_name + "." + img_type, output);
#endif

	return 0;
}

double executionTime(cl::Event event)
{
	cl_ulong start, end;

	start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();

	return (double)1.0e-6 * (double)(end - start); // convert nanoseconds to milli-seconds on return
}