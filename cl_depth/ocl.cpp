#include "ocl.h"

char					platformChooser;
vector<cl::Platform>	platforms;
cl::Context				context;
cl::CommandQueue		commandQueue;
cl::Program				program;

cl::Kernel				gRemap, gRemap_Lytro, gRemap_Image, gCAE, gCAE_Native, gCAE_Initial, gCAE_Bin, gWTA_LF, gSSD, gRefocus;

vector<cl::Device>		devices;
cl_int					errNum = CL_SUCCESS;


void ocl_init()
{
	try {
		// query for platform
		cl::Platform::get(&platforms);
		if (platforms.size() == 0) {
			printf("Platform size 0\n");
			return;
		}

		// Get the number of platform and information about the platform
		printf("Platform number is : %d\n", (int)platforms.size());
		std::string platformVendor;

		for (unsigned int i = 0; i < platforms.size(); ++i) {
			platforms[i].getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platformVendor);
			printf("Platform is by : %s\n", platformVendor.c_str());
			if (::toupper(platformVendor.at(0)) == 'A' || ::toupper(platformVendor.at(0)) == 'N') {
				platformChooser = i; break;
			}
			else if (::toupper(platformVendor.at(0)) == 'I') { platformChooser = 1; }
		}

		// After choose platform save it property(generally choose 0)
		cl_context_properties properties[] =
		{
			CL_CONTEXT_PLATFORM,
			(cl_context_properties)(platforms[platformChooser])(),		// platforms[1] for laptop, platform [0] for desktop in my case...
			0
		};

		// Create context
		context = cl::Context(CL_DEVICE_TYPE_GPU, properties);

		// Get device info
		devices = context.getInfo<CL_CONTEXT_DEVICES>();
		printf("Device number is : %d\n", (int)devices.size());
		for (unsigned int i = 0; i < devices.size(); ++i) {
			printf("Device #%d : %s\n", i, devices[i].getInfo<CL_DEVICE_NAME>().c_str());
		}

		cl_int err = CL_SUCCESS;

		// Generate first command queue for device1
		printf("making command queue for device[0]\n");
		commandQueue = cl::CommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &err); // 3rd parameter!

		std::ifstream sourceFile("constrained_adaptive_entropy.cl");
		std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
		cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));
		cl::Program program = cl::Program(context, source);
		program.build(devices, "-cl-fast-relaxed-math");
		while (program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) != CL_BUILD_SUCCESS) {
			printf("[build err at constrained_adaptive_entropy.cl]  %s\n", program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]).c_str());
			return;
		}

		std::ifstream sourceFile2("wta_lf.cl");
		std::string sourceCode2(std::istreambuf_iterator<char>(sourceFile2), (std::istreambuf_iterator<char>()));
		cl::Program::Sources source2(1, std::make_pair(sourceCode2.c_str(), sourceCode2.length() + 1));
		cl::Program program2 = cl::Program(context, source2);
		program2.build(devices, "-cl-fast-relaxed-math");
		while (program2.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) != CL_BUILD_SUCCESS) {
			printf("[build err at wta_lf.cl]  %s\n", program2.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]).c_str());
			return;
		}

		// create Kernel
		gRemap_Image = cl::Kernel(program, "LF_Remap_Image", &err);
		gRemap = cl::Kernel(program, "LF_Remap", &err);
		gRemap_Lytro = cl::Kernel(program, "LF_Remap_Lytro", &err);
		gCAE = cl::Kernel(program, "LF_CAE", &err);
		gCAE_Native = cl::Kernel(program, "LF_CAE_Naive", &err);
		gCAE_Initial = cl::Kernel(program, "LF_CAE_Initial", &err);
		gCAE_Bin = cl::Kernel(program, "LF_CAE_Bin", &err);
		gSSD = cl::Kernel(program, "LF_SSD", &err);
		gRefocus = cl::Kernel(program, "LF_Refocus", &err);

		gWTA_LF = cl::Kernel(program2, "winnerTakesAll_LF", &err);

		printf("Build Kernel completed\n");
	}
	catch (cl::Error err) {
		printf("Error(%d): %s\n", err.err(), err.what());
	}
}
