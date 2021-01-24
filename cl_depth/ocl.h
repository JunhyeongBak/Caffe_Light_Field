#define __CL_ENABLE_EXCEPTIONS

// OpenCL Headers
#include "CL/cl.hpp"
#include <iostream>
#include <algorithm>
#include <fstream>
#include <vector>

using std::vector;

extern char					platformChooser;
extern vector<cl::Platform> platforms;
extern cl::Context			context;
extern cl::CommandQueue		commandQueue;
extern cl::Program			program;

extern cl::Kernel			gRemap, gRemap_Lytro, gRemap_Image, gCAE, gCAE_Native, gCAE_Initial, gCAE_Bin, gWTA_LF, gSSD, gRefocus;
extern cl_int				errNum;

void ocl_init();

