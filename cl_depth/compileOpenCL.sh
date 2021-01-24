#!/bin/bash
echo "Enter the source files you want to compile."
read SOURCE_FILE
echo "Enter the executable file name."
read OUTPUT_FILE
g++ $SOURCE_FILE -std=c++11 $(pkg-config --libs --cflags opencv) -larm_compute -larm_compute_core -lOpenCL -o $OUTPUT_FILE -DARM_COMPUTE_CL
