#ifndef MLP_H
#define MLP_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <ctime>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define WIN
//#define UNIX

#ifdef WIN
#include<Windows.h>
#endif
#ifdef UNIX
#include <sys/time.h>
#endif

#define THREAD_PER_GROUP 32
#define TESTING_MODE 1
#define TESTING_PLAT 1
//0: intel
//1:nvidia
//2: amd
#define TESTING_DEV 0

//some controling knots Macro here
//#define DEBUG

//interactively select the platform and device
//#define INTERACTIVE_MODE

//if VISUAL is defined, the input data will be draft drawed
#define VISUAL

class mlp
{
public:
	//DS on host for net
	float *h_net;
	int *h_layer_dim;
	cl_uint max_dim;
	int total_item; //total item in the matrix
	int layer_num;

	float *h_input, *h_output;	
	
	mlp(const char* netfile);
	~mlp();
	int getInput();
	int run_device();
	int run_cpu();
	int predict();
	int retrieve_result(float* container, int lang);

	float getKernelTime();
	float getLoadTime();
	float getCPUTime();

private:
	int load_num, cpu_load_num, device_load_num;	
	float kernel_time, CPUtime, load_time;

	//the intermediate result handler for host and device(double buffer)
	float* h_inter_res[2];
	cl_mem d_inter_res[2];

	int load(const char* net_file);
	int load_cpu();
	int load_device();

	float non_linear(float x);	

	//cl related vars and methods
	cl_program load_program(cl_context context, const char* filename);
	cl_platform_id *platforms;
	cl_device_id *devices;
	cl_uint platnum, devnum;
	cl_context context;
	cl_command_queue queue;
	cl_mem *d_net;
	cl_program program;
	cl_kernel vmm;
	cl_event timer;
	cl_ulong start, end, elipse;
};

#endif