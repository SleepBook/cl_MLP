#include "mlp.h"

cl_program mlp::load_program(cl_context context, const char* filename)
{
	std::ifstream in(filename, std::ios_base::binary);
	if (!in.good()) {
		return 0;
	}

	// get file length
	in.seekg(0, std::ios_base::end);
	size_t length = in.tellg();
	in.seekg(0, std::ios_base::beg);

	// read program source
	std::vector<char> data(length + 1);
	in.read(&data[0], length);
	data[length] = 0;

	// create and build program 
	const char* source = &data[0];
	cl_program program = clCreateProgramWithSource(context, 1, &source, 0, 0);
	if (program == 0) {
		std::cout << "create unsucc" << std::endl;
		return 0;
	}

	if (clBuildProgram(program, 0, 0, 0, 0, 0) != CL_SUCCESS) {
		std::cout << "build unsucc" << std::endl;
		return 0;
	}
	return program;
}

float mlp::non_linear(float x)
{
	return 1.0 / (1.0 + exp(0.0 - x));
}

mlp::mlp(const char* netfile)
{
	load_num = 0;
	cpu_load_num = 0;
	device_load_num = 0;

	platforms = NULL;
	devices = NULL;
	platnum = 0;
	devnum = 0;
	context = 0;
	queue = 0;
	d_net = NULL;
	program = 0;
	vmm = 0;
	start = 0;
	end = 0;
	elipse = 0;
	kernel_time = 0;
	load_time = 0;
	
	max_dim = 0;
	total_item = 0; 
	layer_num = 0;
	h_layer_dim = NULL;
	h_net = NULL;

	load(netfile);
}

int mlp::load(const char* net_file)
{
	load_num++;
	load_time = 0;
	if(load_num == 1)
	{
		clock_t start, stop;
		start = clock();
		cl_int err;
		cl_uint num,numm;
		int tmp;
		char *name;
		size_t size;
		FILE *net_f;
		
		int next_dim, curr_dim;
		int i;
		int offset = 0;

		/////////////////////////////////////////////
		//construct the DS on host to store the net
		net_f = fopen(net_file, "r");
		fscanf(net_f, "%d", &layer_num);
		h_layer_dim = (int*)malloc(sizeof(int)*layer_num);
		fscanf(net_f, "%d", &curr_dim);
		max_dim = curr_dim;
		h_layer_dim[0] = curr_dim;
		for (i = 0; i < (layer_num - 1); i++)
		{
			fscanf(net_f, "%d", &next_dim);
			if (next_dim > max_dim)
			{
				max_dim = next_dim;
			}
			h_layer_dim[i + 1] = next_dim;
			total_item += (curr_dim + 1)*next_dim;
			curr_dim = next_dim;
		}
		h_net = (float*)malloc(sizeof(float)*total_item);
		for (i = 0; i<total_item; i++)
		{
			fscanf(net_f, "%f", &h_net[i]);
		}
		fclose(net_f);
#ifdef DEBUG
		printf("check internal DS for net\n");
		printf("layernum %d\n", layer_num);
		printf("layer dim %d %d %d %d\n", h_layer_dim[0], h_layer_dim[1], h_layer_dim[2], h_layer_dim[3]);
		printf("total net item %d\n", total_item);
#endif

		/////////////////////////////////////////////
		//allocate the space to store input & output data
		h_input = (float*)malloc(sizeof(float)*(h_layer_dim[0]+1)); // the reason for +1 is the bias item
		h_output = (float*)malloc(sizeof(float)*h_layer_dim[layer_num - 1]);

		/////////////////////////////////////////////
		//prepare CL environment
		err = clGetPlatformIDs(0,0,&num);
		if (err != CL_SUCCESS) {
			std::cerr << "Unable to get platforms\n";
		}
		platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id)*num);
		err = clGetPlatformIDs(num, platforms, NULL);
		if (err != CL_SUCCESS) {
			std::cerr << "Unable to get platforms\n";
		}
		for (i = 0; i < num; i++)
		{
			err = clGetPlatformInfo(
				platforms[i],
				CL_PLATFORM_NAME,
				0,
				0,
				&size);
			name = (char*)malloc(size);
			err = clGetPlatformInfo(
				platforms[i],
				CL_PLATFORM_NAME,
				size,
				name,
				NULL);
			printf("plat %d: %s\n", i, name);
			free(name);
		}
		platnum = TESTING_PLAT;

#ifdef INTERACTIVE_MODE
		std::cout << "please spicify the platform to use" << std::endl;
		std::cin >> platnum;
#endif
		err = clGetDeviceIDs(platforms[platnum], CL_DEVICE_TYPE_ALL, 0, 0, &num);
		devices = (cl_device_id*)malloc(sizeof(cl_device_id)*num);
		err = clGetDeviceIDs(platforms[platnum], CL_DEVICE_TYPE_ALL, num, devices, 0);
		printf("there are %u devices under this platform:\n", num);
		for (i = 0; i < num; i++)
		{
			err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 0, NULL, &size);
			name = (char*)malloc(size);
			err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, size, name, 0);
			err = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &numm, &size);
			printf("device %u: %s\n\tMAX compute units: %u\n", i, name, numm);
			free(name);	
		}
		devnum = TESTING_DEV;
#ifdef INTERACTIVE_MODE
		std::cout << "please spicify the device to use" << std::endl;
		std::cin >> devnum;
#endif
		cl_context_properties prop[] = { CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platforms[platnum]), 0 };
		context = clCreateContext(prop, 1, &devices[devnum], NULL, NULL, &err);
		if (context == 0) {
			std::cerr << "Can't create OpenCL context\n";
			return -1;
		}
		queue = clCreateCommandQueue(context, devices[devnum], CL_QUEUE_PROFILING_ENABLE, 0);
		if (queue == 0) {
			std::cerr << "Can't create command queue\n";
			clReleaseContext(context);
			return -1;
		}
		
		program = load_program(context, "shader.cl");

		if (program == 0) {
			std::cerr << "Can't load or build program\n";
		}

		vmm = clCreateKernel(program, "vmm", 0);
		if (vmm == 0) {
			std::cerr << "Can't load kernel\n";
		}

		///////////////////////////////////////////////////
		//construct the DS for net on device memory
		
		d_net = (cl_mem*)malloc(sizeof(cl_mem)*(layer_num-1));
		for (i = 0; i < layer_num-1; i++)
		{
			d_net[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*(h_layer_dim[i]+1)*h_layer_dim[i+1],NULL,NULL);
		}
		for(i=0;i<layer_num-1;i++)
		{
			tmp = (h_layer_dim[i]+1)*h_layer_dim[i+1];
			err = clEnqueueWriteBuffer(queue, d_net[i], CL_TRUE, 0, sizeof(float)*tmp, &h_net[offset],0,NULL,NULL);
			offset += tmp;
		}
		clFinish(queue);
		stop = clock();
		load_time = (float)(stop - start)/CLOCKS_PER_SEC;
	}
}

int mlp::getInput()
{
	FILE* f;
	int input_dim;
	int i, j;
	f = fopen("test.dat", "r");
	fscanf(f, "%d", &input_dim);
	fscanf(f, "%d", &input_dim);
	if (input_dim != h_layer_dim[0])
	{
		printf("error, input data mismatch with the net, exiting now\n");
		return -1;
	}
	for (i = 0; i < input_dim; i++)
	{
		fscanf(f, "%f", &h_input[i]);
	}
	h_input[input_dim] = 1.0;
	fclose(f);
#ifdef VISUAL
	for (i = 6; i < 28; i++)
	{
		for (j = 0; j < 28; j++)
		{
			if (h_input[j * 28 + i]>0.1)
			{
				printf("*");
			}
			else
			{
				printf(" ");
			}
		}
		printf("\n");
	}
#endif
	return 0;
}

int mlp::run_cpu()
{
	if (cpu_load_num == 0)
	{
		h_inter_res[0] = (float*)malloc(sizeof(float)*(max_dim + 1));
		h_inter_res[1] = (float*)malloc(sizeof(float)*(max_dim + 1));
		cpu_load_num++;
	}
#ifdef WIN
	LARGE_INTEGER large_int;
	double diff;
	__int64 c1, c2;

	QueryPerformanceFrequency(&large_int);
	diff = large_int.QuadPart;
	QueryPerformanceCounter(&large_int);
	c1 = large_int.QuadPart;
#endif
#ifdef UNIX
	long c1, c2;
	struct timeval tv;
	gettimeofday(&tv, NULL);
	c1 = tv.tv_usec;
#endif	
	float tmp_res;
	int offset,tmp;
	int i,j,k;
	int curr_dim, next_dim;
	int data_pool, result_pool;
	
	for(i=0;i<h_layer_dim[0]+1;i++)
	{
		h_inter_res[0][i] = h_input[i];	
	}

	offset = 0;
	for(i=0;i<layer_num-1;i++)
	{
		curr_dim = h_layer_dim[i] + 1;
		next_dim = h_layer_dim[i+1];
		
		//this is a very smart design, avoid branch to decide which buffer to use
		data_pool = i%2;
		result_pool = (i+1)%2;
		for(j = 0;j<next_dim;j++)
		{
			tmp_res = 0.0;
			for(k = 0;k<curr_dim;k++)
			{
				tmp_res += h_inter_res[data_pool][k]*h_net[offset + next_dim*k + j];
			}
			h_inter_res[result_pool][j] = non_linear(tmp_res);				
		}
		h_inter_res[result_pool][next_dim] = 1.0;
		offset += curr_dim*next_dim;
	}

	if(layer_num%2 == 0)
	{
		for (i = 0; i < h_layer_dim[layer_num - 1]; i++)
		{
			h_output[i] = h_inter_res[1][i];
		}
	}
	else
	{
		for (i = 0; i < h_layer_dim[layer_num - 1]; i++)
		{
			h_output[i] = h_inter_res[0][i];
		}
	}
#ifdef DEBUG
	for (i = 0; i < h_layer_dim[layer_num - 1]; i++)
	{
		printf("%f ", h_output[i]);
		printf("\n");
	}
#endif
#ifdef WIN
	QueryPerformanceCounter(&large_int);
	c2 = large_int.QuadPart;
	CPUtime = (float)((c2 - c1)*1e06 / diff);
#endif
#ifdef UNIX
	gettimeofday(&tv, NULL);
	c2 = tv.tv_usec;
	CPUtime = float(c2 - c1);
#endif
	return 0;
}

int mlp::run_device()
{
	if (device_load_num == 0)
	{
		d_inter_res[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*(max_dim + 1), NULL, NULL);
		d_inter_res[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*(max_dim + 1), NULL, NULL);
		device_load_num++;
	}
	int curr_dim, next_dim;
	int data_pool, result_pool;

	cl_int err;
	int i, j;

	size_t global_work_size;
	size_t local_work_size = THREAD_PER_GROUP;
	size_t size;

	err = clEnqueueWriteBuffer(queue, d_inter_res[0], CL_TRUE, 0, sizeof(float)*(h_layer_dim[0]+1), h_input, 0, NULL, NULL);

	elipse = 0;
	for(i=0;i<layer_num-1;i++)
	{
		curr_dim = h_layer_dim[i];
		next_dim = h_layer_dim[i+1];
		data_pool = i % 2;
		result_pool = (i + 1) % 2;

		clSetKernelArg(vmm, 0, sizeof(cl_int), &curr_dim);
		clSetKernelArg(vmm, 1, sizeof(cl_int), &next_dim);
		clSetKernelArg(vmm, 2, sizeof(cl_mem), &d_net[i]);
		clSetKernelArg(vmm, 3, sizeof(cl_mem), &d_inter_res[data_pool]);
		clSetKernelArg(vmm, 4, sizeof(cl_mem), &d_inter_res[result_pool]);

		global_work_size = next_dim;
		err = clEnqueueNDRangeKernel(queue, vmm, 1, NULL, &global_work_size, NULL, 0, NULL, &timer);
		if (err != CL_SUCCESS)
		{
			printf("error enqueue %d\n",err);
		}
		clFinish(queue);
		err = clGetEventProfilingInfo(timer, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, &size);
		if (err != CL_SUCCESS)
		{
			printf("error prefile, error code is %d\n", err);
		}
		clGetEventProfilingInfo(timer, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, &size);
		if (err != CL_SUCCESS)
		{
			printf("error prefile, error code is %d\n", err);
		}
		elipse += (cl_ulong)end - start;
#ifdef DEBUG
		printf("\n\nthis is the tmp res from layre%d\n", i);
		float* tmpres = (float*)malloc(sizeof(float)*(next_dim + 1));
		err = clEnqueueReadBuffer(queue, d_inter_res[result_pool], CL_TRUE, 0, sizeof(float)*(next_dim + 1), tmpres, 0, NULL, NULL);
		for (j = 0; j < next_dim + 1; j++)
		{
			printf("%f ", tmpres[j]);
		}
		printf("\n");
		free(tmpres);
#endif
	}

	kernel_time = (float)(elipse*1e-03);
	if (layer_num % 2 == 0)
	{
		err = clEnqueueReadBuffer(queue, d_inter_res[1], CL_TRUE, 0, sizeof(float)*next_dim, h_output, 0, 0, 0);
	}
	else
	{
		err = clEnqueueReadBuffer(queue, d_inter_res[0], CL_TRUE, 0, sizeof(float)*next_dim, h_output, 0, 0, 0);
	}
	return 0;
}

int mlp::predict()
{
	float tmp_base = 0.0;
	int prediction;
	int i;
	for (i = 0; i < h_layer_dim[layer_num-1]; i++)
	{
		if (h_output[i] > tmp_base)
		{
			prediction = i;
			tmp_base = h_output[i];
		}
	}
	prediction = (prediction + 1) % 10;
	printf("\nthe prediction is: %d\n", prediction);
	return prediction;
}

int mlp::retrieve_result(float* container, int lang)
{
	int i;
	if (lang >  h_layer_dim[layer_num-1])
	{
		printf("retrieve lenght excede the output dimension\n");
		for (i = 0; i<h_layer_dim[layer_num-1]; i++)
		{
			container[i] = h_output[i];
		}
	}
	else
	{
		for (i = 0; i<lang; i++)
		{
			container[i] = h_output[i];
		}
	}
	return 0;
}

float mlp::getKernelTime() {
	return kernel_time;
}

float mlp::getLoadTime() {
	return load_time;
}

float mlp::getCPUTime()
{
	return CPUtime;
}

mlp::~mlp()
{
	int i;
	clReleaseKernel(vmm);
	clReleaseProgram(program);
	for(i=0;i<layer_num-1;i++)
	{
		clReleaseMemObject(d_net[i]);
	}
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	free(h_net);
	free(h_layer_dim);
	free(d_net);
	free(devices);
	free(platforms);
	free(h_input);
	free(h_output);
	if (cpu_load_num != 0)
	{
		free(h_inter_res[0]);
		free(h_inter_res[1]);
	}
	if (device_load_num != 0)
	{
		clReleaseMemObject(d_inter_res[0]);
		clReleaseMemObject(d_inter_res[1]);
	}
	
}
