#include "mlp.h"
using namespace std;

int main(){
	char input;
	float *result;
	int i, j;
	int input_dim;
	int prediction;
	float ker_time, load_time;
	result = (float*)malloc(sizeof(float) * 10);

	mlp testInstance("mnist_mlp.net");

	for (i = 0; i < 1; i++)
	{
		cout << "mlp calling : " << i << endl;
		testInstance.getInput();
		testInstance.run_device();
		//j = testInstance.retrieve_result(result, 10);
		//for (j = 0; j < 10; j++)
		//{
		//	printf("%f ", result[j]);
		//}
		//prediction = testInstance.predict();
		//ker_time = testInstance.getCPUTime();
		//printf("the CPU time is %fus\n", ker_time);
		prediction = testInstance.predict();
		ker_time = testInstance.getKernelTime();
		printf("the device time is %fus\n", ker_time);
	}
	getchar();
	return 0;
}