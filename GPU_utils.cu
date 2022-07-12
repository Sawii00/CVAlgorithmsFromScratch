#include "GPU_utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <string>

#define PI 3.14159

struct Gradient{
	float module;
	uint8_t direction;
};

struct Index_Cuple{
	int index1;
	int index2;
	Index_Cuple(int i1 = 0, int i2 = 0): index1(i1), index2(i2){};
};

class TimedBlock
	{
	private:

		std::string name;
		std::chrono::time_point<std::chrono::system_clock> m_StartTime;
		std::chrono::time_point<std::chrono::system_clock> m_EndTime;
		uint64_t time;

		void displayTimedBlock()
		{
			if (this->time > 100000)
			{
				std::string out = "Name: " + this->name + " ExecutionTime: " + std::to_string(this->time / 1000) + "ms";
				std::cout << out << std::endl;
			}
			else
			{
				std::string out = "Name: " + this->name + " ExecutionTime: " + std::to_string(this->time) + "us";
				std::cout << out << std::endl;
			}
		}

	public:

		TimedBlock(std::string&& name)
		{
			this->name = name;
			this->time = 0;
			this->m_StartTime = std::chrono::system_clock::now();
		}

		inline void stopTimedBlock()
		{
			this->m_EndTime = std::chrono::system_clock::now();
			this->time = std::chrono::duration_cast<std::chrono::microseconds>(m_EndTime - m_StartTime).count();
			displayTimedBlock();
		}
	};



__global__
void gpu_convolve_core(float* filter, uint8_t* buffer, uint8_t* im, int filter_width, int filter_height, int im_width, int im_height) {
	int stride = int(filter_height / 2);
	int pixel_n = blockDim.x * blockIdx.x + threadIdx.x;
	int x = pixel_n % im_width;
	int y = pixel_n / im_width;
	if (pixel_n < im_height * im_width) {
		float r = 0.0;
		float g = 0.0;
		float b = 0.0;
		int char_index = 0;
		for (int l = 0; l < filter_height; l++) { //y of the filter
			for (int k = 0; k < filter_width; k++) { // x of the filter
				char_index = cudaMirrorGet((x + k - stride), (y + l - stride), im_width, im_height);
				char_index *= 4;
				r += (float)(im[char_index]) * filter[l * filter_width + k]; //Buffer[x,y].R
				g += (float)(im[char_index + 1]) * filter[l * filter_width + k]; //Buffer[x,y].G
				b += (float)(im[char_index + 2]) * filter[l * filter_width + k]; //Buffer[x,y].B
			}
		}
		buffer[4 * (im_width * y + x)] = (uint8_t)(max_(0, min_(255, r)));
		buffer[4 * (im_width * y + x) + 1] = (uint8_t)(max_(0, min_(255, g)));
		buffer[4 * (im_width * y + x) + 2] = (uint8_t)(max_(0, min_(255, b)));
		buffer[4 * (im_width * y + x) + 3] = im[4 * (im_width * y + x) + 3];
	}
}

__global__
void gpu_convolve_gray_core(float* filter, uint8_t* buffer, uint8_t* im, int filter_width, int filter_height, int im_width, int im_height) {
	int stride = int(filter_height / 2);
	int pixel_n = blockDim.x * blockIdx.x + threadIdx.x;
	int x = pixel_n % im_width;
	int y = pixel_n / im_width;
	if (pixel_n < im_height * im_width) {
		float val = 0.0;
		int char_index = 0;
		for (int l = 0; l < filter_height; l++) { //y of the filter
			for (int k = 0; k < filter_width; k++) { // x of the filter
				char_index = cudaMirrorGet((x + k - stride), (y + l - stride), im_width, im_height);
				val += (float)(im[char_index]) * filter[l * filter_width + k]; //Buffer[x,y]
			}
		}
		buffer[(im_width * y + x)] = (uint8_t)(max_(0, min_(255, val)));
	}
}

__global__
void gpu_convolve_core_rgb_to_mono(float* filter, float* result, uint8_t* im, int filter_width, int filter_height, int im_width, int im_height) {
	int stride = int(filter_height / 2);
	int pixel_n = blockDim.x * blockIdx.x + threadIdx.x;
	int x = pixel_n % im_width;
	int y = pixel_n / im_width;
	if (pixel_n < im_height * im_width) {
		float res = 0.0f;
		int char_index = 0;
		for (int l = 0; l < filter_height; l++) { //y of the filter
			for (int k = 0; k < filter_width; k++) { // x of the filter
				char_index = cudaMirrorGet((x + k - stride), (y + l - stride), im_width, im_height);
				char_index *= 4;
				res += (
				0.3*((float)(im[char_index]) * filter[l * filter_width + k]) +
				0.59*((float)(im[char_index + 1]) * filter[l * filter_width + k]) + 
				0.11*((float)(im[char_index + 2]) * filter[l * filter_width + k])
				);
			}
		}
		result[im_width * y + x] = res;
	}
}


__global__
void gpu_gradients_core(float* v_filter, float* h_filter, Gradient* result, uint8_t* im, int filter_width, int filter_height, int im_width, int im_height) {
	int stride = int(filter_height / 2);
	int pixel_n = blockDim.x * blockIdx.x + threadIdx.x;
	int x = pixel_n % im_width;
	int y = pixel_n / im_width;
	if (pixel_n < im_height * im_width) {
		float h_res = 0.0f;
		float v_res = 0.0f;
		int char_index = 0;
		for (int l = 0; l < filter_height; l++) { //y of the filter
			for (int k = 0; k < filter_width; k++) { // x of the filter
				char_index = cudaMirrorGet((x + k - stride), (y + l - stride), im_width, im_height);
				char_index *= 4;
				h_res += (
				0.3*((float)(im[char_index]) * h_filter[l * filter_width + k]) +
				0.59*((float)(im[char_index + 1]) * h_filter[l * filter_width + k]) + 
				0.11*((float)(im[char_index + 2]) * h_filter[l * filter_width + k])
				);
				v_res += (
				0.3*((float)(im[char_index]) * v_filter[l * filter_width + k]) +
				0.59*((float)(im[char_index + 1]) * v_filter[l * filter_width + k]) + 
				0.11*((float)(im[char_index + 2]) * v_filter[l * filter_width + k])
				);
			}
		}
		result[im_width * y + x].module = hypot(h_res, v_res);
		float angle = atan2(v_res, h_res);
		angle = angle >= 0 ? angle : angle + PI;
		angle = (angle * 180) / PI;
		uint8_t valid_angles [9] = {0,45,45,90,90,135,135,0,0};
		result[im_width * y + x].direction = valid_angles[(int)(angle/22.5)];
	}
}

__global__
void gpu_gradients_gray_core(float* v_filter, float* h_filter, Gradient* result, uint8_t* im, int filter_width, int filter_height, int im_width, int im_height) {
	int stride = int(filter_height / 2);
	int pixel_n = blockDim.x * blockIdx.x + threadIdx.x;
	int x = pixel_n % im_width;
	int y = pixel_n / im_width;
	if (pixel_n < im_height * im_width) {
		float h_res = 0.0f;
		float v_res = 0.0f;
		int char_index = 0;
		for (int l = 0; l < filter_height; l++) { //y of the filter
			for (int k = 0; k < filter_width; k++) { // x of the filter
				char_index = cudaMirrorGet((x + k - stride), (y + l - stride), im_width, im_height);
				h_res += ((float)(im[char_index]) * h_filter[l * filter_width + k]);
				v_res += ((float)(im[char_index]) * v_filter[l * filter_width + k]);
			}
		}
		result[im_width * y + x].module = hypot(h_res, v_res);
		float angle = atan2(v_res, h_res);
		angle = angle >= 0 ? angle : angle + PI;
		angle = (angle * 180) / PI;
		uint8_t valid_angles [9] = {0,45,45,90,90,135,135,0,0};
		result[im_width * y + x].direction = valid_angles[(int)(angle/22.5)];
	}
}

__global__
void gpu_max_suppress(Gradient* im, Gradient* result, int im_width, int im_height, float thold_low){
	int pixel_n = blockDim.x * blockIdx.x + threadIdx.x;
	int x = pixel_n % im_width;
	int y = pixel_n / im_width;
	if (pixel_n < im_height * im_width) {
		int index1=3112;
		int index2=3112;
		auto dir = im[pixel_n].direction;
		switch (dir){
			case 0:
				index1 = cudaMirrorGet(x+1,y,im_width, im_height);
				index2 = cudaMirrorGet(x-1,y,im_width, im_height);
				break;
			case 45:
				index1 = cudaMirrorGet(x+1,y+1,im_width, im_height);
				index2 = cudaMirrorGet(x-1,y-1,im_width, im_height);
				break;
			case 90:
				index1 = cudaMirrorGet(x,y+1,im_width, im_height);
				index2 = cudaMirrorGet(x,y-1,im_width, im_height);
				break;
			case 135:
				index1 = cudaMirrorGet(x+1,y-1,im_width, im_height);
				index2 = cudaMirrorGet(x-1,y+1,im_width, im_height);
				break;
			default:
				index1 = 0;
				index2 = 0;
				break;
		}
		result[pixel_n] = im[pixel_n];
		bool valid = (im[pixel_n].module > im[index1].module) && (im[pixel_n].module > im[index2].module) && (im[pixel_n].module > thold_low);
		if (!valid)
		result[pixel_n].module = 0;
	}
}

__global__
void gpu_edge_tracking(Gradient* im, Gradient* result, int im_width, int im_height, float thold_high, float thold_low){
	int pixel_n = blockDim.x * blockIdx.x + threadIdx.x;
	int x = pixel_n % im_width;
	int y = pixel_n / im_width;
	int index = 0;
	if (pixel_n < im_height * im_width) {
		result[pixel_n] = im[pixel_n];
		if (im[pixel_n].module > thold_low && im[pixel_n].module <= thold_high){
			bool keep = false;
			for (int i = -1; i < 2 && !keep; i++){
				for (int j = -1; j < 2 && !keep; j++){
					index = cudaMirrorGet(x+i,y+j,im_width, im_height);
					if ( im[index].module >= thold_high) {keep=true;}
				}
			}
			if (!keep)
			{result[pixel_n].module = 0;}
		}
	}
}


__global__
void gpu_gradient_to_rgb(Gradient* im, uint8_t* result, int im_width, int im_height){
	int pixel_n = blockDim.x * blockIdx.x + threadIdx.x;
	if (pixel_n < im_height * im_width) {
		result[4 * pixel_n] = (uint8_t)(max_(0, min_(255, im[pixel_n].module)));
		result[4 * pixel_n + 1] = (uint8_t)(max_(0, min_(255, im[pixel_n].module)));
		result[4 * pixel_n + 2] = (uint8_t)(max_(0, min_(255, im[pixel_n].module)));
		result[4 * pixel_n + 3] = 255;
	}
}

__global__
void gpu_gradient_to_bin(Gradient* im, uint8_t* result, int im_width, int im_height){
	int pixel_n = blockDim.x * blockIdx.x + threadIdx.x;
	if (pixel_n < im_height * im_width) {
		uint8_t val = im[pixel_n].module != 0 ? 255 : 0;
		result[4 * pixel_n] = val;
		result[4 * pixel_n + 1] = val;
		result[4 * pixel_n + 2] = val;
		result[4 * pixel_n + 3] = 255;
	}
}

__global__
void gpu_gradient_to_gray_bin(Gradient* im, uint8_t* result, int im_width, int im_height){
	int pixel_n = blockDim.x * blockIdx.x + threadIdx.x;
	if (pixel_n < im_height * im_width) {
		uint8_t val = im[pixel_n].module != 0 ? 255 : 0;
		result[pixel_n] = val;
	}
}

__global__
void gpu_RGBtoHSL_core(uint8_t* rgb, uint8_t* hsl, int w, int h){
	int pixel_n = blockDim.x * blockIdx.x + threadIdx.x;
	int hsl_size = sizeof(int) + 2 * sizeof(float);
	if (pixel_n < h * w) {
		float r = ((float)rgb[4*pixel_n]) / 255.0f;
		float g = ((float)rgb[4*pixel_n+1]) / 255.0f;
		float b = ((float)rgb[4*pixel_n+2]) / 255.0f;
		float max_val = max3_(r, g, b);
		float min_val = min3_(r, g, b);
		float c = max_val - min_val;
		*((float*)(hsl+pixel_n*hsl_size+sizeof(int)+sizeof(float))) = min_(1.0f, max_(0.0f, (max_val + min_val) / 2)) ; //hsl.l
		if (c == 0) {
			*((int*)(hsl+pixel_n*hsl_size)) = 0;
			*((float*)(hsl+pixel_n*hsl_size+sizeof(int))) = 0;
		}
		else {
			*((float*)(hsl+pixel_n*hsl_size+sizeof(int))) = min_(1.0f, max_(0.0f, c / (1 - abs(2 * (*((float*)(hsl+pixel_n*hsl_size+sizeof(int)+sizeof(float)))) - 1)))) ; //hsl.s
			if (max_val == r) {
				float segment = (g - b) / c;
				*((int*)(hsl+pixel_n*hsl_size)) = (segment + ((segment >= 0) ? 0.0f : 6.0f)) * 60.0f; //hsl.h
			}
			else if (max_val == g) {
				*((int*)(hsl+pixel_n*hsl_size)) = (((b - r) / c) + (2.0f)) * 60.0f;
			}
			else {
				*((int*)(hsl+pixel_n*hsl_size)) = (((r - g) / c) + (4.0f)) * 60.0f;
			}
		}
	}
}

__global__
void gpu_HSLtoRGB_core(uint8_t* rgb, uint8_t* hsl, int w, int h){
	int pixel_n = blockDim.x * blockIdx.x + threadIdx.x;
	int hsl_size = sizeof(int) + 2 * sizeof(float);
	if (pixel_n < h * w) {
		float c = (1 - abs(2 * *((float*)(hsl+pixel_n*hsl_size+sizeof(int)+sizeof(float))) - 1)) * *((float*)(hsl+pixel_n*hsl_size+sizeof(int)));
		float x = c * (1 - abs(fmodf(float(*((int*)(hsl+pixel_n*hsl_size))) / 60.0f, 2.0f) - 1.0f));
		float m_ = *((float*)(hsl+pixel_n*hsl_size+sizeof(int)+sizeof(float))) - c / 2;
		uint8_t params[6][4] = {
			{(c + m_) * 255.0f, (x + m_) * 255.0f, m_ * 255.0f, 255},
			{(x + m_) * 255.0f, (c + m_) * 255.0f, m_ * 255.0f, 255},
			{m_ * 255.0f, (c + m_) * 255.0f, (x + m_) * 255.0f, 255},
			{m_ * 255.0f, (x + m_) * 255.0f, (c + m_) * 255.0f, 255},
			{(x + m_) * 255.0f, m_ * 255.0f, (c + m_) * 255.0f, 255},
			{(c + m_) * 255.0f, m_ * 255.0f, (x + m_) * 255.0f, 255}
		};
		*((int*)(rgb+pixel_n*4)) = *((int*)(params[(int((*((int*)(hsl+pixel_n*hsl_size))) / 60))]));
	}
}

__global__
void gpu_RGBtoGRAY_core(uint8_t* rgb, uint8_t* gray, int w, int h){
	int pixel_n = blockDim.x * blockIdx.x + threadIdx.x;
	if (pixel_n < h * w) {
		float r = ((float)rgb[4*pixel_n]);
		float g = ((float)rgb[4*pixel_n+1]);
		float b = ((float)rgb[4*pixel_n+2]);
		gray[pixel_n] = round((float)(0.3f * r) + (float)(0.59f * g) + (float)(0.11f * b));
	}
}

__global__
void gpu_HSLtoGRAY_core(uint8_t* gray, uint8_t* hsl, int w, int h){
	int pixel_n = blockDim.x * blockIdx.x + threadIdx.x;
	int hsl_size = sizeof(int) + 2 * sizeof(float);
	if (pixel_n < h * w) {
		gray[pixel_n] = round((*((float*)(hsl+pixel_n*hsl_size+sizeof(int)+sizeof(float)))) * 255.0f);
	}
}
	

int GPU_utils::gpuConvolve(float* filter, uint8_t** im, int filter_width, int filter_height, int im_width, int im_height, int passages){
	cudaError_t err = cudaSuccess;
	int img_size = im_width * im_height * 4;
	int filter_size = filter_width * filter_height * sizeof(float);
	int threadsPerBlock = 256;
	int blocksPerGrid = (im_width * im_height + threadsPerBlock - 1) / threadsPerBlock;
	uint8_t* buffer = new uint8_t[img_size];

	uint8_t* gpu_IMG = nullptr;
	err = cudaMalloc(&gpu_IMG, img_size);
	if (err != cudaSuccess){
		std::cout<<"Error: "<<err<<"\n";
		return 0;
	}

	uint8_t* gpu_BUFFER = nullptr;
	cudaMalloc(&gpu_BUFFER, img_size);

	float* gpu_FILTER = nullptr;
	cudaMalloc(&gpu_FILTER, filter_size);

	err = cudaMemcpy(gpu_IMG, *im, img_size, cudaMemcpyHostToDevice);
	
	if (err != cudaSuccess){
		std::cout<<"Error: "<<err<<"\n";
		return 0;
	}

	cudaMemcpy(gpu_FILTER, filter, filter_size, cudaMemcpyHostToDevice);
	
	for (size_t i = 0; i < passages; i++){
		
		if (!(i%2)){
			gpu_convolve_core << <blocksPerGrid, threadsPerBlock >> > (gpu_FILTER, gpu_BUFFER, gpu_IMG, filter_width, filter_height, im_width, im_height);
			
			if (!i){
				err = cudaGetLastError();
				
				if (err != cudaSuccess){
					std::cout<<"Error: "<<err<<"\n";
					return 0;
				}
			}
		}
		else{
			gpu_convolve_core << <blocksPerGrid, threadsPerBlock >> > (gpu_FILTER, gpu_IMG, gpu_BUFFER, filter_width, filter_height, im_width, im_height);
		}

		

		cudaDeviceSynchronize();
	
	}

	if (passages%2)
		err = cudaMemcpy(buffer, gpu_BUFFER, img_size, cudaMemcpyDeviceToHost);
	else
		err = cudaMemcpy(buffer, gpu_IMG, img_size, cudaMemcpyDeviceToHost);
	
	if (err != cudaSuccess){
		std::cout<<"Error: "<<err<<"\n";
		return 0;
	}

	cudaFree(gpu_IMG);
	cudaFree(gpu_BUFFER);
	cudaFree(gpu_FILTER);

	uint8_t* temp = *im;
	*im = buffer;
	delete[] temp;
	return 1;
}

int GPU_utils::gpuRGBtoHSLImage(uint8_t** rgb_img, uint8_t** hsl_img, int img_w, int img_h){
	cudaError_t err = cudaSuccess;
	int threadsPerBlock = 256;
	int blocksPerGrid = (img_w * img_h + threadsPerBlock - 1) / threadsPerBlock;
	int rgb_size = img_w * img_h * 4;
	int hsl_size = img_w * img_h * (sizeof(int) + 2*sizeof(float));
	
	uint8_t* gpu_RGB = nullptr;
	err = cudaMalloc(&gpu_RGB, rgb_size);
	if (err != cudaSuccess){
		std::cout<<"Error: "<<err<<"\n";
		return 0;
	}
	
	uint8_t* gpu_HSL = nullptr;
	cudaMalloc(&gpu_HSL, hsl_size);
	
	err = cudaMemcpy(gpu_RGB, *rgb_img, rgb_size, cudaMemcpyHostToDevice);
	
	if (err != cudaSuccess){
		std::cout<<"Error: "<<err<<"\n";
		return 0;
	}
	
	gpu_RGBtoHSL_core << <blocksPerGrid, threadsPerBlock >> > (gpu_RGB, gpu_HSL, img_w, img_h);
	
	err = cudaGetLastError();
	
	if (err != cudaSuccess){
		std::cout<<"Error: "<<err<<"\n";
		return 0;
	}

	cudaDeviceSynchronize();

	err = cudaMemcpy(*hsl_img, gpu_HSL, hsl_size, cudaMemcpyDeviceToHost);

	
	if (err != cudaSuccess){
		std::cout<<"Error: "<<err<<"\n";
		return 0;
	}

	cudaFree(gpu_RGB);
	cudaFree(gpu_HSL);
	return 1;
}

int GPU_utils::gpuHSLtoRGBImage(uint8_t** hsl_img, uint8_t** rgb_img, int img_w, int img_h){
	cudaError_t err = cudaSuccess;
	int threadsPerBlock = 256;
	int blocksPerGrid = (img_w * img_h + threadsPerBlock - 1) / threadsPerBlock;
	int rgb_size = img_w * img_h * 4;
	int hsl_size = img_w * img_h * (sizeof(int) + 2*sizeof(float));
	
	//std::cout<<"Width: "<<img_w<<"\nHeigth: "<<img_h<<"\n";
	
	
	uint8_t* gpu_RGB = nullptr;
	err = cudaMalloc(&gpu_RGB, rgb_size);
	if (err != cudaSuccess){
		std::cout<<"HSLtoRGB error alloc: "<<err<<"\n";
		return 0;
	}
	
	uint8_t* gpu_HSL = nullptr;
	cudaMalloc(&gpu_HSL, hsl_size);
	
	
	err = cudaMemcpy(gpu_HSL, *hsl_img, hsl_size, cudaMemcpyHostToDevice);
	
	if (err != cudaSuccess){
		std::cout<<"HSLtoRGB error copy: "<<err<<"\n";
		return 0;
	}
	
	
	gpu_HSLtoRGB_core << <blocksPerGrid, threadsPerBlock >> > (gpu_RGB, gpu_HSL, img_w, img_h);
	
	err = cudaGetLastError();
	
	if (err != cudaSuccess){
		std::cout<<"HSLtoRGB error core: "<<err<<"\n";
		return 0;
	}

	cudaDeviceSynchronize();
	

	
	err = cudaMemcpy(*rgb_img, gpu_RGB, rgb_size, cudaMemcpyDeviceToHost);
	
	if (err != cudaSuccess){
		std::cout<<"HSLtoRGB error copy back: "<<err<<"\n";
		return 0;
	}

	cudaFree(gpu_RGB);
	cudaFree(gpu_HSL);
	
	return 1;
}

int GPU_utils::gpuCannyEdge(uint8_t** rgb_img, int im_w, int im_h, float thold_high, float thold_low, bool rgb_out){
	
	cudaError_t err = cudaSuccess;
	int img_size = im_w * im_h;
	int byte_size = img_size * 4;
	int threadsPerBlock = 256;
	int blocksPerGrid = (im_w * im_h + threadsPerBlock - 1) / threadsPerBlock;
	
	Gradient* result = nullptr;
	uint8_t* gpu_IMG = nullptr;
	float h_filter [25] = {2,1,0,-1,-2,2,1,0,-1,-2,4,2,0,-2,-4,2,1,0,-1,-2,2,1,0,-1,-2};
	float v_filter [25] = {2,2,4,2,2,1,1,2,1,1,0,0,0,0,0,-1,-1,-2,-1,-1,-2,-2,-4,-2,-2};
	float gau_filter [25] = {0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625, 0.015625, 0.0625, 0.09375, 0.0625, 0.015625, 0.0234375, 0.09375, 0.140625, 0.09375, 0.0234375, 0.015625, 0.0625, 0.09375, 0.0625, 0.015625, 0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625};
	float* gpu_H_FILTER = nullptr;
	float* gpu_V_FILTER = nullptr;
	float* gpu_GAU_FILTER = nullptr;
	uint8_t* gpu_BUFFER = nullptr;
	
	err = cudaMalloc(&gpu_IMG, byte_size);
	if (err != cudaSuccess){
		std::cout<<"Allocation error: "<<err<<"\n";
		return 0;
	}
	
	cudaMalloc(&gpu_BUFFER, byte_size);
	cudaMalloc(&result, img_size*sizeof(Gradient));
	cudaMalloc(&gpu_H_FILTER, 25*sizeof(float));
	cudaMalloc(&gpu_V_FILTER, 25*sizeof(float));
	cudaMalloc(&gpu_GAU_FILTER, 25*sizeof(float));
	
	err = cudaMemcpy(gpu_IMG, *rgb_img, byte_size, cudaMemcpyHostToDevice);
	
	if (err != cudaSuccess){
		std::cout<<"Copy fw error: "<<err<<"\n";
		return 0;
	}
	
	cudaMemcpy(gpu_H_FILTER, h_filter, 25*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_V_FILTER, v_filter, 25*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_GAU_FILTER, gau_filter, 25*sizeof(float), cudaMemcpyHostToDevice);
	
	
	gpu_convolve_core << <blocksPerGrid, threadsPerBlock >> > (gpu_GAU_FILTER, gpu_BUFFER, gpu_IMG, 5, 5, im_w, im_h);
	
	
	cudaDeviceSynchronize();
	
	
	gpu_gradients_core << <blocksPerGrid, threadsPerBlock>> > (gpu_V_FILTER, gpu_H_FILTER, result, gpu_BUFFER, 5, 5, im_w, im_h);
	

	cudaDeviceSynchronize();
	
	
	err = cudaGetLastError();
				
	if (err != cudaSuccess){
		std::cout<<"Core exec gradient error: "<<err<<"\n";
		return 0;
	}
	

	
	
	
	Gradient* temp = nullptr;
	cudaMalloc(&temp, img_size*sizeof(Gradient));
	
	/*Index_Cuple* index_list = new Index_Cuple[img_size];
	int* gpu_IND = nullptr;
	cudaMalloc(&gpu_IND, sizeof(Index_Cuple)*img_size);*/
	
	gpu_max_suppress << <blocksPerGrid, threadsPerBlock>> > (result, temp, im_w, im_h, thold_low);
	
	//cudaMemcpy(index_list, gpu_IND, sizeof(Index_Cuple)*img_size, cudaMemcpyDeviceToHost);
	
	cudaDeviceSynchronize();
	
	err = cudaGetLastError();
				
	if (err != cudaSuccess){
		std::cout<<"Core exec max_suppress error: "<<err<<"\n";
		return 0;
	}
	
	
	gpu_edge_tracking << <blocksPerGrid, threadsPerBlock>> > (temp, result, im_w, im_h, thold_high, thold_low);
	
	cudaDeviceSynchronize();
	
	err = cudaGetLastError();
				
	if (err != cudaSuccess){
		std::cout<<"Core exec edge track error: "<<err<<"\n";
		return 0;
	}
	
	if (rgb_out){
		gpu_gradient_to_rgb << <blocksPerGrid, threadsPerBlock>> > (result, gpu_IMG, im_w, im_h);
	
		cudaDeviceSynchronize();
	}
	else{
		gpu_gradient_to_bin << <blocksPerGrid, threadsPerBlock>> > (result, gpu_IMG, im_w, im_h);
	
		cudaDeviceSynchronize();
	}
	
	
	err = cudaGetLastError();
				
	if (err != cudaSuccess){
		std::cout<<"Core exec grad to rgb error: "<<err<<"\n";
		return 0;
	}
	
	
	
	err = cudaMemcpy(*rgb_img, gpu_IMG, byte_size, cudaMemcpyDeviceToHost);	
	
	if (err != cudaSuccess){
		std::cout<<"Error copy back: "<<err<<"\n";
		return 0;
	}

	cudaFree(gpu_IMG);
	cudaFree(result);
	cudaFree(gpu_H_FILTER);
	cudaFree(gpu_V_FILTER);
	cudaFree(temp);
	
	return 1;
	
}

int GPU_utils::gpuRGBtoGRAYImage(uint8_t** rgb_img, uint8_t** gray_img, int img_w, int img_h){
	cudaError_t err = cudaSuccess;
	int threadsPerBlock = 256;
	int blocksPerGrid = (img_w * img_h + threadsPerBlock - 1) / threadsPerBlock;
	int rgb_size = img_w * img_h * 4;
	int gray_size = img_w * img_h;
	
	uint8_t* gpu_RGB = nullptr;
	err = cudaMalloc(&gpu_RGB, rgb_size);
	if (err != cudaSuccess){
		std::cout<<"Error: "<<err<<"\n";
		return 0;
	}
	
	uint8_t* gpu_GRAY = nullptr;
	cudaMalloc(&gpu_GRAY, gray_size);
	
	err = cudaMemcpy(gpu_RGB, *rgb_img, rgb_size, cudaMemcpyHostToDevice);
	
	if (err != cudaSuccess){
		std::cout<<"Error: "<<err<<"\n";
		return 0;
	}
	
	gpu_RGBtoGRAY_core << <blocksPerGrid, threadsPerBlock >> > (gpu_RGB, gpu_GRAY, img_w, img_h);
	
	err = cudaGetLastError();
	
	if (err != cudaSuccess){
		std::cout<<"Error: "<<err<<"\n";
		return 0;
	}

	cudaDeviceSynchronize();

	err = cudaMemcpy(*gray_img, gpu_GRAY, gray_size, cudaMemcpyDeviceToHost);

	
	if (err != cudaSuccess){
		std::cout<<"Error: "<<err<<"\n";
		return 0;
	}

	cudaFree(gpu_RGB);
	cudaFree(gpu_GRAY);
	return 1;
}
int GPU_utils::gpuHSLtoGRAYImage(uint8_t** hsl_img, uint8_t** gray_img, int img_w, int img_h){
	cudaError_t err = cudaSuccess;
	int threadsPerBlock = 256;
	int blocksPerGrid = (img_w * img_h + threadsPerBlock - 1) / threadsPerBlock;
	int gray_size = img_w * img_h;
	int hsl_size = img_w * img_h * (sizeof(int) + 2*sizeof(float));
	
	//std::cout<<"Width: "<<img_w<<"\nHeigth: "<<img_h<<"\n";
	
	
	uint8_t* gpu_GRAY = nullptr;
	err = cudaMalloc(&gpu_GRAY, gray_size);
	if (err != cudaSuccess){
		std::cout<<"HSLtoGRAY error alloc: "<<err<<"\n";
		return 0;
	}
	
	uint8_t* gpu_HSL = nullptr;
	cudaMalloc(&gpu_HSL, hsl_size);
	
	
	err = cudaMemcpy(gpu_HSL, *hsl_img, hsl_size, cudaMemcpyHostToDevice);
	
	if (err != cudaSuccess){
		std::cout<<"HSLtoGRAY error copy: "<<err<<"\n";
		return 0;
	}
	
	
	gpu_HSLtoGRAY_core << <blocksPerGrid, threadsPerBlock >> > (gpu_GRAY, gpu_HSL, img_w, img_h);
	
	err = cudaGetLastError();
	
	if (err != cudaSuccess){
		std::cout<<"HSLtoRGB error core: "<<err<<"\n";
		return 0;
	}

	cudaDeviceSynchronize();
	

	
	err = cudaMemcpy(*gray_img, gpu_GRAY, gray_size, cudaMemcpyDeviceToHost);
	
	if (err != cudaSuccess){
		std::cout<<"HSLtoRGB error copy back: "<<err<<"\n";
		return 0;
	}

	cudaFree(gpu_GRAY);
	cudaFree(gpu_HSL);
	
	return 1;
}
int GPU_utils::gpuGrayConvolve(float* filter, uint8_t** im, int filter_width, int filter_height, int im_width, int im_height, int passages){
	cudaError_t err = cudaSuccess;
	int img_size = im_width * im_height;
	int filter_size = filter_width * filter_height * sizeof(float);
	int threadsPerBlock = 256;
	int blocksPerGrid = (im_width * im_height + threadsPerBlock - 1) / threadsPerBlock;
	uint8_t* buffer = new uint8_t[img_size];

	uint8_t* gpu_IMG = nullptr;
	err = cudaMalloc(&gpu_IMG, img_size);
	if (err != cudaSuccess){
		std::cout<<"Error: "<<err<<"\n";
		return 0;
	}

	uint8_t* gpu_BUFFER = nullptr;
	cudaMalloc(&gpu_BUFFER, img_size);

	float* gpu_FILTER = nullptr;
	cudaMalloc(&gpu_FILTER, filter_size);

	err = cudaMemcpy(gpu_IMG, *im, img_size, cudaMemcpyHostToDevice);
	
	if (err != cudaSuccess){
		std::cout<<"Error: "<<err<<"\n";
		return 0;
	}

	cudaMemcpy(gpu_FILTER, filter, filter_size, cudaMemcpyHostToDevice);
	
	for (size_t i = 0; i < passages; i++){
		
		if (!(i%2)){
			gpu_convolve_gray_core << <blocksPerGrid, threadsPerBlock >> > (gpu_FILTER, gpu_BUFFER, gpu_IMG, filter_width, filter_height, im_width, im_height);
			
			if (!i){
				err = cudaGetLastError();
				
				if (err != cudaSuccess){
					std::cout<<"Error: "<<err<<"\n";
					return 0;
				}
			}
		}
		else{
			gpu_convolve_gray_core << <blocksPerGrid, threadsPerBlock >> > (gpu_FILTER, gpu_IMG, gpu_BUFFER, filter_width, filter_height, im_width, im_height);
		}

		

		cudaDeviceSynchronize();
	
	}

	if (passages%2)
		err = cudaMemcpy(buffer, gpu_BUFFER, img_size, cudaMemcpyDeviceToHost);
	else
		err = cudaMemcpy(buffer, gpu_IMG, img_size, cudaMemcpyDeviceToHost);
	
	if (err != cudaSuccess){
		std::cout<<"Error: "<<err<<"\n";
		return 0;
	}

	cudaFree(gpu_IMG);
	cudaFree(gpu_BUFFER);
	cudaFree(gpu_FILTER);

	uint8_t* temp = *im;
	*im = buffer;
	delete[] temp;
	return 1;
}
int GPU_utils::gpuGrayCannyEdge(uint8_t** gray_img, int im_w, int im_h, float thold_high, float thold_low){
	cudaError_t err = cudaSuccess;
	int img_size = im_w * im_h;
	int byte_size = img_size;
	int threadsPerBlock = 256;
	int blocksPerGrid = (im_w * im_h + threadsPerBlock - 1) / threadsPerBlock;
	
	Gradient* result = nullptr;
	uint8_t* gpu_IMG = nullptr;
	float h_filter [25] = {2,1,0,-1,-2,2,1,0,-1,-2,4,2,0,-2,-4,2,1,0,-1,-2,2,1,0,-1,-2};
	float v_filter [25] = {2,2,4,2,2,1,1,2,1,1,0,0,0,0,0,-1,-1,-2,-1,-1,-2,-2,-4,-2,-2};
	float gau_filter [25] = {0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625, 0.015625, 0.0625, 0.09375, 0.0625, 0.015625, 0.0234375, 0.09375, 0.140625, 0.09375, 0.0234375, 0.015625, 0.0625, 0.09375, 0.0625, 0.015625, 0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625};
	float* gpu_H_FILTER = nullptr;
	float* gpu_V_FILTER = nullptr;
	float* gpu_GAU_FILTER = nullptr;
	uint8_t* gpu_BUFFER = nullptr;
	
	err = cudaMalloc(&gpu_IMG, byte_size);
	if (err != cudaSuccess){
		std::cout<<"Allocation error: "<<err<<"\n";
		return 0;
	}
	
	cudaMalloc(&gpu_BUFFER, byte_size);
	cudaMalloc(&result, img_size*sizeof(Gradient));
	cudaMalloc(&gpu_H_FILTER, 25*sizeof(float));
	cudaMalloc(&gpu_V_FILTER, 25*sizeof(float));
	cudaMalloc(&gpu_GAU_FILTER, 25*sizeof(float));
	
	err = cudaMemcpy(gpu_IMG, *gray_img, byte_size, cudaMemcpyHostToDevice);
	
	if (err != cudaSuccess){
		std::cout<<"Copy fw error: "<<err<<"\n";
		return 0;
	}
	
	cudaMemcpy(gpu_H_FILTER, h_filter, 25*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_V_FILTER, v_filter, 25*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_GAU_FILTER, gau_filter, 25*sizeof(float), cudaMemcpyHostToDevice);
	
	
	gpu_convolve_gray_core << <blocksPerGrid, threadsPerBlock >> > (gpu_GAU_FILTER, gpu_BUFFER, gpu_IMG, 5, 5, im_w, im_h);
	
	
	cudaDeviceSynchronize();
	
	
	gpu_gradients_gray_core << <blocksPerGrid, threadsPerBlock>> > (gpu_V_FILTER, gpu_H_FILTER, result, gpu_BUFFER, 5, 5, im_w, im_h);
	

	cudaDeviceSynchronize();
	
	
	err = cudaGetLastError();
				
	if (err != cudaSuccess){
		std::cout<<"Core exec gradient error: "<<err<<"\n";
		return 0;
	}
	
	Gradient* temp = nullptr;
	cudaMalloc(&temp, img_size*sizeof(Gradient));
	
	/*Index_Cuple* index_list = new Index_Cuple[img_size];
	int* gpu_IND = nullptr;
	cudaMalloc(&gpu_IND, sizeof(Index_Cuple)*img_size);*/
	
	gpu_max_suppress << <blocksPerGrid, threadsPerBlock>> > (result, temp, im_w, im_h, thold_low);
	
	//cudaMemcpy(index_list, gpu_IND, sizeof(Index_Cuple)*img_size, cudaMemcpyDeviceToHost);
	
	cudaDeviceSynchronize();
	
	err = cudaGetLastError();
				
	if (err != cudaSuccess){
		std::cout<<"Core exec max_suppress error: "<<err<<"\n";
		return 0;
	}
	
	
	gpu_edge_tracking << <blocksPerGrid, threadsPerBlock>> > (temp, result, im_w, im_h, thold_high, thold_low);
	
	cudaDeviceSynchronize();
	
	err = cudaGetLastError();
				
	if (err != cudaSuccess){
		std::cout<<"Core exec edge track error: "<<err<<"\n";
		return 0;
	}
	
	gpu_gradient_to_gray_bin << <blocksPerGrid, threadsPerBlock>> > (result, gpu_IMG, im_w, im_h);

	cudaDeviceSynchronize();
	
	
	err = cudaGetLastError();
				
	if (err != cudaSuccess){
		std::cout<<"Core exec grad to bin error: "<<err<<"\n";
		return 0;
	}
	
	
	
	err = cudaMemcpy(*gray_img, gpu_IMG, byte_size, cudaMemcpyDeviceToHost);
	
	if (err != cudaSuccess){
		std::cout<<"Error copy back: "<<err<<"\n";
		return 0;
	}

	cudaFree(gpu_IMG);
	cudaFree(result);
	cudaFree(gpu_H_FILTER);
	cudaFree(gpu_V_FILTER);
	cudaFree(temp);
	
	return 1;
	
}