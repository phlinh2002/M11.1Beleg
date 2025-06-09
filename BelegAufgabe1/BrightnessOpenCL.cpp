#include <CL/cl.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <chrono>

const char* brightnessKernel = R"(
__kernel void adjust_brightness(__global const uchar* input,
                                __global uchar* output,
                                int beta,
                                int totalPixels) {
    int i = get_global_id(0);
    if (i < totalPixels) {
        int val = (int)(input[i]) + beta;
        if (val > 255) val = 255;
        if (val < 0) val = 0;
        output[i] = (uchar)(val);

        // BEGRENZT, um nicht tausende Zeilen zu erzeugen
        if (i % 10000 == 0) {
            printf("Work-Item %d bearbeitet Pixel %d\n", i, i);
    }
}
}
)";

bool adjustBrightness_OpenCL(const std::vector<uchar>& inputColor, std::vector<uchar>& outputGray,
    int beta) {
    printf("-----Helligkeit - OpenCL-----\n");
    auto start = std::chrono::high_resolution_clock::now();
    cl_int err;

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    err = clGetPlatformIDs(1, &platform, nullptr);
    err |= clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    queue = clCreateCommandQueueWithProperties(context, device, 0, &err);

    // Programm und Kernel
    cl_program program = clCreateProgramWithSource(context, 1, &brightnessKernel, nullptr, &err);
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    if (err != CL_SUCCESS) {
        char log[2048];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
        std::cerr << "Build log:\n" << log << std::endl;
        return false;
    }

    cl_kernel kernel = clCreateKernel(program, "adjust_brightness", &err);

    size_t totalPixels = inputColor.size();
    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        totalPixels, const_cast<uchar*>(inputColor.data()), &err);
    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, totalPixels, nullptr, &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    clSetKernelArg(kernel, 2, sizeof(int), &beta);
    clSetKernelArg(kernel, 3, sizeof(int), &totalPixels);

    size_t globalSize = totalPixels;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);

    outputGray.resize(totalPixels);
    clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, totalPixels, outputGray.data(), 0, nullptr, nullptr);

    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    auto ende = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dauer = ende - start;
    std::cout << "Laufzeit adjustBrightness_OpenCL: " << dauer.count() << " ms\n";

    return true;
}
