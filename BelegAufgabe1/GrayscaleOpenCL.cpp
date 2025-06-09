#include <CL/cl.h>
#include <vector>
#include <iostream>

const char* kernelSource = R"(
__kernel void rgb_to_grayscale(__global const uchar* inputImage,
                               __global uchar* outputImage,
                               int width, int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int idx = (y * width + x) * 3;
    int outIdx = y * width + x;


    if (x < width && y < height) {
        uchar r = inputImage[idx];
        uchar g = inputImage[idx + 1];
        uchar b = inputImage[idx + 2];
        outputImage[outIdx] = (uchar)(0.21f * r + 0.72f * g + 0.07f * b);

        // Begrenzte Ausgabe (z. B. alle 64 Pixel)
        if ((x % 64 == 0) && (y % 64 == 0)) {
            printf("Work-Item (%d, %d) verarbeitet Pixel %d\n", x, y, outIdx);
        }
    }
}
)";

bool convertToGrayscale_OpenCL(const std::vector<unsigned char>& inputRGB, int width, int height, std::vector<unsigned char>& outputGray) {
    cl_int err;

    cl_platform_id platform;
    cl_device_id device;
    err = clGetPlatformIDs(1, &platform, nullptr);
    err |= clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL Plattform/Device Fehler\n";
        return false;
    }

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    cl_command_queue_properties props[] = { 0 };
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, props, &err);

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        char log[2048];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
        std::cerr << "Kernel Build Fehler:\n" << log << std::endl;
        return false;
    }

    cl_kernel kernel = clCreateKernel(program, "rgb_to_grayscale", &err);

    size_t imageSize = inputRGB.size();
    size_t graySize = width * height;

    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imageSize, const_cast<unsigned char*>(inputRGB.data()), &err);
    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, graySize, nullptr, &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    clSetKernelArg(kernel, 2, sizeof(int), &width);
    clSetKernelArg(kernel, 3, sizeof(int), &height);

    size_t globalSize[2] = { (size_t)width, (size_t)height };
    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Kernel Ausführung Fehler\n";
        return false;
    }

    outputGray.resize(graySize);
    clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, graySize, outputGray.data(), 0, nullptr, nullptr);

    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return true;
}
