#include <CL/cl.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

const char* brightnessKernel = R"(
__kernel void adjust_brightness(__global const uchar* input,
                                __global uchar* output,
                                int beta) {
    int i = get_global_id(0);
    
    int val = (int)(input[i]) + beta;
    if (val > 255) val = 255;
    if (val < 0) val = 0;
    output[i] = (uchar)(val);
}
)";

bool adjustBrightness_OpenCL(const std::vector<uchar>& inputColor, std::vector<uchar>& outputGray, int beta) {
    cl_int err;

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    // Plattform & Ger�t ausw�hlen
    err = clGetPlatformIDs(1, &platform, nullptr);
    err |= clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL Plattform/Device Fehler\n";
        return false;
    }

    // Kontext und Kommando-Warteschlange erstellen
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    queue = clCreateCommandQueueWithProperties(context, device, 0, &err);

    // Programm und Kernel erstellen
    cl_program program = clCreateProgramWithSource(context, 1, &brightnessKernel, nullptr, &err);
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        char log[2048];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
        std::cerr << "Build log:\n" << log << std::endl;
        return false;
    }
    cl_kernel kernel = clCreateKernel(program, "adjust_brightness", &err);

    // Speicher auf GPU anlegen
    size_t totalPixels = inputColor.size();
    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        totalPixels, const_cast<uchar*>(inputColor.data()), &err);
    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, totalPixels, nullptr, &err);

    // Kernel-Argumente setzen
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    clSetKernelArg(kernel, 2, sizeof(int), &beta);

    // Lokale und globale Gr��e definieren
    size_t localSize = 256; // 256 Threads pro Workgroup
    size_t globalSize = ((totalPixels + localSize - 1) / localSize) * localSize; // aufrunden

    // Pr�fen, ob Workgroup-Gr��e unterst�tzt wird
    size_t kernelWorkGroupSize = 0;
    err = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
        sizeof(size_t), &kernelWorkGroupSize, nullptr);
    if (err == CL_SUCCESS && localSize > kernelWorkGroupSize) {
        std::cerr << "Fehler: Lokale Workgroup-Gr��e ("
            << localSize << ") �berschreitet das Limit des Ger�ts ("
            << kernelWorkGroupSize << ")\n";
        return false;
    }

    // Kernel ausf�hren
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, &localSize, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Kernel Ausf�hrung Fehler: " << err << "\n";
        return false;
    }

    // Ergebnis zur�cklesen
    outputGray.resize(totalPixels);
    clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, totalPixels, outputGray.data(), 0, nullptr, nullptr);

    // Ressourcen freigeben
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);


    return true;
}
