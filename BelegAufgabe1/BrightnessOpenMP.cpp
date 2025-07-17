#include <vector>
#include <omp.h>
#include <iostream>

void adjustBrightness_OpenMP(const std::vector<unsigned char>& input,
    std::vector<unsigned char>& output,
    int beta)
{
    int totalPixels = input.size();
    output.resize(totalPixels);

    // Parallelisierte Schleife
#pragma omp parallel for
    for (int i = 0; i < totalPixels; ++i) {
        int value = static_cast<int>(input[i]) + beta;

        if (value > 255) value = 255;
        if (value < 0) value = 0;

        output[i] = static_cast<unsigned char>(value);
    }

}
