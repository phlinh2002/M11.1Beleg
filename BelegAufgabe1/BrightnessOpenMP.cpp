#include <vector>
#include <omp.h>

// alpha ist Kontrastfaktor (meist > 0)
// beta ist Helligkeitsoffset (kann negativ oder positiv sein)
void adjustBrightness_OpenMP(const std::vector<unsigned char>& input,
    std::vector<unsigned char>& output,
     int beta)
{
    int totalPixels = input.size();
    output.resize(totalPixels);

#pragma omp parallel for
    for (int i = 0; i < totalPixels; ++i) {
        int value = static_cast<int>(input[i]) + beta;

        // Clamping auf [0,255]
        if (value > 255) value = 255;
        if (value < 0) value = 0;

        output[i] = static_cast<unsigned char>(value);
    }
}
