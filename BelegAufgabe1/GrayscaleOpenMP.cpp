#include <vector>
#include <omp.h>
#include <iostream>

void convertToGrayscale_OpenMP(const std::vector<unsigned char>& inputRGB, int width, int height, std::vector<unsigned char>& outputGray) {
	int totalPixels = width * height;
	outputGray.resize(totalPixels);

     // Jetzt die Pixel im Bereich verarbeiten
	#pragma omp parallel for
		for(int i = 0; i < totalPixels; ++i) {
        
			int idx = i * 3;
			unsigned char r = inputRGB[idx];
			unsigned char g = inputRGB[idx + 1];
			unsigned char b = inputRGB[idx + 2];
			outputGray[i] = static_cast<unsigned char>(0.21f * r + 0.72f * g + 0.07f * b);
		}
	
}
