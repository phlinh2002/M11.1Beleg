#include <vector>
#include <omp.h>
#include <iostream>
#include <chrono>

void convertToGrayscale_OpenMP(const std::vector<unsigned char>& inputRGB, int width, int height, std::vector<unsigned char>& outputGray) {
	printf("-----Graustufen - OpenMP-----\n");
	// Zeitmessung starten
	auto start = std::chrono::high_resolution_clock::now();

	int totalPixels = width * height;
	outputGray.resize(totalPixels);

	#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		int nThreads = omp_get_num_threads();
		// Nur ein Thread gibt die Anzahl der verwendeten Threads aus
		
	#pragma omp single
		{
			std::cout << "Anzahl der Threads: " << nThreads << std::endl;
		}

	

		// Ausgabe des Bereichs pro Thread (einmalig)
		/*
	#pragma omp critical
		{
			std::cout << "Thread " << tid << " bearbeitet Pixel von " << startIdx << " bis " << endIdx - 1
				<< " (Anzahl: " << (endIdx - startIdx) << ")\n";
		}*/
		

        // Jetzt die Pixel im Bereich verarbeiten
		for (int i = 0; i < totalPixels; ++i) {
        
			int idx = i * 3;
			unsigned char r = inputRGB[idx];
			unsigned char g = inputRGB[idx + 1];
			unsigned char b = inputRGB[idx + 2];
			outputGray[i] = static_cast<unsigned char>(0.21f * r + 0.72f * g + 0.07f * b);
		}
	}
	// Zeitmessung stoppen
	auto ende = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> dauer = ende - start;
	std::cout << "Laufzeit convertToGrayscale_OpenMP: " << dauer.count() << " ms\n";
}
