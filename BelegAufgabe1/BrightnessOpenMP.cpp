#include <vector>
#include <omp.h>
#include <iostream>
#include <chrono>

void adjustBrightness_OpenMP(const std::vector<unsigned char>& input,
    std::vector<unsigned char>& output,
    int beta)
{
     printf("-----Helligkeit - OpenMP-----\n");
    auto start = std::chrono::high_resolution_clock::now();

    int totalPixels = input.size();
    output.resize(totalPixels);

#pragma omp parallel
    {
        // Ausgabe der Gesamtzahl der Threads, nur einmal durch einen Thread
        /*
        #pragma omp single
        {
            int nThreads = omp_get_num_threads();
            std::cout << "Anzahl der verwendeten Threads: " << nThreads << std::endl;
        }
        */

        int tid = omp_get_thread_num();
        int nThreads = omp_get_num_threads();

     

        // Einmalige Ausgabe pro Thread
        /*
        #pragma omp critical
        {
            std::cout << "Thread " << tid << " bearbeitet Pixel von " << startIdx
                      << " bis " << endIdx - 1
                      << " (Anzahl: " << (endIdx - startIdx) << ")\n";
        }
        */

        for (int i = 0; i < totalPixels; ++i) {
            int value = static_cast<int>(input[i]) + beta;

            if (value > 255) value = 255;
            if (value < 0) value = 0;

            output[i] = static_cast<unsigned char>(value);
        }
    }

    auto ende = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dauer = ende - start;
    std::cout << "Laufzeit adjustBrightness_OpenMP: " << dauer.count() << " ms\n";
}
