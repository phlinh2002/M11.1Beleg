#include <vector>
#include <omp.h>
#include <iostream>

void adjustBrightness_OpenMP(const std::vector<unsigned char>& input,
    std::vector<unsigned char>& output,
    int beta)
{
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

}
