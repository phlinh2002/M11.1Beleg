#include "BenchmarkManager.h"
#include <fstream>
#include <iostream>
#include <functional>
#include <vector>

extern bool convertToGrayscale_OpenCL(const std::vector<unsigned char>&, int, int, std::vector<unsigned char>&);
extern void convertToGrayscale_OpenMP(const std::vector<unsigned char>&, int, int, std::vector<unsigned char>&);
extern void convertToGrayscale_OpenCV(const cv::Mat&, cv::Mat&);

extern bool adjustBrightness_OpenCL(const std::vector<unsigned char>&, std::vector<unsigned char>&, int);
extern void adjustBrightness_OpenMP(const std::vector<unsigned char>&, std::vector<unsigned char>&, int);
extern void adjustBrightness_OpenCV(const cv::Mat&, cv::Mat&, int);

BenchmarkManager::BenchmarkManager(const std::string& csvFilePath) : csvPath(csvFilePath) {
    // CSV Kopfzeile schon schreiben
    std::ofstream file(csvPath, std::ios::out);
    file << "Method,Time(ms)\n";
    file.close();
}

double BenchmarkManager::measureAverageTime(std::function<void()> func, int runs) {
    double total = 0.0;
    for (int i = 0; i < runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        total += std::chrono::duration<double, std::milli>(end - start).count();
    }
    return total / runs;
}
void BenchmarkManager::saveResults(const std::vector<std::pair<std::string, double>>& results) {
    // CSV neu erstellen (überschreiben)
    std::ofstream file(csvPath, std::ios::out);
    if (!file.is_open()) {
        std::cerr << "Fehler: CSV-Datei '" << csvPath << "' konnte nicht geöffnet werden.\n";
        return;
    }

    // Kopfzeile schreiben
    file << "Method,Time(ms)\n";

    // Ergebnisse schreiben
    for (const auto& pair : results) {
        file << pair.first << "," << pair.second << "\n";
    }
    file.close();

    // CSV-Datei öffnen und Inhalt ausgeben
    std::ifstream inFile(csvPath);
    if (!inFile.is_open()) {
        std::cerr << "Fehler: CSV-Datei '" << csvPath << "' konnte nicht zum Lesen geöffnet werden.\n";
        return;
    }

    std::cout << "\nBenchmark Results (CSV Content):\n";
    std::string line;
    while (std::getline(inFile, line)) {
        std::cout << line << "\n";
    }
    inFile.close();
}

void BenchmarkManager::benchmarkAll(const std::vector<unsigned char>& rgbData, int width, int height, const cv::Mat& imgRGB, int beta) {
    std::vector<std::pair<std::string, double>> results;

    // Grayscale OpenMP
    double t1 = measureAverageTime([&]() {
        std::vector<unsigned char> output;
        convertToGrayscale_OpenMP(rgbData, width, height, output);
        }, 10);
    results.emplace_back("Grayscale OpenMP", t1);

    // Grayscale OpenCL
    double t2 = measureAverageTime([&]() {
        std::vector<unsigned char> output;
        convertToGrayscale_OpenCL(rgbData, width, height, output);
        }, 10);
    results.emplace_back("Grayscale OpenCL", t2);

    // Grayscale OpenCV
    double t3 = measureAverageTime([&]() {
        cv::Mat output;
        convertToGrayscale_OpenCV(imgRGB, output);
        }, 10);
    results.emplace_back("Grayscale OpenCV", t3);

    // Brightness OpenMP
    double t4 = measureAverageTime([&]() {
        std::vector<unsigned char> output;
        adjustBrightness_OpenMP(rgbData, output, beta);
        }, 10);
    results.emplace_back("Brightness OpenMP", t4);

    // Brightness OpenCL
    double t5 = measureAverageTime([&]() {
        std::vector<unsigned char> output;
        adjustBrightness_OpenCL(rgbData, output, beta);
        }, 10);
    results.emplace_back("Brightness OpenCL", t5);

    // Brightness OpenCV
    double t6 = measureAverageTime([&]() {
        cv::Mat output;
        adjustBrightness_OpenCV(imgRGB, output, beta);
        }, 10);
    results.emplace_back("Brightness OpenCV", t6);

    // Ergebnisse speichern + ausgeben
    saveResults(results);
}
