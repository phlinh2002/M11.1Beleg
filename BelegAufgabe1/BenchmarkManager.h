#pragma once
#include <vector>
#include <string>
#include <functional>
#include <opencv2/opencv.hpp>

class BenchmarkManager {
public:
    // Konstruktor: erzeugt Datei, falls nicht vorhanden, und schreibt Kopfzeile
    explicit BenchmarkManager(const std::string& csvFilePath);

    // Führt alle Benchmarks aus (führt jede 10x aus und speichert + gibt Ergebnisse aus)
    void benchmarkAll(const std::vector<unsigned char>& rgbData, int width, int height, const cv::Mat& imgRGB, int beta);

private:
    std::string csvPath;

    // Misst Durchschnittsdauer (in ms) über mehrere Runs einer Funktion
    double measureAverageTime(std::function<void()> func, int runs = 10);

    // Speichert Ergebnisse in CSV und gibt sie im Terminal aus
    void saveResults(const std::vector<std::pair<std::string, double>>& results);
};
