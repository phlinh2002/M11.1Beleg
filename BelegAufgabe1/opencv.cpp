#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

void convertToGrayscale_OpenCV(const cv::Mat& inputBGR, cv::Mat& outputGray) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    std::cout << "-----Graustufen - OpenCV-----\n";
    auto start = std::chrono::high_resolution_clock::now();

    cv::cvtColor(inputBGR, outputGray, cv::COLOR_BGR2GRAY);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dauer = end - start;
    std::cout << "Laufzeit convertToGrayscale_OpenCV: " << dauer.count() << " ms\n";
}

void adjustBrightness_OpenCV(const cv::Mat& inputImage, cv::Mat& outputImage, int beta) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    std::cout << "-----Helligkeit - OpenCV-----\n";
    auto start = std::chrono::high_resolution_clock::now();

    inputImage.convertTo(outputImage, -1, 1, beta); // alpha=1 (Kontrast), beta (Helligkeit)

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dauer = end - start;
    std::cout << "Laufzeit adjustBrightness_OpenCV: " << dauer.count() << " ms\n";
}
