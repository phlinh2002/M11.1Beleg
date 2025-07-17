#include <opencv2/opencv.hpp>
#include <iostream>

void convertToGrayscale_OpenCV(const cv::Mat& inputBGR, cv::Mat& outputGray) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    cv::cvtColor(inputBGR, outputGray, cv::COLOR_BGR2GRAY);
}

void adjustBrightness_OpenCV(const cv::Mat& inputImage, cv::Mat& outputImage, int beta) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    inputImage.convertTo(outputImage, -1, 1, beta); // alpha=1 (Kontrast), beta (Helligkeit)
}
