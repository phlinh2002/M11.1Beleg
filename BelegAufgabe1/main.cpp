// main.cpp
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

// Deklarationen der externen Funktionen
bool convertToGrayscale_OpenCL(const std::vector<unsigned char>& inputRGB, int width, int height, std::vector<unsigned char>& outputGray);
void convertToGrayscale_OpenMP(const std::vector<unsigned char>& inputRGB, int width, int height, std::vector<unsigned char>& outputGray);

int main() {
    std::string imagePath = "C:\\Users\\DELL\\source\\repos\\Programmierkonzepte\\images\\animal\\1.kitten_small.jpg";
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        std::cerr << "Fehler: Bild konnte nicht geladen werden.\n";
        return -1;
    }

    int width = img.cols;
    int height = img.rows;

    std::vector<unsigned char> rgbData(width * height * 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);
            int idx = (y * width + x) * 3;
            rgbData[idx] = pixel[2];      // R
            rgbData[idx + 1] = pixel[1];  // G
            rgbData[idx + 2] = pixel[0];  // B
        }
    }

    // OpenMP Graustufen
    std::vector<unsigned char> grayOpenMP;
    convertToGrayscale_OpenMP(rgbData, width, height, grayOpenMP);
    cv::Mat grayMatOpenMP(height, width, CV_8UC1, grayOpenMP.data());

    // OpenCL Graustufen
    std::vector<unsigned char> grayOpenCL;
    if (!convertToGrayscale_OpenCL(rgbData, width, height, grayOpenCL)) {
        std::cerr << "Fehler bei OpenCL Graustufen-Konvertierung.\n";
        return -1;
    }
    cv::Mat grayMatOpenCL(height, width, CV_8UC1, grayOpenCL.data());

    // Anzeigen
    cv::imshow("Original Image", img);
    cv::imshow("Grayscale OpenMP", grayMatOpenMP);
    cv::imshow("Grayscale OpenCL", grayMatOpenCL);

    cv::waitKey(0);
    return 0;
}
