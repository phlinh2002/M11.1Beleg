#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

//Bis jetzt nur Grayscale, Helligkeit funktioniert noch nicht, weil ich möchte kein Konstrast verändert sondern nur Helligkeit 
// Externe Funktionen
bool convertToGrayscale_OpenCL(const std::vector<unsigned char>& inputRGB, int width, int height, std::vector<unsigned char>& outputGray);
void convertToGrayscale_OpenMP(const std::vector<unsigned char>& inputRGB, int width, int height, std::vector<unsigned char>& outputGray);
bool adjustBrightness_OpenCL(const std::vector<unsigned char>& inputRGB, std::vector<unsigned char>& outputGray,  int beta);
void adjustBrightness_OpenMP(const std::vector<unsigned char>& inputRGB, std::vector<unsigned char>& outputGray,int beta);

int main() {
    std::string outputFolder = "./images";
    std::string imagePath = "\images\\1.kitten_small.jpg";
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        std::cerr << "Fehler: Bild konnte nicht geladen werden.\n";
        return -1;
    }

    int width = img.cols;
    int height = img.rows;

    // RGB-Daten extrahieren
    std::vector<unsigned char> rgbData(width * height * 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);
            int idx = (y * width + x) * 3;
            rgbData[idx] = pixel[2]; // R
            rgbData[idx + 1] = pixel[1]; // G
            rgbData[idx + 2] = pixel[0]; // B
        }
    }

    // ===== Graustufen OpenMP =====
    std::vector<unsigned char> grayOpenMP;
    convertToGrayscale_OpenMP(rgbData, width, height, grayOpenMP);
    cv::Mat matGrayOpenMP(height, width, CV_8UC1, grayOpenMP.data());

    // ===== Graustufen OpenCL =====
    std::vector<unsigned char> grayOpenCL;
    if (!convertToGrayscale_OpenCL(rgbData, width, height, grayOpenCL)) {
        std::cerr << "Fehler bei OpenCL Graustufen-Konvertierung.\n";
        return -1;
    }
    cv::Mat matGrayOpenCL(height, width, CV_8UC1, grayOpenCL.data());

    int beta = 50;      // Helligkeitsversatz

    // ===== Helligkeit OpenMP =====
    std::vector<unsigned char> brightOpenMP;
    adjustBrightness_OpenMP(rgbData, brightOpenMP, beta);
    cv::Mat matBrightOpenMP(height, width, CV_8UC3, brightOpenMP.data());

    // ===== Helligkeit OpenCL =====
    std::vector<unsigned char> brightOpenCL;
    if (!adjustBrightness_OpenCL(rgbData, brightOpenCL, beta)) {
        std::cerr << "Fehler bei OpenCL Helligkeitsanpassung.\n";
        return -1;
    }
    cv::Mat matBrightOpenCL(height, width, CV_8UC3, brightOpenCL.data());

    // ===== Bilder speichern =====
    cv::imwrite(outputFolder + "/grayscale_openmp.jpg", matGrayOpenMP);
    cv::imwrite(outputFolder + "/grayscale_opencl.jpg", matGrayOpenCL);
    cv::imwrite(outputFolder + "/brightness_openmp.jpg", matBrightOpenMP);
    cv::imwrite(outputFolder + "/brightness_opencl.jpg", matBrightOpenCL);

    // ===== Bilder anzeigen =====
    cv::imshow("Original", img);
    cv::imshow("Grayscale OpenMP", matGrayOpenMP);
    cv::imshow("Grayscale OpenCL", matGrayOpenCL);
    cv::imshow("Brightness Adjusted OpenMP", matBrightOpenMP);
    cv::imshow("Brightness Adjusted OpenCL", matBrightOpenCL);

    cv::waitKey(0);
    return 0;
}
