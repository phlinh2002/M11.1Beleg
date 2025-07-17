#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

//!!!!!Aufgaben einzelne ausführen wenn die Ausgaben des Threads/Workloads aktiviert

bool convertToGrayscale_OpenCL(const std::vector<unsigned char>& inputRGB, int width, int height, std::vector<unsigned char>& outputGray);
void convertToGrayscale_OpenMP(const std::vector<unsigned char>& inputRGB, int width, int height, std::vector<unsigned char>& outputGray);
bool adjustBrightness_OpenCL(const std::vector<unsigned char>& inputRGB, std::vector<unsigned char>& outputGray,  int beta);
void adjustBrightness_OpenMP(const std::vector<unsigned char>& inputRGB, std::vector<unsigned char>& outputGray,int beta);
void convertToGrayscale_OpenCV(const cv::Mat& inputBGR, cv::Mat& outputGray);
void adjustBrightness_OpenCV(const cv::Mat& inputImage, cv::Mat& outputImage, int beta);

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
    int beta = 50;

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
    cv::Mat imgRGB;
    cv::cvtColor(img, imgRGB, cv::COLOR_BGR2RGB);

    
    // ===== Graustufen OpenMP =====
    std::vector<unsigned char> grayOpenMP;
    convertToGrayscale_OpenMP(rgbData, width, height, grayOpenMP);
    cv::Mat matGrayOpenMP(height, width, CV_8UC1, grayOpenMP.data());
    cv::imshow("Grayscale OpenMP", matGrayOpenMP);
    //cv::imwrite(outputFolder + "/nature_grayscale_openmp.jpg", matGrayOpenMP);
    
    /*
    // ===== Graustufen OpenCL =====
    std::vector<unsigned char> grayOpenCL;
    if (!convertToGrayscale_OpenCL(rgbData, width, height, grayOpenCL)) {
        std::cerr << "Fehler bei OpenCL Graustufen-Konvertierung.\n";
        return -1;
    }
    cv::Mat matGrayOpenCL(height, width, CV_8UC1, grayOpenCL.data());
	cv::imshow("Grayscale OpenCL", matGrayOpenCL);
    //cv::imwrite(outputFolder + "/nature_grayscale_opencl.jpg", matGrayOpenCL);
    
    // ===== Graustufen OpenCV =====
    cv::Mat grayOpenCV;
    convertToGrayscale_OpenCV(imgRGB, grayOpenCV);
	cv::imshow("Grayscale OpenCV", grayOpenCV);
    //cv::imwrite(outputFolder + "/nature_grayscale_opencv.jpg", grayOpenCV);

    
    
    // ===== Helligkeit OpenMP =====
    std::vector<unsigned char> brightOpenMP;
    adjustBrightness_OpenMP(rgbData, brightOpenMP, beta);
    cv::Mat matBrightOpenMP(height, width, CV_8UC3, brightOpenMP.data());
	cv::imshow("Brightness OpenMP", matBrightOpenMP);
    //cv::imwrite(outputFolder + "/nature_brightness_openmp.jpg", matBrightOpenMP);
    
    
    
    // ===== Helligkeit OpenCL =====
    std::vector<unsigned char> brightOpenCL;
    if (!adjustBrightness_OpenCL(rgbData, brightOpenCL, beta)) {
        std::cerr << "Fehler bei OpenCL Helligkeitsanpassung.\n";
        return -1;
    }
    cv::Mat matBrightOpenCL(height, width, CV_8UC3, brightOpenCL.data());
	cv::imshow("Brightness OpenCL", matBrightOpenCL);
    //cv::imwrite(outputFolder + "/nature_brightness_opencl.jpg", matBrightOpenCL);
    
    

	// ===== Helligkeit OpenCV =====
	cv::Mat brightOpenCV;
	adjustBrightness_OpenCV(imgRGB, brightOpenCV, beta);
	cv::imshow("Brightness OpenCV", brightOpenCV);
	//cv::imwrite(outputFolder + "/nature_brightness_opencv.jpg", brightOpenCV);
	*/
    

    cv::waitKey(0);
    return 0;
}
