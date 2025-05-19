// BelegAufgabe1.cpp 
#include <iostream>
#include <omp.h>
#include <vector>
#include <opencv2/opencv.hpp>

struct RGB {
	unsigned char r;
	unsigned char g;
	unsigned char b;
};

//Funktion zur Umwandlung von RGB-Bild in Graustufen-Bild
void convert(const std::vector<RGB>& rgbImage, std::vector<unsigned char>& grayImage) {
	// Anzahl der Pixel im Bild
	int totalPixels = rgbImage.size();
	// Graustufen-Bild initialisieren
	grayImage.resize(totalPixels);
	// Graustufen-Bild berechnen
#pragma omp parallel for
	for (int i = 0;i < totalPixels; ++i) {
		const RGB& pixel = rgbImage[i];
		grayImage[i] = static_cast<unsigned char>(0.21 * pixel.r + 0.72 * pixel.g + 0.07 * pixel.b);
	}
}

int main()
{
	cv::Mat img = cv::imread("C:\\Users\\DELL\\source\\repos\\Programmierkonzepte\\images\\animal\\1.kitten_small.jpg");
	if (img.empty()) {
		std::cerr << "Error: Could not open image file." << std::endl;
		return -1;
	}

	int width = img.cols;
	int height = img.rows;

	std::vector<RGB> rgbImage(width * height);
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);
			rgbImage[y * width + x] = { pixel[2], pixel[1], pixel[0] };
		}
	}
	std::vector<unsigned char> grayImage;
	convert(rgbImage, grayImage);

	cv::Mat grayMat(height, width, CV_8UC1);
	for (int y = 0;y < height; ++y) {
		for (int x = 0;x < width;++x) {
			grayMat.at<uchar>(y, x) = grayImage[y * width + x];
		}
	}

	cv::imshow("Original Image", img);
	cv::imshow("Grayscale Image", grayMat);
	cv::waitKey(0);
	return 0;
}

