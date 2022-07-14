
#include "image.hpp"

#include "PearlImageConfig.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <chrono>
#include <map>

#include "json.hpp"
#include "thrpool.hpp"


//may return 0 when not able to detect
const auto processor_count = std::thread::hardware_concurrency();

static ThreadPool thrPool(processor_count > 0 ? processor_count - 1 : 4);

static void printToFile(const cv::Mat& mat, std::string path);

#define IMAGE_TYPE	CV_16UC4
//#define IMAGE_TYPE	CV_8UC4
#define IMAGE_TYPE2	CV_8UC4
#define IMAGE_TYPE3	CV_8UC1
#define IMAGE_TYPE4	CV_8UC3

#define FONT	cv::FONT_HERSHEY_PLAIN

using namespace image;

PearlImage::PearlImage() {
	
}

PearlImage::PearlImage(cv::Mat image) : PearlImage() {
	this->image = image;
}

PearlImage::PearlImage(int width, int height) : PearlImage(cv::Mat(width, height, IMAGE_TYPE)) {}

PearlImage::PearlImage(std::string path) : PearlImage(loadAlphaImage(path)) {}

cv::Mat PearlImage::makePearlImage(int width, int height, cv::InterpolationFlags algorithem, int gridInterval) {
	return makePearlImage(image, width, height, algorithem, gridInterval);
}

cv::Mat PearlImage::makePearlImage(std::string path, int width, int height, cv::InterpolationFlags algorithem, int gridInterval) {
	cv::Mat imgMat = loadAlphaImage(path);
	return makePearlImage(imgMat, width, height, algorithem, gridInterval);
}

cv::Mat PearlImage::makePearlImage(cv::Mat& image, int width, int height, cv::InterpolationFlags algorithem, int gridInterval) {
	std::pair<int, int> size = calculatePearlDimentions(image, width, height);
	auto start = std::chrono::system_clock::now();
	cv::Mat resizedImg = image::resize(image, size.first, size.second, algorithem);
	auto stop = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	//std::cout << "Resize time: " << duration.count() << "ms" << std::endl;
	auto start2 = std::chrono::system_clock::now();
	cv::Mat perlsImg = drawPearlImage(resizedImg, 12, 3, gridInterval);
	auto stop2 = std::chrono::system_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop2 - start2);
	//std::cout << "Draw time: " << duration.count() << "ms" << std::endl;
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop2 - start);
	//std::cout << "Total time: " << duration.count() << "ms" << std::endl;
	return perlsImg;
}

std::pair<int, int> PearlImage::calculatePearlDimentions(int horizontalPearls, int verticalPearls) {
	return calculatePearlDimentions(image, horizontalPearls, verticalPearls);
}

std::pair<int, int> PearlImage::calculatePearlDimentions(cv::Mat& image, int horizontalPearls, int verticalPearls) {
	if(!horizontalPearls) {
		horizontalPearls = (int) round(image.cols / (image.rows / verticalPearls));
	}
	if(!verticalPearls) {
		verticalPearls = (int) round(image.rows / (image.cols / horizontalPearls));
	}
	return std::pair(horizontalPearls, verticalPearls);
}

cv::Mat PearlImage::resize(int width, int height, cv::InterpolationFlags algorithem) {
	return image::resize(image, width, height, algorithem);
}

void PearlImage::setAvailableColours(std::map<std::string, cv::Scalar> availableColours) {
	this->availableColours = availableColours;
}

bool PearlImage::loadAvailableColours(std::string path) {
	std::ifstream ifs(path);
	if(ifs.is_open()) {
		json coloursJson = json::parse(ifs);
		std::map<std::string, cv::Scalar> availableColours = std::map<std::string, cv::Scalar>();
		for(auto item : coloursJson.items()) {
			availableColours[item.key()] = cv::Scalar(item.value()[0], item.value()[1], item.value()[2], item.value()[3]);
		}
		setAvailableColours(availableColours);
	}
	return ifs.is_open();
}

static double euclidianDistance(cv::Scalar c1, cv::Scalar c2) {
	return sqrt(pow(c1[0] - c2[0], 2) + pow(c1[1] - c2[1], 2) + pow(c1[2] - c2[2], 2) + pow(c1[3] - c2[3], 2));
}

std::string PearlImage::findClosestColourID(cv::Scalar colour) {
	std::string id = "";
	double dist = INT32_MAX;
	for(auto colourPair : availableColours) {
		double currentDist = euclidianDistance(colour, colourPair.second);
		if(currentDist < dist) {
			dist = currentDist;
			id = colourPair.first;
		}
	}
	return id;
}

cv::Mat PearlImage::drawPearlImage(cv::Mat& image, int pearlRadius, int pearlThickness, int gridInterval) {
	int pearlDiameter = pearlRadius * 2 + pearlThickness * 2;
	int w = image.cols * pearlDiameter;
	int h = image.rows * pearlDiameter;
	cv::Mat pearlImg = cv::Mat(h, w, IMAGE_TYPE2, cv::Scalar(0));

	int fontThickness = 1;
	double fontSize = getIdFontSize("00", (pearlDiameter - pearlThickness * 2) * 0.7, FONT, fontThickness);

	std::map<std::string, cv::Mat> pearls = std::map<std::string, cv::Mat>();
	cv::Scalar white(255, 255, 255, 255);
	
	drawGrid(pearlImg, gridInterval, pearlDiameter, white, 5);

	int num_of_threads = (int) thrPool.getSize();
	int rowsPerThread = (int) std::ceil(image.rows / (double) num_of_threads);

	for(auto colourPair : availableColours) {
		pearls[colourPair.first] = makePearl(pearlDiameter, pearlRadius, pearlThickness, colourPair.second, colourPair.first, fontSize, FONT, white, fontThickness);
	}

	for(int i = 1; i < num_of_threads; i++) {
		thrPool.doJob(std::bind(&PearlImage::partielDrawPearlImage, this, &pearlImg, &image, i * rowsPerThread, rowsPerThread, pearlDiameter, pearlRadius, pearlThickness, &pearls, pearlRadius + pearlThickness - pearlDiameter / 2));
	}
	
	partielDrawPearlImage(&pearlImg, &image, 0, rowsPerThread, pearlDiameter, pearlRadius, pearlThickness, &pearls, pearlRadius + pearlThickness - pearlDiameter / 2);

	//std::cout << image.size() << std::endl;

	thrPool.waitWhileBusy();
	
	return pearlImg;
}

void PearlImage::partielDrawPearlImage(cv::Mat* dst, cv::Mat* src, int startY, int rows, int pearlDiameter, int pearlRadius, int pearlThickness, std::map<std::string, cv::Mat>* pearls, int offset) {
	cv::Scalar color;
	std::string id = "";

	rows = std::min(rows, src->rows - startY);

	for(int y = startY; y < startY + rows; y++) {
		for(int x = 0; x < src->cols; x++) {
			color = cv::Scalar(src->at<cv::Vec4b>(y, x));

			if(color[3] > 0) {
				if(!availableColours.empty()) {
					id = findClosestColourID(color);
					color = availableColours[id];
					int xStart = x * pearlDiameter + offset;
					int yStart = y * pearlDiameter + offset;
					copyTo(&(*pearls)[id], xStart, yStart, dst);
				} else {
					drawPearl(*dst, x * pearlDiameter + pearlRadius + pearlThickness, y * pearlDiameter + pearlRadius + pearlThickness, color, pearlThickness, pearlRadius);
				}
			}
		}
	}
}

cv::Mat PearlImage::makePearl(int pearlDiameter, int pearlRadius, int pearlThickness, cv::Scalar pearlColor, std::string id, double fontSize, int font, cv::Scalar idColor, int fontThickness) {
	cv::Mat pearlImg = cv::Mat(pearlDiameter, pearlDiameter, IMAGE_TYPE2, cv::Scalar(0));
	drawColourID(pearlImg, pearlDiameter / 2, pearlDiameter / 2, id, fontSize, FONT, idColor, fontThickness);
	drawPearl(pearlImg, pearlDiameter / 2, pearlDiameter / 2, pearlColor, pearlThickness, pearlRadius);
	//image::save(pearlImg, "C:/Users/CapnOdin/Pictures/pearls/" + id + ".png");
	return pearlImg;
}

void PearlImage::drawPearl(cv::Mat& image, int x, int y, cv::Scalar color, int thickness, int radius) {
	cv::circle(image, cv::Point(x, y), radius, color, thickness, cv::LINE_AA);
}

double PearlImage::getIdFontSize(std::string id, double pearlSize, int font, int thickness) {
	int baseline;
	cv::Size size_default =	cv::getTextSize(id, font, 1, thickness, &baseline);
	//std::cout << "Pearl: " << pearlSize << " / " << size_default.width << " = " << (double) pearlSize / size_default.width << ", Baseline: " << baseline << std::endl;
	return (double) pearlSize / size_default.width;
}

void PearlImage::drawColourID(cv::Mat& image, int x, int y, std::string id, double fontSize, int font, cv::Scalar color, int thickness) {
	int baseline;
	cv::Size size = cv::getTextSize(id, font, fontSize, thickness, &baseline);
	cv::putText(image, id, cv::Point(x - size.width / 2, y + size.height / 2), font, fontSize, color, thickness, cv::LINE_AA);
}

void PearlImage::drawGrid(cv::Mat& image, int gridInterval, int pearlDiameter, cv::Scalar color, int thickness) {
	for(int i = 0; i < image.cols / gridInterval; i++) {
		cv::line(image, cv::Point(i * gridInterval * pearlDiameter, 0), cv::Point(i * gridInterval * pearlDiameter, image.rows), color, thickness);
	}
	for(int i = 0; i < image.rows / gridInterval; i++) {
		cv::line(image, cv::Point(0, i * gridInterval * pearlDiameter), cv::Point(image.cols, i * gridInterval * pearlDiameter), color, thickness);
	}
}

cv::Mat image::getAlpha(cv::Mat& image) {
	std::vector<cv::Mat> matChannels;
	cv::split(image, matChannels);
	return matChannels.at(3);
}

void image::setAlpha(cv::Mat& image, cv::Mat& alpha) {
	std::vector<cv::Mat> matChannels;
	cv::split(image, matChannels);
	matChannels.push_back(alpha);
	cv::merge(matChannels, image);
}

cv::Mat image::blur(cv::Mat& image) {
	cv::Mat blured;
	cv::Mat blured2;
	std::vector<cv::Mat> matChannels;
	cv::split(image, matChannels);
	cv::Mat alpha = matChannels.at(3);
	matChannels.pop_back();
	cv::merge(matChannels, blured);
	blured.convertTo(blured, IMAGE_TYPE4);
	//cv::bilateralFilter(blured, blured2, 9, 75, 75);
	cv::GaussianBlur(blured, blured2, cv::Size(5, 5), 0);
	cv::split(blured2, matChannels);
	matChannels.push_back(alpha);
	cv::merge(matChannels, blured);
	return blured;
}

cv::Mat image::resize(cv::Mat& image, int width, int height, cv::InterpolationFlags algorithem) {
	cv::Mat resized_image;
	cv::resize(image, resized_image, cv::Size(width, height), 0.0, 0.0, algorithem);
	return resized_image;
}

bool image::save(cv::Mat& image, std::string path) {
	return cv::imwrite(path, image);
}

MatTypeConvert image::getMaxValue(int type) {
	type = CV_MAT_DEPTH(type);
	switch(type) {
		case CV_8U:
			return MatTypeConvert(255.0, 0.0);
		case CV_8S:
			return MatTypeConvert(255.0, -128.0);
		case CV_16U:
			return MatTypeConvert(65535.0, 0.0);
		case CV_16S:
			return MatTypeConvert(65535.0, -32768.0);
		case CV_32S:
			return MatTypeConvert(4294967295.0, -2147483648.0);
		case CV_32F:
			return MatTypeConvert(FLT_MAX * 2.0, -FLT_MAX);
		case CV_64F:
			return MatTypeConvert(DBL_MAX * 2.0, -DBL_MAX);
		case CV_16F:
			return MatTypeConvert(0.0, 0.0);
		default:
			return MatTypeConvert(0.0, 0.0);
	}
}

cv::Mat image::convert(cv::Mat& mat, int type) {
	cv::Mat converted;
	MatTypeConvert mtcFrom = getMaxValue(mat.depth());
	MatTypeConvert mtcTo = getMaxValue(type);
	mat.convertTo(converted, type, mtcTo.max / mtcFrom.max, mtcTo.shift);
	return converted;
}

cv::Mat image::loadAlphaImage(std::string path) {
	std::vector<cv::Mat> matChannels;
	cv::Mat alpha;
	cv::Mat mat = cv::imread(path, cv::IMREAD_UNCHANGED);
	if(mat.channels() > 3) {
		cv::split(mat, matChannels);
		alpha = matChannels.at(3);
		alpha = convert(alpha, IMAGE_TYPE3);
	}
	mat = cv::imread(path);
	cv::split(mat, matChannels);
	if(alpha.empty()) {
		alpha = cv::Mat(matChannels.at(0).rows, matChannels.at(0).cols, matChannels.at(0).type(), cv::Scalar(255));
	}
	matChannels.push_back(alpha);
	cv::merge(matChannels, mat);
	return mat;
}

void image::copyTo(cv::Mat* src, int x, int y, cv::Mat* dst) {
	for(int y2 = std::max(y, 0); y2 < dst->rows; ++y2) {
		int fY = y2 - y;

		if(fY >= src->rows) {
			break;
		}

		for(int x2 = std::max(x, 0); x2 < dst->cols; ++x2) {
			int fX = x2 - x;

			if(fX >= src->cols) {
				break;
			}

			double opacity = ((double) src->data[fY * src->step + fX * src->channels() + 3]) / 255;

			for(int c = 0; opacity > 0 && c < dst->channels(); ++c) {
				unsigned char srcPx = src->data[fY * src->step + fX * src->channels() + c];
				unsigned char dstPx = dst->data[y2 * dst->step + x2 * dst->channels() + c];
				dst->data[y2 * dst->step + dst->channels() * x2 + c] = (uchar) (dstPx * (1.0 - opacity) + srcPx * opacity);
			}
		}
	}
}

std::string image::getVersion() {
	return std::to_string(PearlImage_VERSION_MAJOR) + "." + std::to_string(PearlImage_VERSION_MINOR) + "." + std::to_string(PearlImage_VERSION_PATCH) + (PearlImage_VERSION_TWEAK ? "-a." + std::to_string(PearlImage_VERSION_TWEAK) : "");
}

static void printToFile(const cv::Mat& mat, std::string path) {
	std::fstream f(path, std::ios::out);
	f << mat << std::endl;
}