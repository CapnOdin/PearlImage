#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace image {
	class PearlImage {
	private:
		cv::Mat image;
		std::map<std::string, cv::Scalar> availableColours;

	public:
		PearlImage();
		PearlImage(int width, int height);
		PearlImage(std::string path);
		PearlImage(cv::Mat image);

				cv::Mat		makePearlImage(int width, int height, cv::InterpolationFlags algorithem, int gridInterval = 0);
				cv::Mat		makePearlImage(std::string path, int width, int height, cv::InterpolationFlags algorithem, int gridInterval = 0);
				cv::Mat		makePearlImage(cv::Mat& image, int width, int height, cv::InterpolationFlags algorithem, int gridInterval = 0);
				cv::Mat		resize(int width, int height, cv::InterpolationFlags algorithem);
				void		setAvailableColours(std::map<std::string, cv::Scalar> availableColours);
				bool		loadAvailableColours(std::string path);
				std::string	findClosestColourID(cv::Scalar colour);
				std::pair<int, int> calculatePearlDimentions(int horizontalPearls, int verticalPearls);
		static	std::pair<int, int> calculatePearlDimentions(cv::Mat& image, int horizontalPearls, int verticalPearls);

	private:
				cv::Mat	drawPearlImage(cv::Mat& image, int pearlRadius, int pearlThickness, int gridInterval);
		static	void	drawPearl(cv::Mat& image, int x, int y, cv::Scalar color, int thickness, int radius);
		static	void	drawColourID(cv::Mat& image, int x, int y, std::string id, double fontSize, int font = cv::FONT_HERSHEY_PLAIN, cv::Scalar color = cv::Scalar(255, 255, 255, 255), int thickness = 1);
		static	void	drawGrid(cv::Mat& image, int gridInterval, int pearlDiameter, cv::Scalar color, int thickness);
		static	double	getIdFontSize(std::string id, double pearlSize, int font = cv::FONT_HERSHEY_PLAIN, int thickness = 1);
				cv::Mat	makePearl(int pearlDiameter, int pearlRadius, int pearlThickness, cv::Scalar pearlColor, std::string id, double fontSize, int font, cv::Scalar idColor, int fontThickness);
				void	partielDrawPearlImage(cv::Mat* dst, cv::Mat* src, int startY, int rows, int pearlDiameter, int pearlRadius, int pearlThickness, std::map<std::string, cv::Mat>* pearls, int offset);
	};

	struct MatTypeConvert {
		double	max;
		double	shift;
	};

	std::string		getVersion();
	cv::Mat			getAlpha(cv::Mat& image);
	void			setAlpha(cv::Mat& image, cv::Mat& alpha);
	cv::Mat			blur(cv::Mat& image);
	cv::Mat			resize(cv::Mat& image, int width, int height, cv::InterpolationFlags algorithem);
	bool			save(cv::Mat& image, std::string path);
	MatTypeConvert	getMaxValue(int type);
	cv::Mat			convert(cv::Mat& mat, int type);
	cv::Mat			loadAlphaImage(std::string path);
	void			copyTo(cv::Mat* src, int x, int y, cv::Mat* dst);
}
