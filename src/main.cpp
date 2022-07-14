
#include "image.hpp"
#include <string>

#include <chrono>

#define ITERATIONS	100

int main(int argc, char* argv[]) {
	std::cout << argv[0] << " Version " << image::getVersion() << std::endl;

	image::PearlImage pImage("C:/Users/CapnOdin/Pictures/Sparlock the Warrior Wizard.png");
	//image::PearlImage pImage("C:/Users/CapnOdin/Downloads/aphpa.png");

	pImage.loadAvailableColours("LinGust_Hama_colours.json");
	
	auto start = std::chrono::system_clock::now();

	cv::Mat pImageMat;
	for(int i = 0; i < ITERATIONS; i++) {
		pImageMat = pImage.makePearlImage(0, 60, cv::INTER_CUBIC, 5);
	}

	auto stop = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "Pearl time: " << duration.count() / ITERATIONS << "ms" << std::endl;

	image::save(pImageMat, "pearl.png");

	//cv::imshow("Pearl Image", pImageMat);
	//cv::waitKey();
	//cv::destroyAllWindows();
	return 0;
}

	