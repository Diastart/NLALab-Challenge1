#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>

// from https://github.com/nothings/stb/tree/master
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Eigen;

int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		std::cerr << "Usage: " << argv[0] << "<image_path>" << std::endl;
		return 1;
	}

	const char *input_image_path = argv[1];

	// Load the image using stb_image
	int width, height, channels;
	// for greyscale images force to load only one channel
	unsigned char *image_data = stbi_load(input_image_path, &width, &height, &channels, 1);
	if (!image_data)
	{
		std::cerr << "Error: Could not load image " << input_image_path << std::endl;
		return 1;
	}

	std::cout << "Image loaded: " << width << "x" << height << " with "
			  << channels << " channels in the file (forced to 1)." << std::endl;
	return 0;
}
