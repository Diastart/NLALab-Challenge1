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
	// -------------------------Challenge1 Task1-----------------------------------
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

	std::cout << "Size of th matrix: " << width << "x" << height << " with "
			  << channels << " channels in the file." << std::endl;
	
	// -------------------------Challenge1 Task2-----------------------------------
	 Eigen::MatrixXd A(height, width);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int index = y * width + x;
            A(y, x) = image_data[index];
        }
    }

	stbi_image_free(image_data);

	// std::cout << "Size: "
    //         << A.rows() << " x "
    //         << A.cols() << '\n';

	MatrixXd B = MatrixXd::Random(height, width);
	B = 50*B;

	MatrixXd C;

	C = A + B;

	Matrix<unsigned char, Dynamic, Dynamic, RowMajor> grayscale_image(height, width);
		grayscale_image = C.unaryExpr([](double val) -> unsigned char {
			return static_cast<unsigned char>(std::clamp(val, 0.0, 255.0));
		});

	const std::string output_image_path = "noisy_deer.png";
	if (stbi_write_png(output_image_path.c_str(), width, height, 1,
						grayscale_image.data(), width) == 0) {
		std::cerr << "Error: Could not save grayscale image" << std::endl;

		return 1;
	}

	std::cout << "Grayscale image saved to " << output_image_path << std::endl;

	// -------------------------Challenge1 Task3-----------------------------------

	VectorXd v = A.reshaped();
	VectorXd w = C.reshaped();

	std::cout << "The Euclidean norm of v: " << v.norm() << std::endl;

	return 0;
}
