#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <unsupported/Eigen/SparseExtra>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using Matrix = Eigen::MatrixXd;

using ImageMatrix =
			Eigen::Matrix<unsigned char,
			Eigen::Dynamic,
			Eigen::Dynamic,
			Eigen::RowMajor>;

using ImageMap = Eigen::Map<ImageMatrix>;

//HElPER FUNCTIONS START----------------------
ImageMatrix toImage(const Matrix& matrix)
{
	return matrix.unaryExpr([](double value) {
		return static_cast<unsigned char>(
			std::clamp(value, 0.0, 255.0)
		);
	});
}

bool saveImage(const std::string& path, const ImageMatrix& image, int width, int height)
{
	return stbi_write_png(path.c_str(), width, height, 1, image.data(), width) != 0;
}
//HELPER FUNCTIONS END------------------------










int	main(int argc, char *argv[])
{
	int width, height, channels;
	unsigned char *image_data;
	Matrix A;
	Matrix C;

	if (argc < 2){ return std::cerr << "Usage: " << argv[0] << " <image_path>\n", 1;}

	//TASK 1 START--------------------------------------------------------------------------
	image_data = stbi_load(argv[1], &width, &height, &channels, 1);
	if (!image_data) {return std::cerr << "Error: Could not load image " << argv[1] << '\n', 1;}

	A = ImageMap(image_data, height, width).cast<double>();

	stbi_image_free(image_data);

	std::cout << "Size of the matrix: " << height << "x" << width << std::endl;
	//TASK 1 END-----------------------------------------------------------------------------

	//TASK 2 START----------------------------------------------------------------------------------------------------
	C = A + 50.0 * Matrix::Random(height, width);
	ImageMatrix image = toImage(C);
	if (!saveImage("noisy_image.png", image, width, height)){return std::cerr << "Error: Could not save image\n", 1;}
	std::cout << "Noisy image saved to noisy_image.png" << std::endl;
	//TASK 2 END------------------------------------------------------------------------------------------------------
}