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
using Vector = Eigen::VectorXd;
using Eigen::RowMajor;
using SparseMatrix = Eigen::SparseMatrix<double>;

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

SparseMatrix getSparseMatrixFromConv(int n, int m, const double convolution[9])
{
	int size = n * m;
	std::vector<Eigen::Triplet<double>> triplets;
	triplets.reserve(9 * size);

	for (int i = 0; i < size; ++i)
	{
		int c = i % m;
		bool L = c > 0;
		bool R = c < m - 1;
		bool U = i >= m;
		bool D = i < size - m;

		triplets.emplace_back(i, i, convolution[4]);
		if ((R) && convolution[5] != 0) triplets.emplace_back(i, i + 1, convolution[5]);
		if ((L) && convolution[3] != 0) triplets.emplace_back(i, i - 1, convolution[3]);
		if ((D) && convolution[7] != 0) triplets.emplace_back(i, i + m, convolution[7]);
		if ((U) && convolution[1] != 0) triplets.emplace_back(i, i - m, convolution[1]);
		if ((D && R) && convolution[8] != 0) triplets.emplace_back(i, i + m + 1, convolution[8]);
		if ((D && L) && convolution[6] != 0) triplets.emplace_back(i, i + m - 1, convolution[6]);
		if ((U && R) && convolution[2] != 0) triplets.emplace_back(i, i - m + 1, convolution[2]);
		if ((U && L) && convolution[0] != 0) triplets.emplace_back(i, i - m - 1, convolution[0]);
	}

	SparseMatrix M(size, size);
	M.setFromTriplets(triplets.begin(), triplets.end());

	return M;
}
//HELPER FUNCTIONS END------------------------

//CONVOLUTIONS START----------------------------------------------------------------------
double hav1[9] = {1.0/12, 1.0/12, 1.0/12, 1.0/12, 4.0/12, 1.0/12, 1.0/12, 1.0/12, 1.0/12};
//CONVOLUTIONS END------------------------------------------------------------------------





/*										CUT											*/





int	main(int argc, char *argv[])
{
	int width, height, channels;
	unsigned char *image_data;
	Matrix A;
	Matrix C;
	ImageMatrix image;
	Vector v;
	Vector w;
	SparseMatrix A1;
	Vector g1;
	Matrix G1;

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
	image = toImage(C);
	if (!saveImage("noisy_image.png", image, width, height)){return std::cerr << "Error: Could not save image\n", 1;}
	std::cout << "Noisy image saved to noisy_image.png" << std::endl;
	//TASK 2 END------------------------------------------------------------------------------------------------------
	//TASK 3 START----------------------------------------------------
	v = A.reshaped<RowMajor>();
	w = C.reshaped<RowMajor>();
	std::cout << "Size of v: " << v.size() << std::endl;
	std::cout << "Size of w: " << w.size() << std::endl;
	std::cout << "The Euclidean norm of v: " << v.norm() << std::endl;
	//TASK 3 END------------------------------------------------------
	//TASK 4 START------------------------------------------------------------------
	A1 = getSparseMatrixFromConv(A.rows(), A.cols(), hav1);
	std::cout << "Number of non-zero entries in A1: " << A1.nonZeros() << std::endl;
	//TASK 4 END--------------------------------------------------------------------
	//TASK 5 START------------------------------------------------------------------------------------------------------
	g1 = A1*w;
	G1 = g1.reshaped<RowMajor>(A.rows(), A.cols());
	image = toImage(G1);
	if (!saveImage("smoothed_image.png", image, width, height)){return std::cerr << "Error: Could not save image\n", 1;}
	return 0;
	//TASK 5 END--------------------------------------------------------------------------------------------------------
}