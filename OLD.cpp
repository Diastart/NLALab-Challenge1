#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <unsupported/Eigen/SparseExtra>
typedef Eigen::Triplet<double> T;

// from https://github.com/nothings/stb/tree/master
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Eigen;

Eigen::SparseMatrix<double> buildConvolutionMatrix(int n, int m, const double convolution_values[9])
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

        triplets.emplace_back(i, i, convolution_values[4]);
        if ((R) && convolution_values[5] != 0) triplets.emplace_back(i, i + 1, convolution_values[5]);
        if ((L) && convolution_values[3] != 0) triplets.emplace_back(i, i - 1, convolution_values[3]);
        if ((D) && convolution_values[7] != 0) triplets.emplace_back(i, i + m, convolution_values[7]);
        if ((U) && convolution_values[1] != 0) triplets.emplace_back(i, i - m, convolution_values[1]);
        if ((D && R) && convolution_values[8] != 0) triplets.emplace_back(i, i + m + 1, convolution_values[8]);
        if ((D && L) && convolution_values[6] != 0) triplets.emplace_back(i, i + m - 1, convolution_values[6]);
        if ((U && R) && convolution_values[2] != 0) triplets.emplace_back(i, i - m + 1, convolution_values[2]);
        if ((U && L) && convolution_values[0] != 0) triplets.emplace_back(i, i - m - 1, convolution_values[0]);
    }

    Eigen::SparseMatrix<double> M(size, size);
    M.setFromTriplets(triplets.begin(), triplets.end());

    return M;
}

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

	const std::string output_image_path = "Task2.png";
	if (stbi_write_png(output_image_path.c_str(), width, height, 1,
						grayscale_image.data(), width) == 0) {
		std::cerr << "Error: Could not save grayscale image" << std::endl;

		return 1;
	}

	std::cout << "Grayscale image saved to " << output_image_path << std::endl;

	// -------------------------Challenge1 Task3-----------------------------------

	VectorXd v = A.reshaped<RowMajor>();
	VectorXd w = C.reshaped<RowMajor>();

	std::cout << "The Euclidean norm of v: " << v.norm() << std::endl;

	int n = A.rows();
	int m = A.cols();


	// -------------------------Challenge1 Task4------------------------------

	double convolution_values[9] = {1.0/12, 1.0/12, 1.0/12, 1.0/12, 4.0/12, 1.0/12, 1.0/12, 1.0/12, 1.0/12};

	SparseMatrix<double> A1 = buildConvolutionMatrix(n, m, convolution_values);

	std::cout << "Number of non-zero entries in A1: " << A1.nonZeros() << std::endl;

    //------------------------Challenge1 Task5----------------------

	VectorXd g1 = A1*w;
	MatrixXd G = g1.reshaped<RowMajor>(n, m);

	Matrix<unsigned char, Dynamic, Dynamic, RowMajor> grayscale_image2(n, m);
	grayscale_image2 = G.unaryExpr([](double val) -> unsigned char {
		return static_cast<unsigned char>(std::clamp(val, 0.0, 255.0));
	});

	const std::string output_image_path2 = "Task5.png";
	if (stbi_write_png(output_image_path2.c_str(), width, height, 1,
						grayscale_image2.data(), width) == 0) {
		std::cerr << "Error: Could not save grayscale image" << std::endl;
	}

	//-------------------------Challenge1 Task 6----------------------

	double convolution_values1[9] = {0.0, -3.0, 0.0, -1.0, 9.0, -3.0, 0.0, -1.0, 0.0};

	SparseMatrix<double> A2 = buildConvolutionMatrix(n, m, convolution_values1);

	std::cout << "Number of non-zero entries in A2: " << A2.nonZeros() << std::endl;

	if (A2.isApprox(A2.transpose())) std::cout << "The matirx A2 is symmetric" << std::endl;
	else std::cout << "The matirx A2 is not symmetric" << std::endl;

	//-------------------------Challenge1 Task7----------------------
	VectorXd f1 = A2*v;
	MatrixXd F = f1.reshaped<RowMajor>(n, m);

	Matrix<unsigned char, Dynamic, Dynamic, RowMajor> grayscale_image21(n, m);
	grayscale_image21 = F.unaryExpr([](double val) -> unsigned char {
		return static_cast<unsigned char>(std::clamp(val, 0.0, 255.0));
	});

	const std::string output_image_path21 = "Task7.png";
	if (stbi_write_png(output_image_path21.c_str(), width, height, 1,
						grayscale_image21.data(), width) == 0) {
		std::cerr << "Error: Could not save grayscale image" << std::endl;
	}

	//-------------------------Challenge1 Task8----------------------

	Eigen::saveMarket(A2, "./A2.mtx");
	// LIS expects the rhs in "vector coordinate" format, not the matrix format written by saveMarket
	FILE* out = fopen("w.mtx", "w");
	fprintf(out, "%%%%MatrixMarket vector coordinate real general\n");
	fprintf(out, "%d\n", (int)w.size());
	for (int i = 0; i < w.size(); ++i)
		fprintf(out, "%d %.16e\n", i + 1, w(i));
	fclose(out);


	


	//-------------------------Challenge1 Task9----------------------

	//-------------------------Challenge1 Task10----------------------

	double convolution_values2[9] = {-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0};

	SparseMatrix<double> A3 = buildConvolutionMatrix(n, m, convolution_values2);

	if (A3.isApprox(A3.transpose())) std::cout << "The matirx A3 is symmetric" << std::endl;
	else std::cout << "The matirx A3 is not symmetric" << std::endl;

	//-------------------------Challenge1 Task11----------------------

	VectorXd h1 = A3*v;
	MatrixXd H = h1.reshaped<RowMajor>(n, m);

	Matrix<unsigned char, Dynamic, Dynamic, RowMajor> grayscale_image22(n, m);
	grayscale_image22 = H.unaryExpr([](double val) -> unsigned char {
		return static_cast<unsigned char>(std::clamp(val, 0.0, 255.0));
	});

	const std::string output_image_path22 = "Task11.png";
	if (stbi_write_png(output_image_path22.c_str(), width, height, 1,
						grayscale_image22.data(), width) == 0) {
		std::cerr << "Error: Could not save grayscale image" << std::endl;
	}

	//-------------------------Challenge1 Task12----------------------

	//-------------------------Challenge1 Task13----------------------
	return 0;
}
