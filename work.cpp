#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>
typedef Eigen::Triplet<double> T;

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

	int n = A.rows();
	int m = A.cols();


	// -------------------------Challenge1 Task4------------------------------


	int size = n*m;
	std::vector<Eigen::Triplet<double>> triplets;
	triplets.reserve(9*size);
	
	double convolution_values[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};

	for (int i = 0; i < size; ++i) {
    	int c = i % m;
	    bool L = c > 0, R = c < m-1;      // column neighbors exist
    	bool U = i >= m, D = i < size-m;  // row neighbors exist

	    triplets.emplace_back(i, i, convolution_values[4]);
    	if (R)     triplets.emplace_back(i, i+1,  convolution_values[5]);
	    if (L)     triplets.emplace_back(i, i-1,   convolution_values[3]);
    	if (D)     triplets.emplace_back(i, i+m,   convolution_values[7]);
	    if (U)     triplets.emplace_back(i, i-m,   convolution_values[1]);
    	if (D&&R)  triplets.emplace_back(i, i+m+1, convolution_values[8]);
	    if (D&&L)  triplets.emplace_back(i, i+m-1, convolution_values[6]);
    	if (U&&R)  triplets.emplace_back(i, i-m+1, convolution_values[2]);
	    if (U&&L)  triplets.emplace_back(i, i-m-1, convolution_values[0]);
	}

	Eigen::SparseMatrix<double> A1(size, size);
	A1.setFromTriplets(triplets.begin(), triplets.end());


    //------------------------Challenge 1 Task 5----------------------

	VectorXd g1 = A1*w;
	MatrixXd G = g1.reshaped(n, m);

	Matrix<unsigned char, Dynamic, Dynamic, RowMajor> grayscale_image2(m, n);
	grayscale_image2 = G.unaryExpr([](double val) -> unsigned char {
		return static_cast<unsigned char>(std::clamp(val, 0.0, 255.0));
	});

	const std::string output_image_path2 = "task5.png";
	if (stbi_write_png(output_image_path2.c_str(), width, height, 1,
						grayscale_image2.data(), width) == 0) {
		std::cerr << "Error: Could not save grayscale image" << std::endl;
		
	}	
	//---------------------------------------------------------------

	/* Claude Code

	// A1 is (m*n)x(m*n) with at most 9 nonzeros per row: store it in COO format
	// (three parallel arrays: row index, column index, value) instead of dense
	int N = m * n;
	std::vector<int> coo_row, coo_col;
	std::vector<double> coo_val;
	coo_row.reserve(9 * N);
	coo_col.reserve(9 * N);
	coo_val.reserve(9 * N);

	const int offsets[] = {0, 1, -1, n, -n, n + 1, n - 1, -n + 1, -n - 1};
	for (int i = 0; i < N; ++i)
	{
		for (int d : offsets)
		{
			int j = i + d;
			if (j >= 0 && j < N)
			{
				coo_row.push_back(i);
				coo_col.push_back(j);
				coo_val.push_back(d == 0 ? 4.0/12.0 : 1.0/12.0);
			}
		}
	}

	std::cout << "A1 size: " << N << "x" << N
			  << ", nonzeros: " << coo_val.size() << std::endl;

	// matrix-vector product g = A1 * w directly from the COO triplets
	VectorXd g = VectorXd::Zero(N);
	for (size_t k = 0; k < coo_val.size(); ++k)
		g(coo_row[k]) += coo_val[k] * w(coo_col[k]);

	MatrixXd G = g.reshaped(m, n);
	
	MatrixXd G = g.reshaped(m, n);
	
	Matrix<unsigned char, Dynamic, Dynamic, RowMajor> grayscale_image2(m, n);
		grayscale_image2 = G.unaryExpr([](double val) -> unsigned char {
			return static_cast<unsigned char>(std::clamp(val, 0.0, 255.0));
		});

	const std::string output_image_path2 = "task5.png";
	if (stbi_write_png(output_image_path2.c_str(), width, height, 1,
						grayscale_image2.data(), width) == 0) {
		std::cerr << "Error: Could not save grayscale image" << std::endl;

		return 1;
	}

	std::cout << "Grayscale image saved to " << output_image_path2 << std::endl;

		return 1;
	}

	std::cout << "Grayscale image saved to " << output_image_path2 << std::endl;
	
*/
	return 0;
}
