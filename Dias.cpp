#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <unsupported/Eigen/SparseExtra>
#include <fstream>
#include <iomanip>
#include <stdexcept>

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


bool saveLISVector(const Vector& v, const std::string& path)
{
	std::ofstream out(path);

	if (!out)
		return false;

	out << "%%MatrixMarket vector coordinate real general\n";
	out << v.size() << '\n';

	for (int i = 0; i < v.size(); ++i)
		out << i + 1 << ' ' << std::scientific << std::setprecision(16)
			<< v(i) << '\n';

	return true;
}


Vector loadLISVector(const std::string& path)
{
	std::ifstream in(path);

	if (!in)
		throw std::runtime_error("Could not open " + path);

	std::string line;
	std::getline(in, line); // MatrixMarket header

	int size;
	in >> size;

	Vector x(size);

	int index;
	double value;

	while (in >> index >> value)
		x(index - 1) = value;

	return x;
}
//HELPER FUNCTIONS END------------------------

//CONVOLUTIONS START----------------------------------------------------------------------
double hav1[9] = {1.0/12, 1.0/12, 1.0/12, 1.0/12, 4.0/12, 1.0/12, 1.0/12, 1.0/12, 1.0/12};
double hsh1[9] = {0.0, -3.0, 0.0, -1.0, 9.0, -3.0, 0.0, -1.0, 0.0};
double hed2[9] = {-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0};
//CONVOLUTIONS END------------------------------------------------------------------------





/*										CUT											*/





int	main(int argc, char *argv[])
{
	int				width, height, channels;
	unsigned char	*image_data;
	Matrix			A;
	Matrix			C;
	ImageMatrix		image;
	Vector			v;
	Vector			w;
	SparseMatrix	A1;
	Vector			g1;
	Matrix			G1;
	Vector			g2;
	Matrix			G2;
	SparseMatrix	A2;
	Vector			x;
	Matrix			X;
	SparseMatrix	A3;
	Vector			g3;
	Matrix			G3;
	SparseMatrix	A3M;
	Vector			y;
	Matrix			Y;

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
	std::cout << "Smoothed image saved to smoothed_image.png" << std::endl;
	//TASK 5 END--------------------------------------------------------------------------------------------------------
	//TASK 6 START--------------------------------------------------------------------------------------
	A2 = getSparseMatrixFromConv(A.rows(), A.cols(), hsh1);
	std::cout << "Number of non-zero entries in A2: " << A2.nonZeros() << std::endl;
	std::cout << (A2.isApprox(A2.transpose()) ? "A2 is symmetric" : "A2 is NOT symmetric") << std::endl;
	//TASK 6 END----------------------------------------------------------------------------------------
	//TASK 7 START-------------------------------------------------------------------------------------------------------
	g2 = A2*v;
	G2 = g2.reshaped<RowMajor>(A.rows(), A.cols());
	image = toImage(G2);
	if (!saveImage("sharpened_image.png", image, width, height)){return std::cerr << "Error: Could not save image\n", 1;}
	std::cout << "Sharpened image saved to sharpened_image.png" << std::endl;
	//TASK 7 END---------------------------------------------------------------------------------------------------------
	//TASK 8 START------------------------------------------------------------
	Eigen::saveMarket(A2, "./A2.mtx");
	saveLISVector(w, "w.mtx");
	/*		./test1 A2.mtx w.mtx x.mtx -i bicgstab -p ilu -tol 1e-12		*/
	//TASK 8 END--------------------------------------------------------------
	//TASK 9 START------------------------------------------------------------------------------------------------------
	x = loadLISVector("x.mtx");
	X = x.reshaped<RowMajor>(A.rows(), A.cols());
	image = toImage(X);
	if (!saveImage("x_solution_image.png", image, width, height)){return std::cerr << "Error: Could not save image\n", 1;}
	std::cout << "X solution image saved to x_solution_image.png" << std::endl;
	//TASK 9 END--------------------------------------------------------------------------------------------------------
	//TASK 10 START-------------------------------------------------------------------------------------
	A3 = getSparseMatrixFromConv(A.rows(), A.cols(), hed2);
	std::cout << (A3.isApprox(A3.transpose()) ? "A3 is symmetric" : "A3 is NOT symmetric") << std::endl;
	//TASK 10 END---------------------------------------------------------------------------------------
	//TASK 11 START----------------------------------------------------------------------------------------------------------
	g3 = A3*v;
	G3 = g3.reshaped<RowMajor>(A.rows(), A.cols());
	image = toImage(G3);
	if (!saveImage("edge_detected_image.png", image, width, height)){return std::cerr << "Error: Could not save image\n", 1;}
	std::cout << "Edge dectected image saved to edge_detected_image.png" << std::endl;
	//TASK 11 END------------------------------------------------------------------------------------------------------------
	//TASK 12 START--------------------------------------------------------------------------
	A3M = A3;
	A3M.diagonal().array() += 4.0;
	Eigen::BiCGSTAB<SparseMatrix> solver;
	solver.setTolerance(1e-10);
	solver.compute(A3M);
	if (solver.info() != Eigen::Success){return std::cerr << "Failed to decompose A3M\n", 1;}
	y = solver.solve(w);
	if (solver.info() != Eigen::Success){return std::cerr << "Solver failed\n", 1;}
	std::cout << "Iterations: " << solver.iterations() << std::endl;
	std::cout << "Final residual: " << solver.error() << std::endl;
	//TASK 12 END----------------------------------------------------------------------------
	//TASK 13 START-------------------------------------------------------------------------------------------------------
	Y = y.reshaped<RowMajor>(A.rows(), A.cols());
	image = toImage(Y);
	if (!saveImage("y_solution_image.png", image, width, height)){return std::cerr << "Error: Could not save image\n", 1;}
	std::cout << "Y solution image saved to y_solution_image.png" << std::endl;
	//TASK 13 END---------------------------------------------------------------------------------------------------------
	return 0;
}