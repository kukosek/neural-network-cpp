#include "NeuralNetwork.hpp"
#include "gpumatrix/GpuMatrixBase.h"
#include <iostream>
#include <iterator>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <gpumatrix/CORE>

namespace fs = std::filesystem;
using namespace std::chrono;

namespace Eigen{
	template<class Matrix>
	void write_binary(std::string filename, const Matrix& matrix){
		std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
		typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
		out.write((char*) (&rows), sizeof(typename Matrix::Index));
		out.write((char*) (&cols), sizeof(typename Matrix::Index));
		out.write((char*) matrix.data(), rows*cols*sizeof(typename Matrix::Scalar) );
		out.close();
	}
	template<class Matrix>
	void read_binary(const char* filename, Matrix& matrix){
		std::ifstream in(filename, std::ios::in | std::ios::binary);
		typename Matrix::Index rows=0, cols=0;
		in.read((char*) (&rows),sizeof(typename Matrix::Index));
		in.read((char*) (&cols),sizeof(typename Matrix::Index));
		matrix.resize(rows, cols);
		in.read( (char *) matrix.data() , rows*cols*sizeof(typename Matrix::Scalar) );
		in.close();
	}
} // Eigen::

NeuralNetwork::NeuralNetwork() {
	cublasInit();
	int row1 = rand()%1000+1;
	int col1 = rand()%1000+1;

	int row2 = rand()%1000+1;
	int col2 = rand()%1000+1;
	Eigen::MatrixXd h_A1 = Eigen::MatrixXd::Random(row1,col1);
	Eigen::MatrixXd h_A2 = Eigen::MatrixXd::Random(row2,col2);

	Eigen::VectorXd h_B1 = Eigen::VectorXd::Random(col1);
	Eigen::VectorXd h_B2 = Eigen::VectorXd::Random(row2);

	gpumatrix::Matrix<double> d_A1(h_A1), d_A2(h_A2);
	gpumatrix::Vector<double> d_B1(h_B1), d_B2(h_B2);

	gpumatrix::Vector<double> d_C1, d_C2;


	auto start = high_resolution_clock::now();
	d_C1 =  d_A1*d_B1;
	d_C2 =  d_A2.transpose()*d_B2;
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	std::cout << "On GPU: " << duration.count() << " microseconds" << std::endl;

	start = high_resolution_clock::now();
	Eigen::VectorXd h_C1 = h_A1*h_B1;
	Eigen::VectorXd h_C2 = h_A2.transpose()*h_B2;
	stop = high_resolution_clock::now();
	duration = duration_cast<microseconds>(stop - start);
	std::cout << "On CPU: " << duration.count() << " microseconds" << std::endl;

	std::cout << d_C1.rows() << "x" << d_C1.cols() << std::endl;
	cublasShutdown();

}


void NeuralNetwork::reset_training_batch() {
	for (int layer_idx = 0; layer_idx < layers.size(); layer_idx++) {
		Layer * layer = &layers[layer_idx];

		// reset matrix of summed training results
		layer->weight_gradients_sum =
			MatrixXd::Constant(
				layer->connection_weights.rows(),
				layer->connection_weights.cols(),
				0.0
			);

		// same thing but for biases
		layer->bias_gradients_sum =
			VectorXd::Constant(
					layer->biases.rows(),
					0.0
				);
		// reset training count
		training_count = 0;
	}
}

void NeuralNetwork::apply_training_batch() {
	for (int layer_idx = 0; layer_idx < layers.size(); layer_idx++) {
		Layer * layer = &layers[layer_idx];

		MatrixXd weight_avg_gradient =(layer->weight_gradients_sum / training_count);
		layer->connection_weights -=
			weight_avg_gradient * training_rate;
		layer->biases -=
			(layer->bias_gradients_sum / training_count) * training_rate;
	}

}

void NeuralNetwork::add_to_sum_from_network(NeuralNetwork & src_network) {
	if (src_network.layers.size() == layers.size()) {
		for (int layer_idx=0; layer_idx < layers.size(); layer_idx++) {
			Layer * src_layer = &src_network.layers[layer_idx];
			Layer * dest_layer = &layers[layer_idx];

			if (src_layer->bias_gradients_sum.rows() == dest_layer->bias_gradients_sum.rows()) {
				if (src_layer->weight_gradients_sum.rows() == dest_layer->weight_gradients_sum.rows()
						&& src_layer->weight_gradients_sum.cols() == dest_layer->weight_gradients_sum.cols()
				   ) {
					dest_layer->bias_gradients_sum += src_layer->bias_gradients_sum;
					dest_layer->weight_gradients_sum += src_layer->weight_gradients_sum;
				}
				else std::cout << "Assert weight cols rows failed" << std::endl;
			}
			else std::cout << "Assert bias rows failed " << std::endl;
		}
	}else{
		std::cout << "Assert src_network layers size failed";
	}
}

double map11to01(double x) {
	return (x+1) /2;
}

void NeuralNetwork::randomize_layer(int layer_idx) {
	Layer * layer = &layers[layer_idx];
	layer->biases = VectorXd::Random(layer->activations.rows()).unaryExpr(&map11to01);

	if (layer_idx > 0) {
		Layer * last_layer = &layers[layer_idx-1];
		last_layer->connection_weights =
			MatrixXd::Random(
				layer->activations.size(),
				last_layer->activations.size()
			);//.unaryExpr(&map01);

	}
}

void NeuralNetwork::randomize() {
	for (int layer_idx=0; layer_idx<layers.size(); layer_idx++) {
		randomize_layer(layer_idx);
	}
}

void NeuralNetwork::save_trainresults_as_best() {
	for (int layer_idx=0; layer_idx<layers.size(); layer_idx++) {
		Layer * layer = &layers[layer_idx];
		layer->biases_best_found = layer->biases;
		layer->weights_best_found = layer->connection_weights;
	}
}

void NeuralNetwork::load_best_setup() {
	for (int layer_idx=0; layer_idx<layers.size(); layer_idx++) {
		Layer * layer = &layers[layer_idx];
		layer->biases = layer->biases_best_found;
		layer->connection_weights = layer->weights_best_found;
	}
}


// Adds a layer of n ('size') neurons to the network
void NeuralNetwork::add_layer(unsigned int size) {
	layers.push_back(Layer());
	Layer * layer = &layers[layers.size()-1];
	layer->activations = VectorXd::Constant(size, 1.0);
	randomize_layer(layers.size()-1);
}

double NeuralNetwork::get_cost(VectorXd target_outcome) {
	VectorXd deltas = (layers[layers.size()-1].activations - target_outcome);
	return (deltas.transpose() * deltas )(0, 0);
}

double leakyReluAplpha = 0.001;
double relu(double x) {
	if (x>0) return x;
	else return x*leakyReluAplpha;
}

double derivativeOfReluOf(double x) {
	if (x>0) return 1;
	else return leakyReluAplpha;
}
void NeuralNetwork::train_on(VectorXd target_outcome) {
	Layer * last_layer = &layers[layers.size()-1];
	VectorXd a_cost_gradient = 2*(
			last_layer->activations - target_outcome);
	for (int layer_idx=layers.size()-1; layer_idx > 0; layer_idx--) {
		Layer * layer = &layers[layer_idx];
		last_layer = &layers[layer_idx-1];


		VectorXd bias_gradient = ( a_cost_gradient.array() *
			layer->z.unaryExpr(&derivativeOfReluOf).array()).matrix();
		MatrixXd weight_gradient = bias_gradient *
			last_layer->activations.transpose();


		// Add it to the sum
		layer->bias_gradients_sum += bias_gradient;
		last_layer->weight_gradients_sum += weight_gradient;

		// Calculate A cost gradient that will be used for calculations
		// of weight and bias gradients in the next layer
		MatrixXd tmp_m = (
			 layer->z.unaryExpr(&derivativeOfReluOf).array() *
			a_cost_gradient.array()
			 ).matrix();


		a_cost_gradient = last_layer->connection_weights.transpose() *
			tmp_m;
	}

	training_count++;
}


void NeuralNetwork::calculate() {
	// for every layer
	for (int layer_idx = 1; layer_idx < layers.size(); layer_idx++) {
		Layer * layer = &layers[layer_idx];
		Layer * last_layer = &layers[layer_idx-1];
		layer->z =
			(
				 last_layer->connection_weights
				 * last_layer->activations
				 + layer->biases
			);
		layer->activations = layer->z.unaryExpr(&relu);
	}
}

void NeuralNetwork::save_to_files(std::string foldername) {
	fs::remove_all(foldername);
	fs::create_directory(foldername);
	std::ofstream metadata;
	metadata.open(foldername+"/network_metadata.txt");
	if (!metadata) {
		std::cout << "... Meta file not created!";
	}
	metadata << layers.size() << std::endl;
	for (int layer_idx = 0; layer_idx < layers.size(); layer_idx++) {
		Layer * layer = &layers[layer_idx];

		if (layer_idx != 0) metadata << " ";
		metadata << layer->activations.size();

		Eigen::write_binary(
				foldername+'/'+"biases"+std::to_string(layer_idx)+".dat",
				layer->biases
			);
		Eigen::write_binary(
				foldername+'/'+"weights"+std::to_string(layer_idx)+".dat",
				layer->connection_weights
			);
	}
	metadata << std::endl;
	metadata << training_rate << std::endl;
	metadata.close();
}

bool NeuralNetwork::load_from_files(std::string foldername) {
	if (fs::exists(foldername)) {
		std::ifstream metadata;
		metadata.open(foldername+"/network_metadata.txt");
		bool error = false;
		if (metadata) {
			std::string n_of_layers_str;
			getline(metadata, n_of_layers_str);
			int n_of_layers = std::stoi(n_of_layers_str);

			std::string layers_sizes_str;
			getline(metadata, layers_sizes_str);
			std::istringstream iss(layers_sizes_str);
			std::vector<std::string> layers_sizes_strarr(
					(std::istream_iterator<std::string>(iss)),
                                 std::istream_iterator<std::string>());
			if (n_of_layers == layers_sizes_strarr.size()) {
				std::vector<int> layers_sizes;
				for (int i=0; i<n_of_layers; i++) {
					int neurons = std::stoi(layers_sizes_strarr[i]);
					if (neurons > 0) layers_sizes.push_back(neurons);
					else {error = true; break;};
				}
				if (!error) {
					layers.clear();
					for (int i=0; i<n_of_layers; i++) {
						add_layer(layers_sizes[i]);
					}
				}
			}
			std::string rate_str;
			getline(metadata, rate_str);
			double rate = ::atof(rate_str.c_str());
			if (rate != 0) {
				training_rate = rate;
			}
			metadata.close();
		}
		for (int layer_idx = 0; layer_idx < layers.size(); layer_idx++) {
			Layer * layer = &layers[layer_idx];


			MatrixXd readed_weights;
			Eigen::read_binary(
					(foldername+'/'+"weights"+std::to_string(layer_idx)+".dat").c_str(),
					readed_weights
				);
			if (readed_weights.rows() == layer->connection_weights.rows()
					&& readed_weights.cols() == layer->connection_weights.cols()
			   ) {
				VectorXd readed_biases;
				Eigen::read_binary(
						(foldername+'/'+"biases"+std::to_string(layer_idx)+".dat").c_str(),
						readed_biases
					);
				if (readed_biases.rows() == layer->biases.rows()
						&& readed_biases.cols() == layer->biases.cols()
				   ) {
					layer->biases = readed_biases;
					layer->connection_weights = readed_weights;
				} else{
					return false;
				}
			} else {
				return false;
			}
		}
		return !error;
	} else {
		return false;
	}
}
