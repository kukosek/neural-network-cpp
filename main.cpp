#include <iostream>
#include <time.h>
#include <vector>
#include "mnist/mnist_reader.hpp"
#include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;

struct Layer {
	VectorXd z;
	VectorXd activations;

	MatrixXd connection_weights;
	MatrixXd weight_gradients_sum;

	VectorXd biases;
	VectorXd bias_gradients_sum;
};

class NeuralNetwork {
	public:
		void add_layer(unsigned int size);
		std::vector<Layer> layers;
		void calculate();
		double get_cost(VectorXd target_outcome);
		void train_on(VectorXd target_outcome);
		void apply_training_batch();
		void reset_training_batch();
		double training_rate = 0.00005;
	private:
		void backprop(unsigned int layer_idx, VectorXd expected);

		unsigned int training_count = 0;

		double divide_by_training_count(double x) {
			return x/training_count;
		};
};

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

double map01(double x) {
	return (x+1) / 2;
}

void NeuralNetwork::add_layer(unsigned int size) {
	layers.push_back(Layer());
	Layer * layer = &layers[layers.size()-1];
	layer->activations = VectorXd::Constant(size, 1.0);

	layer->biases = VectorXd::Constant(size, 1.0);

	if (layers.size() > 1) {
		Layer * last_layer = &layers[layers.size()-2];
		last_layer->connection_weights =
			MatrixXd::Random(
				layer->activations.size(),
				last_layer->activations.size()
			);//.unaryExpr(&map01);

	}
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
		/*std::cout << "uspech weight gradientu" << std::endl;

		std::cout << "weight gradient :" << weight_gradient.rows() << "x" <<
			weight_gradient.cols() << std::endl;
			*/

		// Add it to the sum
		layer->bias_gradients_sum += bias_gradient;
		//std::cout << "1 pricteni " << std::endl;
		last_layer->weight_gradients_sum += weight_gradient;

		//std::cout << "uspech pricteni do sumy" << std::endl;

		// Calculate A cost gradient that will be used for calculations
		// of weight and bias gradients in the next layer
		MatrixXd tmp_m = (
			 layer->z.unaryExpr(&derivativeOfReluOf).array() *
			a_cost_gradient.array()
			 ).matrix();

		/*std::cout << "conn weights: " << last_layer->connection_weights.transpose().rows() << "x" <<
last_layer->connection_weights.transpose().cols();
		std::cout << "tmp_m: " << tmp_m.rows() << "x" << tmp_m.cols() << std::endl;
		*/


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

int main(int argc, char* argv[]) {
	//seed random number gen by time
	srand(time(NULL)); // "randomize" seed

    // MNIST_DATA_LOCATION set by MNIST cmake config std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

    // Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

	NeuralNetwork network = NeuralNetwork();
	network.add_layer(784);
	network.add_layer(16);
	network.add_layer(16);
	network.add_layer(10);


	const int desired_number_scalar = 255;

	std::cout << "starting training" << std::endl;
	network.reset_training_batch();

	const int batch_n_images = 10;
	for (int image_batch=0; image_batch < 1000; image_batch++) {
		for (int image_idx=0; image_idx <  batch_n_images; image_idx++) {

			int image_result_idx = image_batch*batch_n_images + image_idx;
			for (int i=0; i<dataset.training_images[image_result_idx].size(); i++) {
				double value = dataset.training_images[image_result_idx][i] / 255.0;
				network.layers[0].activations[i] = value;
			}

			network.calculate();


			VectorXd desired = VectorXd::Constant(10, 0.0);
			int target_number = dataset.training_labels[image_result_idx];
			desired[target_number] = desired_number_scalar;

			network.train_on(desired);
		}
		network.apply_training_batch();
	}

	std::cout << "training complete, i am going to test" << std::endl;

	int correct_tests = 0;
	int bad_tests = 0;
	for (int image_idx=0; image_idx < 1000; image_idx++) {
		for (int i=0; i<dataset.training_images[image_idx].size(); i++) {
			double value = dataset.training_images[image_idx][i] / 255.0;
			network.layers[0].activations[i] = value;
		}

		network.calculate();
		//std::cout << network.layers[1].z << std::endl;


		VectorXd desired = VectorXd::Constant(10, 0.0);
		int target_number = dataset.training_labels[image_idx];
		desired[target_number] = desired_number_scalar;

		VectorXd result_activations = network.layers[network.layers.size()-1].activations;

		int max_idx = 0;
		double max_value = 0.0;
		for (int i=0; i<result_activations.size(); i++) {
			if (result_activations[i] > max_value) {
				max_idx = i;
				max_value = result_activations[i];
			}
		}

		if (max_idx == target_number) correct_tests++;
		else bad_tests++;

		const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, "\t", " ", "", "", "", "");
		//std::cout << target_number << " cost " << network.get_cost(desired) << " output: " << network.layers[network.layers.size()-1].activations.format(fmt) << std::endl;
	}

	std::cout << "Test finished. Correct " << correct_tests << "/" << correct_tests+bad_tests << std::endl;

    return 0;
}
