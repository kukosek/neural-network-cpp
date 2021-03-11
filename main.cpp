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
		unsigned int training_count = 0;
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

// Adds a layer of n ('size') neurons to the network
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

	// setup the network
	NeuralNetwork network = NeuralNetwork();

	//input layer (28*28 pixels = 784 neurons)
	network.add_layer(784);

	//hidden layers, can be tweaked
	network.add_layer(16);
	network.add_layer(16);

	//output layer ( digits 0-9 )
	network.add_layer(10);


	const int desired_number_scalar = 255;

	std::cout << "starting training" << std::endl;
	network.reset_training_batch();

	const int batch_n_images = 20; // average of how many gradients will be applied
	// for every "training batch" of images
	for (int image_batch=0; image_batch < dataset.training_images.size()/batch_n_images; image_batch++) {
		for (int image_idx=0; image_idx <  batch_n_images; image_idx++) {

			int image_result_idx = image_batch*batch_n_images + image_idx;

			// for every pixel in the image
			for (int i=0; i<dataset.training_images[image_result_idx].size(); i++) {
				double value = dataset.training_images[image_result_idx][i] / 255.0;
				// put it in the first layer's activations
				network.layers[0].activations[i] = value;
			}

			// calculate all neuron activations
			network.calculate();


			// this is the target vector we want the last layer to be
			VectorXd desired = VectorXd::Constant(10, 0.0);
			int target_number = dataset.training_labels[image_result_idx];
			desired[target_number] = desired_number_scalar;

			// calculates the curent gradient based on the target vector
			network.train_on(desired);
		}
		// applies the average gradient
		network.apply_training_batch();
	}

	std::cout << "training complete, i am going to test" << std::endl;

	int correct_tests = 0;
	int bad_tests = 0;
	// for every image in test dataset
	for (int image_idx=0; image_idx < dataset.test_labels.size(); image_idx++) {
		// for every pixel of image
		for (int i=0; i<dataset.test_images[image_idx].size(); i++) {
			double value = dataset.test_images[image_idx][i] / 255.0;
			network.layers[0].activations[i] = value;
		}

		// calculate activations of neuron
		network.calculate();


		// can be used for cost calculations
		VectorXd desired = VectorXd::Constant(10, 0.0);
		int target_number = dataset.test_labels[image_idx];
		desired[target_number] = desired_number_scalar;

		// activations of the output layer
		VectorXd result_activations = network.layers[network.layers.size()-1].activations;

		// determine the brightest neuron
		// (thats the guess of the network, whats the digit)
		int max_idx = 0;
		double max_value = 0.0;
		for (int i=0; i<result_activations.size(); i++) {
			if (result_activations[i] > max_value) {
				max_idx = i;
				max_value = result_activations[i];
			}
		}

		// increase stat counters
		if (max_idx == target_number) correct_tests++;
		else bad_tests++;

		// detailed test prints
		// const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, "\t", " ", "", "", "", "");
		// std::cout << target_number << " cost " << network.get_cost(desired) << " output: " << network.layers[network.layers.size()-1].activations.format(fmt) << std::endl;
	}

	std::cout << "Test finished. Correct " << correct_tests << "/" << correct_tests+bad_tests << std::endl;

    return 0;
}
