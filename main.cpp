#include <iostream>
#include <vector>
#include "mnist/mnist_reader.hpp"
#include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;

struct Layer {
	VectorXd activations;

	MatrixXd connection_weights;
	MatrixXd training_connection_weightsums;

	VectorXd biases;
	VectorXd training_biases_sum;
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
		layer->training_connection_weightsums =
			MatrixXd::Constant(
				layer->connection_weights.rows(),
				layer->connection_weights.cols(),
				0.0
			);

		// same thing but for biases
		layer->training_biases_sum =
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

		layer->connection_weights =
			layer->training_connection_weightsums / training_count;
		layer->biases =
			layer->training_biases_sum / training_count;
	}

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
			);
	}
}

double NeuralNetwork::get_cost(VectorXd target_outcome) {
	if (target_outcome.rows() == layers[layers.size()-1].activations.size()) {
		return target_outcome.transpose() * layers[layers.size()-1].activations;
	}else {
		std::cout << "yo dont have same amount of cost items as last layer size" << std::endl;
		return 0;
	}
}

// Backpropagation for one layer (recursively calls
// itself on layers before)
void NeuralNetwork::backprop(unsigned int layer_idx, VectorXd expected) {
	Layer * layer = &layers[layer_idx];
	Layer * last_layer = &layers[layer_idx-1];


	VectorXd activations_error = expected - layer->activations;

	MatrixXd adjusted_weights = last_layer->connection_weights;

	VectorXd adjusted_biases = layer->biases;

	VectorXd next_layer_expected = last_layer->activations;
	// for every neuron in that layer
	for (int neuron_idx=0; neuron_idx < layer->activations.size(); neuron_idx++) {
		double delta_desired = activations_error[neuron_idx];

		// increase bias
		adjusted_biases[neuron_idx] += delta_desired;


		// increase weights coming to me in proportion to their activations

		// for every connected neuron from the left to the neuron
		for (int conn_neuron_idx=0;
				conn_neuron_idx < last_layer->activations.size();
				conn_neuron_idx++
			) {
			double conn_neuron_activation =
				last_layer->activations[conn_neuron_idx];
			// increase weight based on activation and desired increase
			adjusted_weights(neuron_idx,conn_neuron_idx) +=
				conn_neuron_activation * delta_desired;

			// increase activations in proportion to the corresponding weights
			if (layer_idx > 2) {
				double corresponding_weight = last_layer->connection_weights(neuron_idx, conn_neuron_idx);
				if (next_layer_expected[neuron_idx] > 0) {
					next_layer_expected[neuron_idx] += corresponding_weight;
				} else {
					next_layer_expected[neuron_idx] -= corresponding_weight;
				}
			}

		}

	}
	layer->training_biases_sum += adjusted_biases;
	last_layer->training_connection_weightsums += adjusted_weights;

	if (layer_idx > 2) {
		// increase activations in the layer before (recursive call of this function again)
		backprop(layer_idx-1, next_layer_expected);
	}
}

void NeuralNetwork::train_on(VectorXd target_outcome) {
	backprop(layers.size()-1, target_outcome);
	training_count++;
}

double sigmoid(double x) {
	return x / (1 + abs(x));
}

void NeuralNetwork::calculate() {
	for (int layer_idx = 1; layer_idx < layers.size(); layer_idx++) {
		Layer * layer = &layers[layer_idx];
		Layer * last_layer = &layers[layer_idx-1];
		layer->activations =
			(
				 last_layer->connection_weights
				 * last_layer->activations
				 + layer->biases
			).unaryExpr(&sigmoid);
	}
}

int main(int argc, char* argv[]) {
    // MNIST_DATA_LOCATION set by MNIST cmake config
    std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

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

	network.reset_training_batch();

	for (int image_idx=0; image_idx < /*dataset.training_images.size()*/ 1000; image_idx++) {
		for (int i=0; i<dataset.training_images[image_idx].size(); i++) {
			double value = dataset.training_images[image_idx][i] / 255.0;
			dataset.training_images[image_idx][i] = value;
		}

		network.calculate();


		VectorXd desired = VectorXd::Constant(10, 0.0);
		int target_number = dataset.training_labels[image_idx];
		desired[target_number] = 1.0;

		network.train_on(desired);
		network.apply_training_batch();

	}
	network.apply_training_batch();

	for (int image_idx=0; image_idx < /*dataset.training_images.size()*/ 100; image_idx++) {
		for (int i=0; i<dataset.training_images[image_idx].size(); i++) {
			double value = dataset.training_images[image_idx][i] / 255.0;
			dataset.training_images[image_idx][i] = value;
		}

		network.calculate();


		VectorXd desired = VectorXd::Constant(10, 0.0);
		int target_number = dataset.training_labels[image_idx];
		desired[target_number] = 1.0;

		const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, "\t", " ", "", "", "", "");
		std::cout << target_number << " outpu: " << network.layers[network.layers.size()-1].activations.format(fmt) << std::endl;
	}

    return 0;
}
