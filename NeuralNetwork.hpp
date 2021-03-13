#include <Eigen/Dense>
#include <iostream>
#include <vector>

using Eigen::VectorXd;
using Eigen::MatrixXd;

struct Layer {
	VectorXd z;
	VectorXd activations;

	MatrixXd connection_weights;
	MatrixXd weight_gradients_sum;
	MatrixXd weights_best_found;

	VectorXd biases;
	VectorXd bias_gradients_sum;
	VectorXd biases_best_found;
};

class NeuralNetwork {
	public:
		void add_layer(unsigned int size);
		std::vector<Layer> layers;
		void calculate();
		double get_cost(VectorXd target_outcome);
		void train_on(VectorXd target_outcome);
		void apply_training_batch();
		void add_to_sum_from_network(NeuralNetwork & src_network);
		void reset_training_batch();
		void randomize();
		void save_trainresults_as_best();
		void load_best_setup();
		void save_to_files(std::string foldername);
		bool load_from_files(std::string foldername);
		double training_rate = 0.00005;
		NeuralNetwork() = default;
		//copy constructor - probably doesnt work
		NeuralNetwork(const NeuralNetwork &src_network) {
			layers = src_network.layers;
			for (int layer_idx=0; layer_idx<layers.size(); layer_idx++) {
				layers[layer_idx].activations = src_network.layers[layer_idx].activations;
				layers[layer_idx].weight_gradients_sum = src_network.layers[layer_idx].weight_gradients_sum;
				layers[layer_idx].bias_gradients_sum = src_network.layers[layer_idx].bias_gradients_sum;
				layers[layer_idx].z = src_network.layers[layer_idx].z;

			}
		}
	private:
		unsigned int training_count = 0;
		void randomize_layer(int layer_idx);
};
