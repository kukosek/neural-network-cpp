#include <Eigen/Dense>
#include <vector>

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
