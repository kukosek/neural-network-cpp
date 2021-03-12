#include "NeuralNetwork.hpp"

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
