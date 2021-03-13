#include <iostream>
#include <time.h>
#include <signal.h>
#include <vector>
#include "mnist/mnist_reader.hpp"
#include <Eigen/Dense>
#include "NeuralNetwork.hpp"
#include <omp.h>

using Eigen::VectorXd;

NeuralNetwork network;

const char * datafolder_name = "training_data";

void signal_handler(int s) {
	std::cout << std::endl << "Training ended. Saving." << std::endl;
	network.load_best_setup();
	network.save_to_files(datafolder_name);
	std::cout << "Bye" << std::endl;
	exit(1);
}
int main(int argc, char* argv[]) {
	struct sigaction sigIntHandler;

	sigIntHandler.sa_handler = signal_handler;
	sigemptyset(&sigIntHandler.sa_mask);
	sigIntHandler.sa_flags = 0;

	sigaction(SIGINT, &sigIntHandler, NULL);

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
	network = NeuralNetwork();

	//input layer (28*28 pixels = 784 neurons)
	network.add_layer(784);

	//hidden layers, can be tweaked
	network.add_layer(16);
	network.add_layer(16);

	//output layer ( digits 0-9 )
	network.add_layer(10);

	bool skip_next_training = network.load_from_files(datafolder_name);

	std::cout << "Starting training iterations" << std::endl << std::endl;

	double min_cost = 0.0;
	double max_success_rate = 0.0;
	unsigned int training_attempts = 0;
	while (true) {
		training_attempts++;
		const int desired_number_scalar = 255;

		if (!skip_next_training) {
			network.reset_training_batch();

			const int batch_n_images = 20; // average of how many gradients will be applied
			// for every "training batch" of images
			const int train_images_total = dataset.training_images.size();
			const int n_of_batches = train_images_total/batch_n_images;
			for (int image_batch=0; image_batch < n_of_batches; image_batch++) {
				std::cout << (int)(((float)image_batch/n_of_batches)*100) <<
					"%" << "\t\r" << std::flush;

				//#pragma omp parallel for
				for (int image_idx=0; image_idx < batch_n_images; image_idx++) {
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
				};
				// applies the average gradient
				network.apply_training_batch();
			}
		}

		int correct_tests = 0;
		int bad_tests = 0;
		double cost_sum = 0.0;
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

			cost_sum += network.get_cost(desired);

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

		unsigned int n_of_tests = (correct_tests+bad_tests);
		double network_cost = cost_sum / n_of_tests;
		if (network_cost < min_cost || training_attempts == 1) {
			min_cost = network_cost;
		}
		double network_sucess_rate = (double)correct_tests / n_of_tests;
		if (network_sucess_rate > max_success_rate) {
			max_success_rate = network_sucess_rate;
			network.save_trainresults_as_best();
		}
		std::cout << "    of train attempt #" << training_attempts+1 <<
			"; Last correct: " << correct_tests << "/" << n_of_tests
			<< "; Highest %: " << (int)(max_success_rate * 100)
			<< ", min cost: " << min_cost
			<< "\t\r" << std::flush;

		if (skip_next_training) {
			skip_next_training = false;
		}else{
			//network.randomize();
		}
	}

    return 0;
}
