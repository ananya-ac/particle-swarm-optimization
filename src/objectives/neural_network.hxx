#pragma once

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "objective_function.hxx"
#include "sphere.hxx"

class NeuralNetwork : public ObjectiveFunction {
    private:
        int32_t n_layers;
        int32_t layer_size;

        vector<string> labels;

        vector<vector<double>> sample_values;
        vector<string> sample_labels;

        vector<vector<double>> nodes;

    public:

        /**
         * Constructs a NeuralNetwork objective function object.
         *
         * \param min_parameter_value is the minimum possible parameter value for the search
         * \param max_parameter_value is the maximum possible parameter value for the search
         * \param dataset_filename is the filename of the dataset to train the neural network on
         * \param n_layers is the number of hidden layers in the neural network
         */
        NeuralNetwork(double min_parameter_value, double max_parameter_value, string dataset_filename, int32_t n_layers);

        /**
         * Computes the sphere objective function.
         *
         * \param parameters are the parameters to evaluate.
         *
         * \return the fitness value of the sphere function given the
         * parameters.
         */
        virtual double evaluate(const vector<double> &parameters);
};

