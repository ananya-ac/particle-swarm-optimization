#include "neural_network.hxx"

#include <iostream>
using std::cout;
using std::endl;
using std::ostream;

#include <fstream>
using std::ifstream;

#include <sstream>
using std::istringstream;

#include <string>
using std::getline;
using std::stod;
using std::string;

#include <algorithm>
#include <cmath>
using std::exp;

#include <vector>
using std::vector;

double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

NeuralNetwork::NeuralNetwork(double min_parameter_value,
                             double max_parameter_value,
                             string dataset_filename, int32_t n_layers)
    : ObjectiveFunction(min_parameter_value, max_parameter_value) {
  this->min_parameter_value = min_parameter_value;
  this->max_parameter_value = max_parameter_value;
  this->n_layers = n_layers;

  cout << "reading dataset: '" << dataset_filename << "'" << endl;
  ifstream input_stream(dataset_filename);

  string label;
  string line;
  while (getline(input_stream, line)) {
    istringstream string_stream(line);
    // cout << "line: " << line << endl;

    getline(string_stream, label, ',');
    sample_labels.push_back(label);

    // add the label to the unique list of labels if it is not there
    // yet
    if (find(labels.begin(), labels.end(), label) == labels.end()) {
      labels.push_back(label);
    }

    vector<double> values;

    string value_str;
    while (getline(string_stream, value_str, ',')) {
      values.push_back(stod(value_str));
    }
    sample_values.push_back(values);
  }

  cout << "dataset labels:";
  for (string label : labels) {
    cout << " " << label;
  }
  cout << endl;

  // all the samples should be the same length
  layer_size = sample_values[0].size();

  // normalize the sample values
  vector<double> means(layer_size, 0.0);
  vector<double> stds(layer_size, 0.0);

  for (int32_t sample = 0; sample < sample_values.size(); sample++) {
    for (int32_t i = 0; i < layer_size; i++) {
      means[i] += sample_values[sample][i];
    }
  }

  for (int32_t i = 0; i < layer_size; i++) {
    means[i] /= sample_values.size();
  }

  for (int32_t sample = 0; sample < sample_values.size(); sample++) {
    for (int32_t i = 0; i < layer_size; i++) {
      double diff = sample_values[sample][i] - means[i];
      stds[i] += diff * diff;
    }
  }

  for (int32_t i = 0; i < layer_size; i++) {
    stds[i] = sqrt(stds[i] / sample_values.size());
  }

  for (int32_t i = 0; i < layer_size; i++) {
    cout << "parameter " << i << " mean: " << means[i]
         << ", stddev: " << stds[i] << endl;
  }

  for (int32_t sample = 0; sample < sample_values.size(); sample++) {
    for (int32_t i = 0; i < layer_size; i++) {
      sample_values[sample][i] =
          (sample_values[sample][i] - means[i]) / stds[i];
    }
  }

  for (int32_t sample = 0; sample < sample_values.size(); sample++) {
    cout << "sample " << sample << " values [";
    for (int32_t i = 0; i < layer_size; i++) {
      cout << " " << sample_values[sample][i];
    }
    cout << "] -- label " << sample_labels[sample] << endl;
  }

  // calculate the number of weights in the neural network

  // layers will be fully connected, and then there will be a fully connected
  // layer to the output layer which will have N nodes where N is the number of
  // labels
  n_parameters =
      (layer_size * layer_size) * n_layers + (layer_size * labels.size());
  cout << "created a " << n_layers << " layer neural network, with "
       << layer_size << " nodes per layer, with " << n_parameters << " weights."
       << endl;

  // create the neural network nodes
  nodes.push_back(vector<double>(layer_size, 0));

  for (int32_t i = 0; i < n_layers; i++) {
    nodes.push_back(vector<double>(layer_size, 0));
  }

  nodes.push_back(vector<double>(labels.size(), 0));
}

double NeuralNetwork::evaluate(const vector<double> &parameters) {
  double total_loss = 0.0;
  double accuracy = 0.0;

  for (int32_t sample = 0; sample < sample_values.size(); sample++) {
    // reset the network by zeroing out the nodes
    // we don't need to reset the input because we are going to
    // set its values next.
    for (int32_t layer = 1; layer < nodes.size(); layer++) {
      for (int32_t node = 0; node < nodes[layer].size(); node++) {
        nodes[layer][node] = 0.0;
      }
    }

    // populate the input node values
    for (int32_t node = 0; node < layer_size; node++) {
      nodes[0][node] = sample_values[sample][node];
    }

    // do the forward pass thorugh the network's hidden layers
    int32_t weight_index = 0;
    for (int32_t layer = 0; layer < n_layers; layer++) {
      for (int32_t in = 0; in < layer_size; in++) {
        for (int32_t out = 0; out < layer_size; out++) {
          nodes[layer + 1][out] += nodes[layer][in] * parameters[weight_index];
          weight_index++;
        }
      }

      // apply the sigmoid activation function
      for (int32_t node = 0; node < layer_size; node++) {
        nodes[layer + 1][node] = sigmoid(nodes[layer + 1][node]);
      }
    }

    // do the forward pass to the output layer
    for (int32_t in = 0; in < layer_size; in++) {
      for (int32_t out = 0; out < labels.size(); out++) {
        nodes[n_layers + 1][out] +=
            nodes[n_layers][in] * parameters[weight_index];
        weight_index++;
      }
    }
    // cout << "last weight index is: " << weight_index << endl;

    for (int32_t out = 0; out < labels.size(); out++) {
      // apply the sigmoid activation function
      nodes[n_layers + 1][out] = sigmoid(nodes[n_layers + 1][out]);
    }

    // calculate the softmax of the output layer
    double sum = 0.0;
    double max_value = 0.0;
    int32_t max_index = -1;
    for (int32_t node = 0; node < labels.size(); node++) {
      double value = nodes[n_layers + 1][node];
      if (value > max_value) {
        max_value = value;
        max_index = node;
      }
    }

    for (int32_t node = 0; node < labels.size(); node++) {
      double value = exp(max_value - nodes[n_layers + 1][node]);
      sum += value;
    }

    for (int32_t node = 0; node < labels.size(); node++) {
      nodes[n_layers + 1][node] = exp(nodes[n_layers + 1][node]) / sum;
    }

    // calculate the sample loss
    int32_t expected_index =
        find(labels.begin(), labels.end(), sample_labels[sample]) -
        labels.begin();
    // cout << "label: " << sample_labels[sample] << ", index: " <<
    // expected_index << endl;

    if (expected_index == max_index) {
      accuracy += 1.0;
    }

    double sample_loss = 0.0;
    for (int32_t node = 0; node < labels.size(); node++) {
      if (node == expected_index) {
        // output node for the target label should be 1
        sample_loss += 1.0 - nodes[n_layers + 1][node];
        // cout << "    target node value: " << nodes[n_layers+1][node] << "
        // loss: " << 1.0 - nodes[n_layers+1][node] << endl;
      } else {
        // output nodes for the non target label should be 0
        sample_loss += nodes[n_layers + 1][node];
        // cout << "non target node value: " << nodes[n_layers+1][node] << "
        // loss: " << nodes[n_layers+1][node] << endl;
      }
    }
    total_loss += sample_loss;
  }

  accuracy /= sample_values.size();
  // cout << "accuracy: " << accuracy << endl;

  // return total_loss;
  return -accuracy;
}
