#include <algorithm>
using std::find;
using std::shuffle;

#include <cmath>
using std::sqrt;

#include <iostream>
using std::cout;
using std::endl;

#include <fstream>
using std::ifstream;

#include <random>
using std::mt19937;
using std::random_device;

#include <string>
using std::getline;

#include "dataset.hxx"
#include "sample.hxx"

void Dataset::read_samples_from_file(const string &filename) {
    ifstream input_stream(filename);

    string line;
    while (getline(input_stream, line)) {
        Sample *sample = new Sample(line);
        samples.push_back(sample);
        // cout << sample << endl;

        if (find(labels.begin(), labels.end(), sample->label) == labels.end()) {
            labels.push_back(sample->label);
        }
    }

    cout << "dataset labels:";
    for (string label : labels) {
        cout << " " << label;
    }
    cout << endl;
}

Dataset::Dataset(const string &filename, int32_t seed) {
    random_number_generator = mt19937(seed);

    read_samples_from_file(filename);
}

Dataset::Dataset(const string &filename) {
    random_device rd;
    random_number_generator = mt19937(rd());

    read_samples_from_file(filename);
}


Dataset::~Dataset() {
    /**********
     * TODO: Implement the destructor
     **********/

    /********** ENDTODO **********/
}

int32_t Dataset::sample_size() {
    if (samples.size() == 0) {
        return -1;
    } else {
        //all samples should be the same size
        return samples[0]->size();
    }
}

void Dataset::get_statistics(vector<double> &sample_means, vector<double> &sample_stds) {
    /**********
     * TODO: Resize the input sample means and calculate
     * the mean for each sample value.
     ***********/

    /********** ENDTODO **********/
}

void Dataset::normalize(const vector<double> &sample_means, vector<double> &sample_stds) {
    /**********
     * TODO: Normalize the samples using z-score normalization
     * with the provided means and standard deviations.
     ***********/

    //cout << "normalized samples: " << endl;

    /********** ENDTODO **********/
}

void Dataset::shuffle() {
    // need to specify the namespace to avoid the compiler being confused
    // with the class's shuffle method.
    std::shuffle(samples.begin(), samples.end(), random_number_generator);
}


Sample* Dataset::operator[](uint32_t index) {
    return samples.at(index);
}
