#ifndef DATASET_HXX
#define DATASET_HXX

#include <random>
using std::mt19937;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "sample.hxx"


class Dataset {
    public:
        /**
         * Make a Mersenne Twister random number generator for the shuffle
         * method.
         */
        mt19937 random_number_generator;

        /**
         * A list of each of the labels found in the dataset.
         */
        vector<string> labels;

        /**
         * Used to hold each sample in a file.
         */
        vector<Sample*> samples;

        /**
         * A helper function used by the constructors to read the
         * samples from the provided dataset file.
         *
         * \param filename Is the path to an an input comma separated
         *      value (CSV) file where each line in it is used to populate a 
         *      vector of pointers to sample objects. The first value in the
         *      line is the class of the sample.
         */
        void read_samples_from_file(const string &filename);

        /**
         * Constructor for the dataset class.
         *
         * \param filename Is the path to an an input comma separated
         *      value (CSV) file where each line in it is used to populate a 
         *      vector of pointers to sample objects. The first value in the
         *      line is the class of the sample.
         *
         * \param seed is the seed value for the random number
         *      generator. If no seed is provided, a random seed will be provided
         *      for the random number generator used to shuffle the dataset.
         */
        Dataset(const string &filename);
        Dataset(const string &filename, int32_t seed);

        /**
         * Destructor for dataset objects. Will call delete on each sample
         * in the samples vector to clean up memory used by the object.
         */
        ~Dataset();

        /**
         * \return the size (number of values) in each sample. returns -1 if
         * there are no samples in the dataset.
         */
        int32_t sample_size();

        /**
         * Gets the mean and standard deviation of each value (parameter) for
         * all the samples in the dataset. The length of each vector will be
         * resized to the number of parameters in the dataset's samples.
         *
         * \param sample_means will be set to the mean value of each value
         * \param sample_stds will be set to the standard deviation of each value
         */
        void get_statistics(vector<double> &sample_means, vector<double> &sample_stds);

        /*
         * Z-score normalizes all the samples in the dataset using the
         * provided means and standard deviations.  Note these may come
         * from a different dataset (test data should be noramlized with
         * the statistics from the training data).
         *
         * \param sample_means are the mean values to use for z-score normalization
         * \param sample_stds are the standard deviations to use for z-score normalization
         */
        void normalize(const vector<double> &sample_means, vector<double> &sample_stds);


        /**
         * Shuffles the samples vector (used between epochs for various ML
         * algorithms).
         */
        void shuffle();

        /**
         * Overloads the [] operator to more easily and safely get
         * samples from the sample vector.
         *
         * \param index is the index of the sample to get
         */
        Sample* operator[](uint32_t index);
};

#endif
