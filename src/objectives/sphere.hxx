#pragma once

#include <vector>
using std::vector;

#include "objective_function.hxx"
#include "sphere.hxx"

class Sphere : public ObjectiveFunction {
    public:

        /**
         * Constructs a Sphere objective function object.
         *
         * \param n_parameters how many parameters the sphere objective function will
         * evaluate
         * \param min_parameter_value is the minimum possible parameter value for the search
         * \param max_parameter_value is the maximum possible parameter value for the search
         */
        Sphere(int n_parameters, double min_parameter_value, double max_parameter_value);

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

