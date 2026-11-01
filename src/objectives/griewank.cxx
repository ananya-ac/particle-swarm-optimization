#include <cmath>

#include <vector>
using std::vector;

#include "griewank.hxx"

Griewank::Griewank(int n_parameters, double min_parameter_value, double max_parameter_value) : ObjectiveFunction(n_parameters, min_parameter_value, max_parameter_value) {
}

double Griewank::evaluate(const vector<double> &parameters) {
    uint32_t i;
    double sum1, sum2;

    sum1 = 0.0;
    for (i = 0; i < parameters.size(); i++) {
        sum1 += parameters[i] * parameters[i];
    }
    sum1 /= 4000;

    sum2 = cos( parameters[0] / sqrt(1.0) );
    for (i = 1; i < parameters.size(); i++) {
        sum2 *= cos( parameters[i] / sqrt((double)i+1) );
    }

    return (sum1 - sum2 + 1);
}
