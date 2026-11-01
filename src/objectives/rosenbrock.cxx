#include <cmath>

#include <vector>
using std::vector;

#include "rosenbrock.hxx"

Rosenbrock::Rosenbrock(int n_parameters, double min_parameter_value, double max_parameter_value) : ObjectiveFunction(n_parameters, min_parameter_value, max_parameter_value) {
}

double Rosenbrock::evaluate(const vector<double> &parameters) {
    uint32_t i;
    double sum, tmp;

    sum = 0.0;
    for (i = 0; i < parameters.size() - 1; i++) {
        tmp = (parameters[i + 1] - (parameters[i] * parameters[i]));
        sum += (100 * tmp * tmp) + ((parameters[i] - 1) * (parameters[i] - 1));
    }

    return sum;
}
