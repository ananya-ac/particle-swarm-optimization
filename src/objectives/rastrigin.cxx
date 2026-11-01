#include <cmath>

#include <vector>
using std::vector;

#include "rastrigin.hxx"

Rastrigin::Rastrigin(int n_parameters, double min_parameter_value, double max_parameter_value) : ObjectiveFunction(n_parameters, min_parameter_value, max_parameter_value) {
}

double Rastrigin::evaluate(const vector<double> &parameters) {
    uint32_t i;
    double sum;

    sum = 0.0;
    for (i = 0; i < parameters.size(); i++) {
        sum += (parameters[i] * parameters[i]) - (10 * cos(2 * M_PI * parameters[i])) + 10;
    }
    return sum;
}
