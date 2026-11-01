#include <cmath>

#include <vector>
using std::vector;

#include "ackley.hxx"

Ackley::Ackley(int n_parameters, double min_parameter_value, double max_parameter_value) : ObjectiveFunction(n_parameters, min_parameter_value, max_parameter_value) {
}

double Ackley::evaluate(const vector<double> &parameters) {
    uint32_t i;
    double sum1, sum2;

    sum1 = 0.0;
    for (i = 0; i < parameters.size(); i++) {
        sum1 += parameters[i] * parameters[i];
    }
    sum1 /= parameters.size();
    sum1 = -0.2 * sqrt(sum1);

    sum2 = 0.0;
    for (i = 0; i < parameters.size(); i++) {
        sum2 += cos(2 * M_PI * parameters[i]);
    }
    sum2 /= parameters.size();

    return 20 + M_E - (20 * exp(sum1)) - exp(sum2);
}
