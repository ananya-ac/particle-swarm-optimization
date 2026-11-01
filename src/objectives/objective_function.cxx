
#include <cstdint>
#include <vector>
using std::vector;

#include "objective_function.hxx"

ObjectiveFunction::ObjectiveFunction(double _min_parameter_value, double _max_parameter_value) : min_parameter_value(_min_parameter_value), max_parameter_value(_max_parameter_value) {
}

ObjectiveFunction::ObjectiveFunction(int _n_parameters, double _min_parameter_value, double _max_parameter_value) : n_parameters(_n_parameters), min_parameter_value(_min_parameter_value), max_parameter_value(_max_parameter_value) {
}

int32_t ObjectiveFunction::get_n_parameters() {
    return n_parameters;
}

vector<double> ObjectiveFunction::get_min_bounds() {
    return vector<double>(n_parameters, min_parameter_value);
}

vector<double> ObjectiveFunction::get_max_bounds() {
    return vector<double>(n_parameters, max_parameter_value);
}
