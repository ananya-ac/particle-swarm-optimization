#include "sphere.hxx"

#include <cstdint>
#include <vector>
using std::vector;

Sphere::Sphere(int n_parameters, double min_parameter_value,
               double max_parameter_value)
    : ObjectiveFunction(n_parameters, min_parameter_value,
                        max_parameter_value) {}

double Sphere::evaluate(const vector<double> &parameters) {
  uint32_t i;
  double sum;

  sum = 0.0;
  for (i = 0; i < parameters.size(); i++) {
    sum += parameters[i] * parameters[i];
  }
  return sum;
}
