#pragma once
#include <cstdint>
#include <vector>
using std::vector;

class ObjectiveFunction {
protected:
  // the number of parameters
  int n_parameters;

  // the minimum parameter value
  double min_parameter_value;

  // the maximum parameter_value
  double max_parameter_value;

public:
  /**
   * Constructs an objective function with the given min/max parameter values.
   * This is used by the neural network objective function which calculates
   * the number of parameters based on the dataset.
   *
   * \param min_parameter_value is the minimum value a parameter can take
   * \param max_parameter_value is the maximum value a parameter can take
   */
  ObjectiveFunction(double _min_parameter_value, double _max_parameter_value);

  /**
   * Constructs an objective function with the given number
   * of parameters and min/max parameter values.
   *
   * \param n_parameters is the number of parameters the objective function has
   * \param min_parameter_value is the minimum value a parameter can take
   * \param max_parameter_value is the maximum value a parameter can take
   */
  ObjectiveFunction(int _n_parameters, double _min_parameter_value,
                    double _max_parameter_value);

  /**
   * Evaluates the objective function on a given set of parameters
   *
   * \param parameters are the parameters to evaluate the objective function
   * on.
   *
   * \return the fitness of the objective function given the parameters.
   */
  virtual double evaluate(const vector<double> &parameters) = 0;

  /**
   * \return the number of parameters for this objecive function.
   */
  int32_t get_n_parameters();

  /**
   * \return a vector of the minimum possible parameter values for the
   * objective function
   */
  vector<double> get_min_bounds();

  /**
   * \return a vector of the maximum possible parameter values for the
   * objective function
   */
  vector<double> get_max_bounds();
};
