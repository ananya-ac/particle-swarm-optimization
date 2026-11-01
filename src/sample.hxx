#ifndef SAMPLE_HXX
#define SAMPLE_HXX

#include <iostream>
using std::ostream;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include <cstdint>

class Sample {
public:
  // The label for the sample line (the class).
  string label;

  // The values for the sample (the parameters).
  vector<double> values;

  /**
   * Constructs a sample object by parsing out the values in a CSV
   * line.
   *
   * \param line is a line from a CSV data file. the first value is
   *      the label for the line, and the other values are its parameters.
   */
  Sample(const string &line);

  /**
   * Destructor for the sample class.
   */
  ~Sample();

  /**
   * Returns the length of the values vector.
   */
  uint32_t size() const;

  /**
   * Overloading the [] operator for the sample class so we can more
   * easily (and safely) access its values.
   *
   * \param index is the index of the value to return.
   */
  double &operator[](uint32_t index);

  /**
   * Overloading the [] operator for the sample class so we can more
   * easily (and safely) access its values. This allows us to use this
   * on const instances of samples (e.g., in the << operator).
   *
   * \param index is the index of the value to return.
   */
  const double &operator[](uint32_t index) const;

  /**
   * Overloads the << operator so we can nicely print out the sample class
   * to cout, cerr or any stream.
   */
  friend ostream &operator<<(ostream &stream, const Sample &sample);

  /**
   * Overloads the << operator so we can nicely print out the sample class
   * to cout, cerr or any stream. This one will accept a pointer instead of
   * a non-pointer version of the class.
   */
  friend ostream &operator<<(ostream &stream, const Sample *sample);
};

#endif
