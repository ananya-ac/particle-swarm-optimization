#include <iostream>
using std::endl;
using std::ostream;

#include <string>
using std::getline;
using std::stod;
using std::string;

#include <sstream>
using std::istringstream;

#include "sample.hxx"

Sample::Sample(const string &line) {
  istringstream ss(line);
  string value;
  getline(ss, label, ',');
  while (getline(ss, value, ',')) {
    values.push_back(stod(value));
  }
}

Sample::~Sample() {}

uint32_t Sample::size() const { return values.size(); }

double &Sample::operator[](uint32_t index) { return values.at(index); }

const double &Sample::operator[](uint32_t index) const {
  return values.at(index);
}

ostream &operator<<(ostream &stream, const Sample &sample) {
  stream << "[sample | label: " << sample.label << " | ";

  for (uint32_t i = 0; i < sample.values.size(); i++) {
    if (i > 0)
      stream << ", ";
    stream << sample[i];
  }

  stream << "]";
  return stream;
}

ostream &operator<<(ostream &stream, const Sample *sample) {
  // this will call the non-pointer version by dereferencing
  // the pointer to sample
  stream << *sample;
  return stream;
}
