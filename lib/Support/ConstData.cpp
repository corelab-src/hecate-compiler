#include <hecate/Support/Support.h>

#include <fstream>
#include <iostream>
#include <stdexcept>

namespace hecate {
// Constructor
ConstData::ConstData() {}

// Destructor
ConstData::~ConstData() { clear(); }

// Move constructor
ConstData::ConstData(ConstData &&other) noexcept { moveFrom(std::move(other)); }

// Move assignment operator
ConstData &ConstData::operator=(ConstData &&other) noexcept {
  if (this != &other) {
    clear();
    moveFrom(std::move(other));
  }
  return *this;
}

// Helper function to move resources from another instance
void ConstData::moveFrom(ConstData &&other) noexcept {
  arrays_ = std::move(other.arrays_);
}

// Load the data structure from a binary file, integrating arrays according to
// start index
void ConstData::load(const std::string &filename) {
  std::ifstream inFile(filename, std::ios::binary);
  if (!inFile) {
    assert("Error: Unable to open file for reading: " + filename);
  }

  // Read the start index
  int64_t startIndex;
  inFile.read(reinterpret_cast<char *>(&startIndex), sizeof(int64_t));
  if (!inFile) {
    assert("Error: Unable to read start index");
  }

  // Read the number of arrays
  int64_t arrayCount;
  inFile.read(reinterpret_cast<char *>(&arrayCount), sizeof(int64_t));
  if (!inFile) {
    assert("Error: Unable to read array count");
  }

  // Ensure the arrays_ vector is large enough
  size_t requiredSize = static_cast<size_t>(startIndex + arrayCount);
  if (arrays_.size() < requiredSize) {
    arrays_.resize(requiredSize);
  }

  // Read each array
  for (int64_t i = 0; i < arrayCount; ++i) {
    int64_t innerSize;
    inFile.read(reinterpret_cast<char *>(&innerSize), sizeof(int64_t));
    if (!inFile) {
      assert("Error: Unable to read inner array size");
    }

    size_t index = static_cast<size_t>(startIndex + i);

    if (innerSize < 0) {
      // Empty array; store an empty vector
      arrays_[index] = std::vector<double>();
      continue;
    }

    // Read the array data
    std::vector<double> data(static_cast<size_t>(innerSize));
    inFile.read(reinterpret_cast<char *>(data.data()),
                innerSize * sizeof(double));
    if (!inFile) {
      assert("Error: Unable to read array data");
    }

    arrays_[index] = std::move(data);
  }
}

// Save the data structure to a binary file with the given start index
void ConstData::save(const std::string &filename, size_t startIndex) {
  std::ofstream outFile(filename, std::ios::binary | std::ios::trunc);
  if (!outFile) {
    assert("Error: Unable to open file for writing: " + filename);
  }

  size_t arrayCount = arrays_.size();

  // Write the start index
  int64_t startIndex64 = static_cast<int64_t>(startIndex);
  outFile.write(reinterpret_cast<const char *>(&startIndex64), sizeof(int64_t));

  // Write the number of arrays
  int64_t arrayCount64 = static_cast<int64_t>(arrayCount);
  outFile.write(reinterpret_cast<const char *>(&arrayCount64), sizeof(int64_t));

  // Write each array
  for (size_t i = 0; i < arrays_.size(); ++i) {
    const std::vector<double> &arr = arrays_[i];
    int64_t innerSize = static_cast<int64_t>(arr.size());

    // Write the inner size
    outFile.write(reinterpret_cast<const char *>(&innerSize), sizeof(int64_t));

    // Write the array data
    outFile.write(reinterpret_cast<const char *>(arr.data()),
                  arr.size() * sizeof(double));
  }
  std::cout << filename << '\n';
}

// Overload the subscript operator to return a reference to
// std::vector<double>
std::vector<double> &ConstData::operator[](size_t index) {
  if (index >= arrays_.size()) {
    assert("Index out of range");
  }
  return arrays_[index];
}

// Overload the subscript operator for const access
const std::vector<double> &ConstData::operator[](size_t index) const {
  if (index >= arrays_.size()) {
    assert("Index out of range");
  }
  return arrays_[index];
}

// Push back an array to the data structure
void ConstData::push_back(const std::vector<double> &arr) {
  arrays_.push_back(arr);
}

// Get the number of arrays stored (size of the vector)
size_t ConstData::size() const { return arrays_.size(); }

// Clear the data
void ConstData::clear() { arrays_.clear(); }
} // namespace hecate
