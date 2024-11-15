#ifndef HECATE_SUPPORT_CONSTDATA
#define HECATE_SUPPORT_CONSTDATA

#include <cstdint>
#include <string>
#include <vector>

namespace hecate {
class ConstData {
public:
  ConstData();
  ~ConstData();

  // Disable copy constructor and copy assignment
  ConstData(const ConstData &) = delete;
  ConstData &operator=(const ConstData &) = delete;

  // Allow move constructor and move assignment
  ConstData(ConstData &&) noexcept;
  ConstData &operator=(ConstData &&) noexcept;

  // Load the data structure from a binary file, integrating arrays according to
  // start index
  void load(const std::string &filename);

  // Save the data structure to a binary file with the given start index
  void save(const std::string &filename, size_t startIndex);

  // Overload the subscript operator to return a reference to
  // std::vector<double>
  std::vector<double> &operator[](size_t index);

  // Overload the subscript operator for const access
  const std::vector<double> &operator[](size_t index) const;

  // Push back an array to the data structure
  void push_back(const std::vector<double> &arr);

  // Get the number of arrays stored (size of the vector)
  size_t size() const;

  // Clear the data
  void clear();

private:
  // Vector of arrays
  std::vector<std::vector<double>> arrays_;

  // Helper function for moving
  void moveFrom(ConstData &&other) noexcept;
};
} // namespace hecate

#endif // HECATE_SUPPORT_CONSTDATA
