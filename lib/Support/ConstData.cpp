#include <hecate/Support/ConstData.h>

#include <cassert>
#include <cstring>
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
    assert("Error: Unable to open file for reading");
  }

  // Decompress the constant data to buffer
  std::vector<char> buffer;
  decompressData(inFile, buffer);

  size_t offset = 0;
  // Read the start index
  int64_t startIndex;
  std::memcpy(&startIndex, buffer.data() + offset, sizeof(int64_t));
  offset += sizeof(int64_t);

  // Read the number of arrays
  int64_t arrayCount;
  std::memcpy(&arrayCount, buffer.data() + offset, sizeof(int64_t));
  offset += sizeof(int64_t);

  // Ensure the arrays_ vector is large enough
  size_t requiredSize = static_cast<size_t>(startIndex + arrayCount);
  if (arrays_.size() < requiredSize) {
    arrays_.resize(requiredSize);
  }

  // Read each array
  for (int64_t i = 0; i < arrayCount; ++i) {
    int64_t innerSize;
    std::memcpy(&innerSize, buffer.data() + offset, sizeof(int64_t));
    offset += sizeof(int64_t);

    size_t index = static_cast<size_t>(startIndex + i);

    if (innerSize < 0) {
      // Empty array; store an empty vector
      arrays_[index] = std::vector<double>();
      continue;
    }

    // Read the array data
    std::vector<double> data(static_cast<size_t>(innerSize));
    std::memcpy(data.data(), buffer.data() + offset,
                innerSize * sizeof(double));
    offset += innerSize * sizeof(double);

    arrays_[index] = std::move(data);
  }
  std::cerr << "Constant Loaded From " << filename << std::endl;
}

// Save the data structure to a binary file with the given start index
void ConstData::save(const std::string &filename, size_t startIndex) {
  std::ofstream outFile(filename, std::ios::binary | std::ios::trunc);
  if (!outFile) {
    assert("Error: Unable to open file for writing");
  }
  size_t arrayCount = arrays_.size();

  // Build the buffer to serialize the data
  size_t totalSize =
      2 * sizeof(int64_t); // StartIndex(8 bytes) and arrayCount(8 bytes)
  for (auto &&vec : arrays_) {
    totalSize += sizeof(int64_t);             // innerSize (8 bytes)
    totalSize += vec.size() * sizeof(double); // vector size
  }
  std::vector<char> buffer(totalSize, 0);
  size_t offset = 0;

  // Write the start index
  int64_t startIndex64 = static_cast<int64_t>(startIndex);
  std::memcpy(buffer.data() + offset, &startIndex64, sizeof(int64_t));
  offset += sizeof(int64_t);

  // Write the number of arrays
  int64_t arrayCount64 = static_cast<int64_t>(arrayCount);
  std::memcpy(buffer.data() + offset, &arrayCount64, sizeof(int64_t));
  offset += sizeof(int64_t);

  // Write each array
  for (size_t i = 0; i < arrays_.size(); ++i) {
    const std::vector<double> &arr = arrays_[i];
    int64_t innerSize = static_cast<int64_t>(arr.size());

    // Write the inner size
    std::memcpy(buffer.data() + offset, &innerSize, sizeof(int64_t));
    offset += sizeof(int64_t);

    // Write the array data
    std::memcpy(buffer.data() + offset, arr.data(),
                arr.size() * sizeof(double));
    offset += arr.size() * sizeof(double);
  }

  // Compress the data from buffer
  compressData(outFile, buffer);

  // Write the all data with serialized form
  outFile.write(reinterpret_cast<const char *>(buffer.data()), buffer.size());
  std::cout << filename << '\n';
}

// Compress the data structure with from start index to last index
void ConstData::compressData(std::ofstream &outFile,
                             std::vector<char> &serializedBuffer) {

  // compress the constant data
  size_t srcSize = serializedBuffer.size();
  uLongf compressedSize = compressBound(srcSize * sizeof(char));
  std::vector<Bytef> compressed(compressedSize);
  int res = compress(compressed.data(), &compressedSize,
                     (Bytef *)(serializedBuffer.data()),
                     serializedBuffer.size() * sizeof(char));

  // check compression is succeed
  if (res != Z_OK) {
    std::cerr << "Compression failed: " << res << "\n";
    return;
  }

  // Store the compressed form of constant
  serializedBuffer.resize(compressedSize);
  serializedBuffer.assign(
      reinterpret_cast<char *>(compressed.data()),
      reinterpret_cast<char *>(compressed.data() + compressedSize));

  // Write decompressed buffer size
  outFile.write(reinterpret_cast<const char *>(&srcSize), sizeof(size_t));

  // Write compressed buffer size
  outFile.write(reinterpret_cast<const char *>(&compressedSize),
                sizeof(size_t));
}

// Decompress the constant data structure with all compressed data;
void ConstData::decompressData(std::ifstream &inFile,
                               std::vector<char> &result) {

  // get compressed size
  size_t decompressedSize;
  inFile.read(reinterpret_cast<char *>(&decompressedSize), sizeof(size_t));
  if (!inFile) {
    assert("Error: Unable to read decompressed size");
  }
  result.resize(decompressedSize);

  // get decompressed size
  size_t compressedSize;
  inFile.read(reinterpret_cast<char *>(&compressedSize), sizeof(size_t));
  if (!inFile) {
    assert("Error: Unable to read compressed size");
  }

  // Read the compressed data
  std::vector<char> compressedBuffer(static_cast<size_t>(compressedSize));
  inFile.read(reinterpret_cast<char *>(compressedBuffer.data()),
              compressedSize * sizeof(char));

  // Decompress the constant data
  std::vector<char> decompressed(decompressedSize);
  uncompress(reinterpret_cast<Bytef *>(result.data()), &decompressedSize,
             reinterpret_cast<const Bytef *>(compressedBuffer.data()),
             compressedBuffer.size());
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
