#ifndef HECATE_SUPPORT_RANGE_DEBUGGER
#define HECATE_SUPPORT_RANGE_DEBUGGER

#include "hecate/Support/HEVMHeader.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

// OpInfo class to store operation information
class OpInfo {
public:
  HEVMOperation op;
  int ssa; // operation number
  std::string opName;
  std::vector<double> result;

  double minValue;
  double maxValue;
  double avgValue;

  OpInfo(HEVMOperation op, int ssa, const std::string opName,
         const std::vector<double> result)
      : op(op), ssa(ssa), opName(opName), result(result) {

    if (!result.empty()) {
      if (op.opcode == 0) {
        minValue = result[0];
        maxValue = result[0];
        avgValue = result[0];
      } else {
        minValue = *std::min_element(result.begin(), result.end());
        maxValue = *std::max_element(result.begin(), result.end());
        // NEED TO CHECK result.size()
        avgValue =
            std::accumulate(result.begin(), result.end(), 0.0) / result.size();
      }
    }
  }
};

namespace hecate {

class RangeTracker {
private:
  std::vector<OpInfo> opLog;
  uint64_t N;
  uint64_t L;
  uint64_t slot_size;

public:
  RangeTracker(uint64_t N, uint64_t L) : N(N), L(L), slot_size(N >> 1) {
    // Initialize the opLog vector
    opLog.reserve(1000); // Reserve space for 1000 operations initially
  }
  virtual ~RangeTracker() = default;

  void logOperation(HEVMOperation &op, int ssa, const std::string opName,
                    std::vector<double> result) {
    // Create OpInfo object and add it to the log
    OpInfo opInfo(op, ssa, opName, result);
    opLog.push_back(opInfo);
  }

  void printOpInfo(const OpInfo &info) const {
    /* std::cerr << "\nOperation Info:" << std::endl; */
    /* std::cerr << "opcode: " << info.opcode << std::endl; */
    std::cerr << "SSA: %" << info.ssa << std::endl;
    std::cerr << "Operation: " << info.opName << std::endl;
    std::cerr << "Min Value: " << info.minValue
              << ", Max Value: " << info.maxValue
              << ", Avg Value: " << info.avgValue << std::endl;
    /* std::cerr << "Decrypted Result (first 3 values): "; */
    for (size_t i = 0; i < 3; i++) {
      std::cerr << info.result[i] << " ";
    }
  }
  void exportOpLog(const std::string &outputDir,
                   const std::string &filename) const {

    // Construct the file path using std::filesystem
    std::filesystem::path dirPath(outputDir);
    std::filesystem::path filePath = dirPath / filename;

    // Ensure the directory exists
    if (!dirPath.empty() && !std::filesystem::exists(dirPath)) {
      std::filesystem::create_directories(dirPath);
    }

    // Open the file
    std::ofstream outFile(filePath);
    if (!outFile.is_open()) {
      std::cerr << "Failed to open file: " << filePath << std::endl;
      return;
    }

    // Write the header line
    outFile << "SSA,OperationName,Opcode,DST,LHS,RHS,MinValue,MaxValue,"
               "AvgValue,Result\n";

    // Write data to the file
    for (const auto &info : opLog) {
      outFile << info.ssa << ',' << info.opName << ',' << info.op.opcode << ','
              << info.op.dst << ',' << info.op.lhs << ',' << info.op.rhs << ','
              << info.minValue << ',' << info.maxValue << ',' << info.avgValue
              << ',' << '\n';
    }

    outFile.close();

    // Get the absolute path of the file
    std::filesystem::path absoluteFilePath =
        std::filesystem::absolute(filePath);
    std::cerr << "Debug log exported to " << absoluteFilePath << std::endl;
  }
};

} // namespace hecate
#endif // HEVM_DEBUGGER
