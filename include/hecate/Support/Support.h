
#ifndef HECATE_SUPPORT_SUPPORT
#define HECATE_SUPPORT_SUPPORT

#include <cstdint>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>
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

// Non-adjacent form (NAF) representation of an integer

inline llvm::SmallVector<int, 4> naf(int value) {
  llvm::SmallVector<int, 4> res;

  // Record the sign of the original value and compute abs
  bool sign = value < 0;
  value = std::abs(value);

  // Transform to non-adjacent form (NAF)
  for (int i = 0; value; i++) {
    int zi = (value & int(0x1)) ? 2 - (value & int(0x3)) : 0;
    value = (value - zi) >> 1;
    if (zi) {
      res.push_back((sign ? -zi : zi) * (1 << i));
    }
  }

  return res;
}

inline void setIntegerAttr(llvm::StringRef name, mlir::Value v, int64_t data) {
  unsigned argnum = 0;
  mlir::Operation *op = nullptr;
  if (auto ba = v.dyn_cast<mlir::BlockArgument>()) {
    argnum = ba.getArgNumber();
    op = ba.getOwner()->getParentOp();
  } else if (auto opr = v.dyn_cast<mlir::OpResult>()) {
    argnum = opr.getResultNumber();
    op = opr.getOwner();
  } else {
    assert(0 && "Value should be either block argument or op result");
  }
  auto builder = mlir::OpBuilder(op);
  op->setAttr(std::string(name) + std::to_string(argnum),
              builder.getI64IntegerAttr(data));
}

inline int64_t getIntegerAttr(llvm::StringRef name, mlir::Value v) {
  unsigned argnum = 0;
  mlir::Operation *op = nullptr;
  if (auto ba = v.dyn_cast<mlir::BlockArgument>()) {
    argnum = ba.getArgNumber();
    op = ba.getOwner()->getParentOp();
  } else if (auto opr = v.dyn_cast<mlir::OpResult>()) {
    argnum = opr.getResultNumber();
    op = opr.getOwner();
  } else {
    assert(0 && "Value should be either block argument or op result");
  }
  if (auto attr = op->getAttr(std::string(name) + std::to_string(argnum))) {
    return attr.dyn_cast<mlir::IntegerAttr>().getInt();
  } else {
    return -1;
  }
}

using Complex = std::complex<double>;
using Diagonal = std::vector<Complex>;
using DiagonalMatrix = std::vector<Diagonal>;
using StepMatrix = std::pair<DiagonalMatrix, int>;
void roll_vector(std::vector<Complex> &vec, int shift);

mlir::Value dvMultGen(mlir::OpBuilder builder, mlir::Value X,
                      const StepMatrix &WS);
std::vector<Complex> dvMult(const std::vector<Complex> &X, const StepMatrix &W,
                            int bs = 4);

StepMatrix ddMult(const StepMatrix &W0, const StepMatrix &W1);
// std::vector<DiagonalMatrix> generateConstant(int n);
std::vector<StepMatrix> generateConstant(int n);
StepMatrix dInverse(const StepMatrix &W, int n = 1);
std::vector<StepMatrix> mergeMatrix(const std::vector<StepMatrix> &dM,
                                    std::vector<int> interval);

} // namespace hecate

#endif
