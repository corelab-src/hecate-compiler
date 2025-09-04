
#ifndef HECATE_ANALYSIS_FUNCIONANALYSIS
#define HECATE_ANALYSIS_FUNCIONANALYSIS

#include "hecate/Dialect/Earth/Analysis/ScaleManagementUnit.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/AnalysisManager.h"
#include "llvm/ADT/SmallSet.h"
#include <set>

namespace hecate {

using KeyT = mlir::Operation *;
struct FunctionTraits {
public:
  FunctionTraits(mlir::Operation *op);
  void build();
  bool isLevelConsumed() const { return _isLevelConsumed; }
  bool isShapeDependent() const { return _isShapeDependent; }
  std::string getName() const { return _name; }

private:
  void containsMulOp();
  mlir::Operation *_op; // function op
  std::string _name;
  bool _isLevelConsumed;
  bool _isShapeDependent;
};

struct FunctionAnalysis {
  FunctionAnalysis(mlir::Operation *op);
  FunctionTraits *getFunctionTraits(mlir::func::FuncOp func);

private:
  mlir::Operation *_op; // module op
  std::map<KeyT, FunctionTraits *> _funcTraits;
};
} // namespace hecate

#endif
