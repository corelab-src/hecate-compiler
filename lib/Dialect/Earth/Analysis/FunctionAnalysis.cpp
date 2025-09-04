#include "hecate/Dialect/Earth/Analysis/FunctionAnalysis.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Support/Support.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include <cstddef>
#include <optional>
#include <type_traits>

using namespace hecate;

FunctionTraits::FunctionTraits(mlir::Operation *op) : _op(op) { build(); }

void hecate::FunctionTraits::build() {
  containsMulOp();
  auto &&func = dyn_cast<mlir::func::FuncOp>(_op);
  _name = func->getName().getStringRef().str();
}

void FunctionTraits::containsMulOp() {
  _isLevelConsumed = false;
  auto &&func = llvm::dyn_cast<mlir::func::FuncOp>(_op);
  auto &&bb = func.getBody().getBlocks().front();
  for (auto iter = bb.begin(); iter != bb.end(); ++iter) {
    mlir::Operation *op = &*iter;
    if (llvm::isa<hecate::earth::MulOp>(op)) {
      _isLevelConsumed = true;
      return;
    }
  }
}

FunctionAnalysis::FunctionAnalysis(mlir::Operation *op) : _op(op) {
  auto module = llvm::dyn_cast<mlir::ModuleOp>(op);
  if (!module)
    return;
  for (mlir::func::FuncOp func : module.getOps<mlir::func::FuncOp>()) {
    if (func.isDeclaration())
      continue; // skip external declarations
    auto *key = func.getOperation();
    _funcTraits.try_emplace(key, new FunctionTraits(key));
  }
}

FunctionTraits *FunctionAnalysis::getFunctionTraits(mlir::func::FuncOp func) {
  auto *op = func.getOperation();
  auto it = _funcTraits.find(op);
  assert(it != _funcTraits.end() &&
         "FunctionTraits not found for the given function");
  return it->second;
}
