
#include "hecate/Dialect/Earth/Analysis/FunctionAnalysis.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"
#include <fstream>

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_CALLERBOOTSTRAPMGMT
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct CallerBootstrapMgmtPass
    : public hecate::earth::impl::CallerBootstrapMgmtBase<
          CallerBootstrapMgmtPass> {
  CallerBootstrapMgmtPass() {}

  void runOnOperation() override {

    mlir::ModuleOp module = getOperation();
    // Function Classification Analysis
    auto &fa = getAnalysis<hecate::FunctionAnalysis>();

    // auto &sym = getAnalysis<mlir::SymbolTableCollection>();
    module.walk([&](mlir::func::FuncOp func) {
      llvm::errs() << "Function Name: " << func.getName() << "\n";
      auto traits = fa.getFunctionTraits(func);
      llvm::errs() << "isLevelConsumed: " << traits->isLevelConsumed() << "\n";
      func.walk([&](mlir::func::CallOp callop) {
        auto calleeRef = callop.getCalleeAttr(); // FlatSymbolRefAttr
        if (!calleeRef)
          return;
        if (auto callee =
                mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(
                    callop, calleeRef)) {
          llvm::errs() << "CallOp to " << callee.getName() << "\n";
          // callee.dump();
        }
      });
    });

    markAnalysesPreserved<hecate::FunctionAnalysis>();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
