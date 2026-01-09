
#include "hecate/Dialect/Earth/Analysis/FunctionAnalysis.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include <fstream>

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_CALLEEBOOTSTRAPMGMT
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct CalleeBootstrapMgmtPass
    : public hecate::earth::impl::CalleeBootstrapMgmtBase<
          CalleeBootstrapMgmtPass> {
  CalleeBootstrapMgmtPass() {}

  void runOnOperation() override {
    auto func = getOperation();
    auto &fa = getAnalysis<hecate::FunctionAnalysis>();
    //
    // llvm::errs() << "Function Name: " << func.getName() << "\n";
    // llvm::errs() << "isLevelConsumed: " << fc.isLevelConsumed() << "\n";
    // markAnalysesPreserved<hecate::FunctionAnalysis>();

    // pm.addNestedPass<func::FuncOp>(hecate::earth::createApplyDaCapoToLoop(
    //     {waterline, output_val, true}));
    // pm.addNestedPass<func::FuncOp>(hecate::earth::createLoopUnroll());
    // pm.addPass(mlir::createSymbolDCEPass());
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
