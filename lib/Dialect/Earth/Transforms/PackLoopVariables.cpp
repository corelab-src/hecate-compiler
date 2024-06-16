
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "hecate/Support/Support.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include <cmath>
#include <numeric>

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_PACKLOOPVARIABLES
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct PackLoopVariablesPass
    : public hecate::earth::impl::PackLoopVariablesBase<PackLoopVariablesPass> {
  PackLoopVariablesPass() {}

  mlir::SmallVector<scf::ForOp, 4> scfForQueue;
  void schedulePackForOp(mlir::scf::ForOp &op) {
    auto &&bb = op.getBody();
    for (auto iter = bb->begin(); iter != bb->end(); ++iter) {
      if (auto forOp = dyn_cast<mlir::scf::ForOp>(*iter)) {
        schedulePackForOp(forOp);
      }
    }
    scfForQueue.push_back(op);
    return;
  }

  void runOnOperation() override {
    auto func = getOperation();
    OpBuilder builder(func);
    IRRewriter rewriter(builder);

    auto &&bb = func.getBody().getBlocks().front();
    for (auto iter = bb.begin(); iter != bb.end(); ++iter) {
      if (auto forOp = dyn_cast<mlir::scf::ForOp>(*iter)) {
        schedulePackForOp(forOp);
      }
    }

    for (auto fop : scfForQueue) {
      auto num_elements =
          hecate::getIntegerAttr("num_elements", fop.getResult(0));
      builder.setInsertionPoint(fop);
      SmallVector<mlir::Value, 2> inputs, regionInputs;
      SmallVector<mlir::Value, 2> yields, rets;

      auto &&beforeLoopPackOp = builder.create<hecate::earth::PackOp>(
          fop.getLoc(), fop.getInitArgs(), num_elements,
          fop.getNumRegionIterArgs());
      for (size_t i = 0; i < beforeLoopPackOp.getNumResults(); i++) {
        fop.setOperand(i + fop.getNumControlOperands(),
                       beforeLoopPackOp.getResult(i));
        fop.getRegionIterArg(i).setType(
            fop.getOperand(i + fop.getNumControlOperands()).getType());
        inputs.push_back(beforeLoopPackOp.getResult(i));
        regionInputs.push_back(fop.getRegionIterArg(i));
      }

      auto yieldOp = fop.getBody()->getTerminator();
      builder.setInsertionPoint(yieldOp);
      auto &&beforeYieldPackOp = builder.create<hecate::earth::PackOp>(
          fop.getLoc(), yieldOp->getOperands(), num_elements,
          yieldOp->getNumOperands());
      for (size_t i = 0; i < beforeYieldPackOp.getNumResults(); i++) {
        yieldOp->setOperand(i, beforeYieldPackOp.getResult(i));
        fop.getResult(i).setType(yieldOp->getOperand(i).getType());
        rets.push_back(fop.getResult(i));
        yields.push_back(yieldOp->getOperand(i));
      }

      builder.setInsertionPointToStart(fop.getBody());
      auto &&insideLoopUnPackOp = builder.create<hecate::earth::UnPackOp>(
          fop.getLoc(), regionInputs, num_elements, fop.getNumResults());
      for (size_t i = 0; i < fop.getNumRegionIterArgs(); i++) {
        fop.getRegionIterArg(i).replaceAllUsesExcept(
            insideLoopUnPackOp.getResult(i), insideLoopUnPackOp);
      }

      builder.setInsertionPointAfter(fop);
      auto &&afterLoopUnPackOp = builder.create<hecate::earth::UnPackOp>(
          fop.getLoc(), rets, num_elements, fop.getNumRegionIterArgs());
      for (size_t i = 0; i < fop.getNumResults(); i++) {
        fop.getResult(i).replaceAllUsesExcept(afterLoopUnPackOp.getResult(i),
                                              afterLoopUnPackOp);
      }

      // replace All uses with packed one [SHOULD BE FIXED TO MULTIPLE
      // CIPHERTEXTS]
      for (size_t i = fop.getNumRegionIterArgs() - 1; i > 0; i--) {
        fop.getRegionIterArg(i).replaceAllUsesWith(fop.getRegionIterArg(0));
        fop.getBody()->eraseArgument(i + 1);
      }
      for (size_t i = fop.getNumResults() - 1; i > 0; i--) {
        fop.getResult(i).replaceAllUsesWith(fop.getResult(0));
        fop.getBody()->getTerminator()->eraseOperand(i);
      }
      // Replace the SCF ForOp to NewForOp
      builder.setInsertionPointAfter(fop);
      auto &&packedLoop = builder.create<mlir::scf::ForOp>(
          fop.getLoc(), fop.getLowerBound(), fop.getUpperBound(), fop.getStep(),
          inputs);
      fop.getResult(0).replaceAllUsesWith(packedLoop.getResult(0));
      fop.getBody()->moveBefore(packedLoop.getBody());
      packedLoop.getRegion().back().erase();
      fop.erase();

      // mark packing forOp
      packedLoop->setAttr("is_packed", builder.getBoolAttr(true));
    }
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
