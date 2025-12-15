
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <filesystem>
#include <fstream>
#include <queue>

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_LOOPROTATION
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct LoopRotationPass
    : public hecate::earth::impl::LoopRotationBase<LoopRotationPass> {
  LoopRotationPass() {}

  mlir::SmallVector<scf::ForOp, 4> scfForQueue;
  void schedulePeelForOp(mlir::scf::ForOp &op) {
    auto &&bb = op.getBody();
    for (auto iter = bb->begin(); iter != bb->end(); ++iter) {
      if (auto forOp = dyn_cast<mlir::scf::ForOp>(*iter)) {
        schedulePeelForOp(forOp);
      }
    }
    scfForQueue.push_back(op);
    return;
  }

  // Helper function to clean up ErasedType wrappers in a ForOp
  void cleanUpErasedTypeWrapper(scf::ForOp targetLoop) {
    if (!targetLoop)
      return;

    auto regionIterArgs = targetLoop.getRegionIterArgs();
    auto initArgs = targetLoop.getInitArgsMutable();
    auto yieldOp = cast<scf::YieldOp>(targetLoop.getBody()->getTerminator());

    for (unsigned i = 0; i < regionIterArgs.size(); ++i) {
      // Check if the argument is an 'Erased' type
      if (hecate::earth::getScaleType(regionIterArgs[i]).isErased()) {

        // 1. Handle Init Args (Loop Input)
        OpOperand &initOperand = initArgs[i];
        // Value wrapperArgValue = initOperand.get(); // e.g., Cast Op
        Value val = initOperand.get();

        // Unwrap the ErasedType wrapper to get the real value
        if (hecate::earth::getScaleType(val).isErased() &&
            val.getDefiningOp()) {
          val = val.getDefiningOp()->getOperand(0);
          initOperand.set(val);
        }
        // Update the block argument type inside the loop
        regionIterArgs[i].setType(val.getType());

        // 2. Handle Yield Args (Loop Output/Next Iteration)
        Value yieldVal = yieldOp.getOperand(i); // e.g., Cast Op

        if (hecate::earth::getScaleType(yieldVal).isErased() &&
            yieldVal.getDefiningOp()) {
          yieldVal = yieldVal.getDefiningOp()->getOperand(0);
          // Replace the yield operand with the real value
          yieldOp.setOperand(i, yieldVal);
        }
        targetLoop.getResult(i).setType(yieldVal.getType());
      }
    }

    // 3. Walk internal operations to update types (Bootstrap, Rotate, Negate)
    targetLoop.walk([&](hecate::earth::HEScaleOpInterface sop) {
      if (llvm::isa<hecate::earth::BootstrapOp>(sop) ||
          llvm::isa<hecate::earth::RotateOp>(sop) ||
          llvm::isa<hecate::earth::NegateOp>(sop)) {
        // Propagate operand type to result type
        sop->getResult(0).setType(sop->getOperand(0).getType());
      }
    });
  }
  void runOnOperation() override {

    auto func = getOperation();
    auto mod = func->getParentOfType<mlir::ModuleOp>();

    mlir::OpBuilder builder(func);
    mlir::IRRewriter rewriter(builder);
    mlir::RewritePatternSet patterns(builder.getContext());
    scf::ForOp::getCanonicalizationPatterns(patterns, builder.getContext());

    PassManager pm(mod.getContext());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<func::FuncOp>(hecate::earth::createPrivatizeConstant());

    auto &&bb = func.getBody().getBlocks().front();
    for (auto iter = bb.begin(); iter != bb.end(); ++iter) {
      if (auto forOp = dyn_cast<mlir::scf::ForOp>(*iter)) {
        schedulePeelForOp(forOp);
      }
    }
    if (scfForQueue.empty())
      return;

    for (auto forOp : scfForQueue) {
      scf::ForOp result;
      bool anyOfPlainArg =
          llvm::any_of(forOp.getInitArgs(), [](mlir::Value arg) {
            if (hecate::earth::getScaleType(arg).isErased())
              return true;
            return false;
          });
      if (anyOfPlainArg) {
        LogicalResult status =
            scf::peelForLoopFirstIteration(rewriter, forOp, result);
        if (failed(status))
          llvm::errs() << "failed to peel the first iteration\n";
        cleanUpErasedTypeWrapper(result);
        if (failed(applyPatternsAndFoldGreedily(result->getParentOp(),
                                                std::move(patterns)))) {
          llvm::errs() << "failed to canonicalize (flatten) the peeled loop\n";
        }
        cleanUpErasedTypeWrapper(forOp);
      }
    }
    // Update function signature if return types have changed
    auto funcType = func.getFunctionType();
    mlir::SmallVector<mlir::Type, 4> retTypes;
    for (auto ret : func.getRegion().front().getTerminator()->getOperands()) {
      retTypes.push_back(ret.getType());
    }
    func.setFunctionType(
        builder.getFunctionType(funcType.getInputs(), retTypes));

    if (failed(pm.run(mod))) {
      llvm::errs() << "loop transform failed" << '\n';
      func.dump();
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<arith::ArithDialect>();
  }
};
} // namespace
