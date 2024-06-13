
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
#include <filesystem>
#include <fstream>
#include <queue>

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_LOOPUNROLL
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct LoopUnrollPass
    : public hecate::earth::impl::LoopUnrollBase<LoopUnrollPass> {
  LoopUnrollPass() {}

  mlir::SmallVector<scf::ForOp, 4> scfForQueue;
  void scheduleUnrollForOp(mlir::scf::ForOp &op) {
    auto &&bb = op.getBody();
    for (auto iter = bb->begin(); iter != bb->end(); ++iter) {
      if (auto forOp = dyn_cast<mlir::scf::ForOp>(*iter)) {
        scheduleUnrollForOp(forOp);
      }
    }
    scfForQueue.push_back(op);
    return;
  }

  void runOnOperation() override {

    auto func = getOperation();
    auto mod = func->getParentOfType<mlir::ModuleOp>();

    mlir::OpBuilder builder(func);
    mlir::IRRewriter rewriter(builder);
    mlir::RewritePatternSet pattern(builder.getContext());

    /* PassManager pm(mod.getContext()); */
    auto &&bb = func.getBody().getBlocks().front();
    for (auto iter = bb.begin(); iter != bb.end(); ++iter) {
      if (auto forOp = dyn_cast<mlir::scf::ForOp>(*iter)) {
        scheduleUnrollForOp(forOp);
      }
    }
    for (auto forOp : scfForQueue) {
      (void)mlir::loopUnrollByFactor(forOp, 2);
      mlir::scf::populateSCFForLoopCanonicalizationPatterns(pattern);
    }
    func.dump();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<arith::ArithDialect>();
  }
};
} // namespace
