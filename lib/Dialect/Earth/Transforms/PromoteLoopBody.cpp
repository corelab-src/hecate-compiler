
#include "hecate/Dialect/Earth/Analysis/CandidateAnalysis.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Common.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "hecate/Support/Support.h"
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
#define GEN_PASS_DEF_PROMOTELOOPBODY
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct PromoteLoopBodyPass
    : public hecate::earth::impl::PromoteLoopBodyBase<PromoteLoopBodyPass> {
  PromoteLoopBodyPass() {}
  PromoteLoopBodyPass(hecate::earth::PromoteLoopBodyOptions ops) {
    this->waterline = ops.waterline;
    this->output_val = ops.output_val;
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto mod = func->getParentOfType<mlir::ModuleOp>();

    mlir::OpBuilder builder(func);
    mlir::IRRewriter rewriter(builder);

    func.walk([&](scf::ForOp fop) {
      for (size_t i = 0; i < fop.getNumRegionIterArgs(); i++) {
        auto initArg = fop.getOperand(i + fop.getNumControlOperands());
        fop.getRegionIterArg(i).replaceAllUsesWith(initArg);
      }
      auto &&loopOp = dyn_cast<mlir::LoopLikeOpInterface>(fop.getOperation());
      auto &&bb = fop.getBody();
      for (auto iter = bb->begin(); iter != bb->end(); iter = bb->begin()) {
        mlir::Operation *op = &*iter;
        if (auto yop = dyn_cast<mlir::scf::YieldOp>(op)) {
          fop->getResults().replaceAllUsesWith(yop.getOperands());
          fop->erase();
          return;
        }
        loopOp.moveOutOfLoop(op);
      }
    });
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<arith::ArithDialect>();
  }
};
} // namespace
