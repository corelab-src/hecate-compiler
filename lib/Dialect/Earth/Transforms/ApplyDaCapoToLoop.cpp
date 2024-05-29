
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
#define GEN_PASS_DEF_APPLYDACAPOTOLOOP
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct ApplyDaCapoToLoopPass
    : public hecate::earth::impl::ApplyDaCapoToLoopBase<ApplyDaCapoToLoopPass> {
  ApplyDaCapoToLoopPass() {}

  mlir::SmallVector<scf::ForOp, 4> scfForQueue;
  void scheduleDaCapoForOp(mlir::scf::ForOp &op) {
    auto &&bb = op.getBody();
    for (auto iter = bb->begin(); iter != bb->end(); ++iter) {
      if (auto forOp = dyn_cast<mlir::scf::ForOp>(*iter)) {
        scheduleDaCapoForOp(forOp);
      }
    }
    scfForQueue.push_back(op);
    return;
  }

  void runOnOperation() override {

    auto func = getOperation();
    /* auto mod = func->getParentOfType<mlir::ModuleOp>(); */
    auto mod = mlir::ModuleOp::create(func.getLoc());

    mlir::OpBuilder builder(func);
    mlir::IRRewriter rewriter(builder);

    PassManager pm(mod.getContext());
    auto rescalingFactor = hecate::earth::EarthDialect::rescalingFactor;
    pm.addNestedPass<func::FuncOp>(
        hecate::earth::createBypassDetection({rescalingFactor, 0.5}));
    pm.addNestedPass<func::FuncOp>(
        hecate::earth::createCandidateSelection({rescalingFactor, 10}));
    pm.addNestedPass<func::FuncOp>(
        hecate::earth::createDaCapoPlanner({rescalingFactor, 10}));
    pm.addNestedPass<func::FuncOp>(hecate::earth::createBootstrapPlacement());

    auto &&bb = func.getBody().getBlocks().front();
    for (auto iter = bb.begin(); iter != bb.end(); ++iter) {
      if (auto forOp = dyn_cast<mlir::scf::ForOp>(*iter)) {
        scheduleDaCapoForOp(forOp);
      }
    }
    for (auto forOp : scfForQueue) {
      /* scf::ForOp result; */
      /* mlir::scf::populateSCFForLoopCanonicalizationPatterns(pattern); */
      SmallVector<mlir::Type> arg_types;
      for (auto iterArg : forOp.getRegionIterArgs())
        arg_types.push_back(iterArg.getType());

      auto yieldOp = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      auto funcType =
          builder.getFunctionType(arg_types, yieldOp.getOperandTypes());
      auto funcOp =
          mlir::func::FuncOp::create(forOp.getLoc(), "scfToFunc", funcType);
      auto entryBlock = funcOp.addEntryBlock();
      /* builder.setInsertionPointToStart(entryBlock); */
      /* forOp.getBody()->moveBefore(entryBlock); */
      /* funcOp.dump(); */
      /* entryBlock->moveBefore(forOp.getBody()); */
      /* func.dump(); */
      /* mod.push_back(funcOp); */

      auto dup = func.clone();

      scf::ForOp result;
      LogicalResult status =
          scf::peelForLoopFirstIteration(rewriter, forOp, result);
      /* mlir::SmallVector<mlir::Value, 4> rets; */
      /* rets = yieldOp.getOperands(); */
      auto ter = dup.front().getTerminator();
      ter->erase();
      builder.setInsertionPointToEnd(&dup.front());
      forOp.getBody()->moveBefore(forOp.getBody());
      builder.create<func::ReturnOp>(dup.getLoc(), yieldOp.getOperands());
      dup.dump();

      /* forOp.getBody() */
      /* if (failed(pm.run(mod))) { */
      /*   llvm::errs() << "Apply DaCapo failed" << '\n'; */
      /* } */
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<arith::ArithDialect>();
  }
};
} // namespace
