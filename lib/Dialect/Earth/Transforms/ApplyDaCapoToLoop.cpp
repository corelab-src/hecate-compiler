
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
  ApplyDaCapoToLoopPass(hecate::earth::ApplyDaCapoToLoopOptions ops) {
    this->waterline = ops.waterline;
    this->output_val = ops.output_val;
  }

  mlir::SmallVector<scf::ForOp, 4> scfForQueue;

  // reverse-dfs scheduling
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
    auto &ca = getAnalysis<hecate::CandidateAnalysis>();

    mlir::OpBuilder builder(func);
    mlir::IRRewriter rewriter(builder);

    auto mod = mlir::ModuleOp::create(func.getLoc());
    PassManager pm_dacapo(mod.getContext());
    auto rescalingFactor = hecate::earth::EarthDialect::rescalingFactor;
    pm_dacapo.addNestedPass<func::FuncOp>(
        hecate::earth::createCodeSegmentation());
    pm_dacapo.addNestedPass<func::FuncOp>(
        hecate::earth::createPromoteLoopBody());
    /* pm_dacapo.addNestedPass<func::FuncOp>( */
    /*     hecate::earth::createProactiveRescaling({waterline, output_val})); */
    pm_dacapo.addNestedPass<func::FuncOp>(
        hecate::earth::createBypassDetection({waterline, 0.5}));
    pm_dacapo.addNestedPass<func::FuncOp>(
        hecate::earth::createCandidateSelection({rescalingFactor, output_val}));
    pm_dacapo.addNestedPass<func::FuncOp>(
        hecate::earth::createDaCapoPlanner({rescalingFactor, output_val}));
    /* pm_dacapo.addNestedPass<func::FuncOp>( */
    /*     hecate::earth::createBootstrapPlacement()); */

    PassManager pm_segment(mod.getContext());
    pm_segment.addNestedPass<func::FuncOp>(
        hecate::earth::createCodeSegmentation());

    auto &&bb = func.getBody().getBlocks().front();
    for (auto iter = bb.begin(); iter != bb.end(); ++iter) {
      if (auto forOp = dyn_cast<mlir::scf::ForOp>(*iter)) {
        scheduleDaCapoForOp(forOp);
      }
    }

    for (auto forOp : scfForQueue) {
      auto &&from = hecate::getIntegerAttr("opid", forOp.getResult(0));
      SmallVector<mlir::Type, 4> inputTy;
      /* hecate::earth::refineInputValues(dup, builder, inputTy, waterline, */
      /*                                  output_val); */
      /* func.dump(); */
      for (auto argval : func.getArguments()) {
        if (auto input_type = dyn_cast<hecate::earth::HEScaleTypeInterface>(
                argval.getType())) {
          auto tp = mlir::RankedTensorType::get(
              llvm::SmallVector<int64_t, 1>{1},
              builder.getType<hecate::earth::CipherType>(waterline, 0));
          inputTy.push_back(tp);
          continue;
        }
        inputTy.push_back(argval.getType());
      }
      for (auto argval : forOp.getRegionIterArgs()) {
        auto tp = mlir::RankedTensorType::get(
            llvm::SmallVector<int64_t, 1>{1},
            builder.getType<hecate::earth::CipherType>(rescalingFactor, 0));
        inputTy.push_back(tp);
      }
      /* for (auto tt : inputTy) */
      /*   llvm::errs() << tt << '\n'; */

      SmallVector<int64_t, 1> yieldIds;
      for (auto ydval : forOp.getResults()) {
        yieldIds.push_back(hecate::getIntegerAttr("opid", ydval));
      }
      /* llvm::errs() << "Yielded Opids" << '\n'; */
      /* for (auto tt : yieldIds) */
      /*   llvm::errs() << tt << '\n'; */

      auto dup = func.clone();
      /* dup.walk([&](scf::ForOp fop) { */
      /*   for (size_t i = 0; i < fop.getNumRegionIterArgs(); i++) { */
      /*     auto initArg = fop.getOperand(i + fop.getNumControlOperands()); */
      /*     fop.getRegionIterArg(i).replaceAllUsesWith(initArg); */
      /*   } */
      /*   auto &&loopOp =
       * dyn_cast<mlir::LoopLikeOpInterface>(fop.getOperation()); */
      /*   auto &&bb = fop.getBody(); */
      /*   for (auto iter = bb->begin(); iter != bb->end(); iter = bb->begin())
       * { */
      /*     mlir::Operation *op = &*iter; */
      /*     if (auto yop = dyn_cast<mlir::scf::YieldOp>(op)) { */
      /*       fop->getResults().replaceAllUsesWith(yop.getOperands()); */
      /*       return; */
      /*     } */
      /*     loopOp.moveOutOfLoop(op); */
      /*   } */
      /* }); */

      dup->setAttr("cutted_edge",
                   builder.getDenseI64ArrayAttr({from - 1, yieldIds.back()}));
      dup->setAttr("segment_input",
                   builder.getDenseI64ArrayAttr(
                       ca.getValueInfo(from - 1)->getLiveOuts()));
      /* llvm::errs() << from << " loop input: " */
      /*              << ca.getValueInfo(from)->getLiveOuts().size() << '\n'; */
      /* for (auto tt : ca.getValueInfo(from + 1)->getLiveOuts()) */
      /*   llvm::errs() << tt << '\n'; */
      dup->setAttr("segment_inputType", builder.getTypeArrayAttr(inputTy));
      dup->setAttr("segment_return", builder.getDenseI64ArrayAttr(yieldIds));
      dup->setAttr("is_mid_segment", builder.getBoolAttr(true));
      mod.push_back(dup);
      if (failed(pm_dacapo.run(mod))) {
        llvm::errs() << "Apply DaCapo failed" << '\n';
      }

      // After scf loop body to segment
      dup.dump();
    }
    // After completed segmentation
    /* if (failed(pm_dacapo.run(mod))) { */
    /*   llvm::errs() << "Apply DaCapo failed" << '\n'; */
    /* } */

    /* dup.erase(); */

    /* for (auto forOp : scfForQueue) { */
    /*   /1* scf::ForOp result; *1/ */
    /*   /1* mlir::scf::populateSCFForLoopCanonicalizationPatterns(pattern); *1/
     */
    /*   SmallVector<mlir::Type> arg_types; */
    /*   for (auto iterArg : forOp.getRegionIterArgs()) */
    /*     arg_types.push_back(iterArg.getType()); */

    /*   auto yieldOp =
     * dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator()); */
    /*   auto funcType = */
    /*       builder.getFunctionType(arg_types, yieldOp.getOperandTypes()); */
    /*   auto funcOp = */
    /*       mlir::func::FuncOp::create(forOp.getLoc(), "scfToFunc", funcType);
     */
    /*   auto entryBlock = funcOp.addEntryBlock(); */
    /*   /1* builder.setInsertionPointToStart(entryBlock); *1/ */
    /*   /1* forOp.getBody()->moveBefore(entryBlock); *1/ */
    /*   /1* funcOp.dump(); *1/ */
    /*   /1* entryBlock->moveBefore(forOp.getBody()); *1/ */
    /*   /1* func.dump(); *1/ */
    /*   /1* mod.push_back(funcOp); *1/ */

    /*   auto dup = func.clone(); */

    /*   scf::ForOp result; */
    /*   LogicalResult status = */
    /*       scf::peelForLoopFirstIteration(rewriter, forOp, result); */
    /*   /1* mlir::SmallVector<mlir::Value, 4> rets; *1/ */
    /*   /1* rets = yieldOp.getOperands(); *1/ */
    /*   auto ter = dup.front().getTerminator(); */
    /*   ter->erase(); */
    /*   builder.setInsertionPointToEnd(&dup.front()); */
    /*   forOp.getBody()->moveBefore(forOp.getBody()); */
    /*   builder.create<func::ReturnOp>(dup.getLoc(), yieldOp.getOperands()); */
    /*   dup.dump(); */

    /*   /1* forOp.getBody() *1/ */
    /*   /1* if (failed(pm.run(mod))) { *1/ */
    /*   /1*   llvm::errs() << "Apply DaCapo failed" << '\n'; *1/ */
    /*   /1* } *1/ */
    /* } */
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<arith::ArithDialect>();
  }
};
} // namespace
