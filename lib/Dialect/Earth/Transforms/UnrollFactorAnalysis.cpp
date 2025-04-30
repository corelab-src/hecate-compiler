
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
#include "mlir/IR/Value.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include <filesystem>
#include <fstream>
#include <queue>
#include <stack>

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_UNROLLFACTORANALYSIS
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct UnrollFactorAnalysisPass
    : public hecate::earth::impl::UnrollFactorAnalysisBase<
          UnrollFactorAnalysisPass> {
  UnrollFactorAnalysisPass() {}

  mlir::SmallVector<scf::ForOp, 4> scfForQueue;

  // start operation : {end operations, depth, accmDepth}
  DenseMap<mlir::Value,
           SmallVector<std::tuple<mlir::Operation *, uint64_t, uint64_t>, 4>>
      depthMap;
  // start operation,accmDepth
  std::queue<std::tuple<Value, uint64_t>> startVals;

  mlir::SmallVector<mlir::Value> visited;
  std::stack<mlir::Value> st;
  uint64_t max_depth = 0;

  void dfsLevelDepth(mlir::Value startVal, mlir::Operation *curOp,
                     mlir::Value curVal, uint64_t startDepth,
                     uint64_t accmDepth) {
    if (std::find(visited.begin(), visited.end(), curVal) != visited.end())
      return;
    visited.push_back(curVal);
    /* curOp->dump(); */
    uint64_t wastedDownfactor = 0;
    if (!isa<mlir::BlockArgument>(curVal)) {
      if (auto mop =
              dyn_cast<hecate::earth::ModswitchOp>(curVal.getDefiningOp())) {
        wastedDownfactor += mop.getDownFactor();
      }
    }

    if (auto btp = dyn_cast<hecate::earth::BootstrapOp>(curOp)) {
      auto depth = hecate::earth::getScaleType(btp.getOperand()).getLevel();
      startDepth += wastedDownfactor;
      auto gap = depth - startDepth;
      depthMap[startVal].push_back({curOp, gap, accmDepth + gap});
      startVals.push({btp, accmDepth + gap});
      return;
    } else if (auto pack = dyn_cast<hecate::earth::PackOp>(curOp)) {
      auto depth = hecate::earth::getScaleType(pack->getResult(0)).getLevel();
      startDepth += wastedDownfactor;
      auto gap = depth - startDepth;
      depthMap[startVal].push_back({curOp, gap, accmDepth + gap});
      for (auto &&res : pack->getResults()) {
        startVals.push({res, accmDepth + gap});
      }
      return;
    } else if (auto rop = dyn_cast<func::ReturnOp>(curOp)) {
      auto depth = hecate::earth::getScaleType(curVal).getLevel();
      startDepth += wastedDownfactor;
      auto gap = depth - startDepth;
      depthMap[startVal].push_back({curOp, gap, accmDepth + gap});
      return;
    } else if (auto unpack = dyn_cast<hecate::earth::UnPackOp>(curOp)) {
      startDepth += wastedDownfactor;
    }

    for (auto &&res : curOp->getResults()) {
      for (auto &&use : res.getUses()) {
        dfsLevelDepth(startVal, use.getOwner(), use.get(), startDepth,
                      accmDepth);
      }
    }
    return;
  }

  void findMaxDepth(mlir::Operation *curOp, mlir::Value curVal,
                    uint64_t accmDepth) {

    if (!isa<mlir::BlockArgument>(curVal) && isa<func::ReturnOp>(curOp)) {
      max_depth = max_depth < accmDepth ? accmDepth : max_depth;
      return;
    }
    for (auto &&opInfo : depthMap[curVal]) {
      mlir::Operation *destOp = std::get<0>(opInfo);
      auto &&dist = std::get<1>(opInfo);
      accmDepth += dist;
      findMaxDepth(destOp, destOp->getResult(0), accmDepth);
      accmDepth -= dist;
    }
  }

  void runOnOperation() override {

    auto func = getOperation();
    auto mod = func->getParentOfType<mlir::ModuleOp>();
    mlir::OpBuilder builder(func);
    mlir::IRRewriter rewriter(builder);
    mlir::RewritePatternSet pattern(builder.getContext());

    auto &&rf = hecate::earth::EarthDialect::rescalingFactor;
    if (func->hasAttr("be_unroll") &&
        func->getAttrOfType<mlir::BoolAttr>("be_unroll")) {
      bool is_bootstrapped = false;
      func.walk(
          [&](hecate::earth::BootstrapOp bop) { is_bootstrapped = true; });
      if (is_bootstrapped) {
        func->setAttr("unroll_factor", builder.getI64IntegerAttr(1));
        return;
      }

      for (auto blockArg : func.getBody().getArguments()) {
        if (auto sop = dyn_cast<hecate::earth::HEScaleTypeInterface>(
                blockArg.getType())) {
          if (sop.getScale() == rf && sop.getLevel() == 0) {
            startVals.push({blockArg, 0});
            while (!startVals.empty()) {
              visited.clear();
              auto &&startVal = std::get<0>(startVals.front());
              auto &&accmDepth = std::get<1>(startVals.front());
              depthMap[startVal] = {};
              for (auto &&use : startVal.getUses()) {
                dfsLevelDepth(startVal, use.getOwner(), use.get(),
                              hecate::earth::getScaleType(startVal).getLevel(),
                              accmDepth);
              }
              startVals.pop();
            }
            // draw DepthMap
            /* for (auto &&target : depthMap) { */
            /*   auto &&startVal = target.getFirst(); */
            /*   startVal.dump(); */
            /*   llvm::errs() << "----------------------\n"; */
            /*   for (auto &&depths : target.getSecond()) { */
            /*     std::get<0>(depths)->dump(); */
            /*     llvm::errs() << "depth? : " << std::get<1>(depths) << "\n\n";
             */
            /*     if (max_depth < std::get<1>(depths)) */
            /*       max_depth = std::get<1>(depths); */
            /*   } */
            /*   llvm::errs() << "==============================\n\n"; */
            /* } */

            findMaxDepth(nullptr, blockArg, 0);
          }
        }
      }
      /* llvm::errs() << "PRINT DEPTH MAP\n"; */
      /* llvm::errs() << "MAX DEPTH : " << max_depth << '\n'; */
      auto lowerBound =
          func->getAttrOfType<mlir::BoolAttr>("is_packed").getValue()
              ? hecate::earth::EarthDialect::bootstrapLevelLowerBound + 1
              : hecate::earth::EarthDialect::bootstrapLevelLowerBound;
      auto max_bound =
          hecate::earth::EarthDialect::bootstrapLevelUpperBound - lowerBound;
      uint64_t unroll_factor = max_bound / max_depth;
      /* llvm::errs() << "UNROLL FACTOR : " << unroll_factor << '\n'; */
      func->setAttr("unroll_factor", builder.getI64IntegerAttr(unroll_factor));
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<arith::ArithDialect>();
  }
};
} // namespace
