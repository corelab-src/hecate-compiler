
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
  DenseMap<mlir::Operation *,
           SmallVector<std::tuple<mlir::Operation *, uint64_t, uint64_t>, 4>>
      depthMap;
  // start operation,accmDepth
  std::queue<std::tuple<Operation *, uint64_t>> startOps;

  mlir::SmallVector<Operation *> visited;
  std::stack<mlir::Operation *> st;
  uint64_t max_depth = 0;

  void dfsLevelDepth(mlir::Operation *startOp, mlir::Operation *curOp,
                     uint64_t startDepth, uint64_t accmDepth) {
    if (std::find(visited.begin(), visited.end(), curOp) != visited.end())
      return;
    visited.push_back(curOp);
    if (auto btp = dyn_cast<hecate::earth::BootstrapOp>(curOp)) {
      auto depth = hecate::earth::getScaleType(btp.getOperand()).getLevel();
      auto gap = depth - startDepth;
      depthMap[startOp].push_back({curOp, gap, accmDepth + gap});
      startOps.push({btp, accmDepth + gap});
      return;
    } else if (auto pack = dyn_cast<hecate::earth::PackOp>(curOp)) {
      auto depth = hecate::earth::getScaleType(pack.getOperand(0)).getLevel();
      auto gap = depth - startDepth;
      depthMap[startOp].push_back({curOp, gap, accmDepth + gap});
      return;
    } else if (auto pack = dyn_cast<mlir::scf::YieldOp>(curOp)) {
      auto depth = hecate::earth::getScaleType(pack.getOperand(0)).getLevel();
      auto gap = depth - startDepth;
      depthMap[startOp].push_back({curOp, gap, accmDepth + gap});
      return;
    }

    for (auto &&use : curOp->getUsers()) {
      dfsLevelDepth(startOp, use, startDepth, accmDepth);
    }
    return;
  }

  void findMaxDepth(mlir::Operation *edgeOp, uint64_t accmDepth) {

    if (isa<hecate::earth::PackOp>(edgeOp)) {
      max_depth = max_depth < accmDepth ? accmDepth : max_depth;
      return;
    }
    for (auto &&opInfo : depthMap[edgeOp]) {
      mlir::Operation *destOp = std::get<0>(opInfo);
      auto &&dist = std::get<1>(opInfo);
      accmDepth += dist;
      findMaxDepth(destOp, accmDepth);
      accmDepth -= dist;
    }
  }

  void runOnOperation() override {

    auto func = getOperation();
    auto mod = func->getParentOfType<mlir::ModuleOp>();
    mlir::OpBuilder builder(func);
    mlir::IRRewriter rewriter(builder);
    mlir::RewritePatternSet pattern(builder.getContext());
    /* llvm::errs() << "FUNC DUMP\n"; */
    /* func.dump(); */

    if (func->hasAttr("be_unroll") &&
        func->getAttrOfType<mlir::BoolAttr>("be_unroll")) {
      func.walk([&](hecate::earth::UnPackOp pop) {
        /* max_depth = 0; */
        startOps.push({pop, 0});
        while (!startOps.empty()) {
          visited.clear();
          auto &&startOp = std::get<0>(startOps.front());
          auto &&accmDepth = std::get<1>(startOps.front());
          depthMap[startOp] = {};
          for (auto &&use : startOp->getUsers()) {
            dfsLevelDepth(
                startOp, use,
                hecate::earth::getScaleType(startOp->getResult(0)).getLevel(),
                accmDepth);
          }
          startOps.pop();
        }

        findMaxDepth(pop, 0);
        mlir::Operation *edgeOp = pop;
        uint64_t chain_depth = 0;
        while (!isa<hecate::earth::PackOp>(edgeOp)) {
          mlir::Operation *destOp = std::get<0>(depthMap[edgeOp].front());
          auto &&dist = std::get<1>(depthMap[edgeOp].front());
          chain_depth += dist;
          edgeOp = destOp;
        }
        /* max_depth = max_depth < chain_depth ? chain_depth : max_depth; */
      });
      llvm::errs() << "PRINT DEPTH MAP\n";
      llvm::errs() << "MAX DEPTH : " << max_depth << '\n';
      for (auto &&target : depthMap) {
        auto &&startOp = target.getFirst();
        startOp->dump();
        llvm::errs() << "----------------------\n";
        for (auto &&depths : target.getSecond()) {
          std::get<0>(depths)->dump();
          llvm::errs() << "depth? : " << std::get<1>(depths) << "\n\n";
          if (max_depth < std::get<1>(depths))
            max_depth = std::get<1>(depths);
        }
        llvm::errs() << "==============================\n\n";
      }

      auto lowerBound =
          func->getAttrOfType<mlir::BoolAttr>("is_packed").getValue()
              ? hecate::earth::EarthDialect::bootstrapLevelLowerBound + 1
              : hecate::earth::EarthDialect::bootstrapLevelLowerBound;
      auto max_bound =
          hecate::earth::EarthDialect::bootstrapLevelUpperBound - lowerBound;
      uint64_t unroll_factor = max_bound / max_depth;
      llvm::errs() << "UNROLL FACTOR : " << unroll_factor << '\n';
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
