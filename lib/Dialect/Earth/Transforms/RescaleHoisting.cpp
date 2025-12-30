#include "hecate/Dialect/Earth/Analysis/CandidateAnalysis.h"
#include "hecate/Dialect/Earth/Analysis/ScaleManagementUnit.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_RESCALEHOISTING
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
// #include "llvm/ADT/STLExtras.h" // just with forward iterator
} // namespace earth
} // namespace hecate

#define DEBUG_TYPE "hecate_em"

using namespace mlir;
bool debug = false;

namespace {
struct RescaleHoistingPass
    : public hecate::earth::impl::RescaleHoistingBase<RescaleHoistingPass> {
  RescaleHoistingPass() {}

  void runOnOperation() override {
    auto func = getOperation();
    markAnalysesPreserved<hecate::ScaleManagementUnit>();
    markAnalysesPreserved<hecate::CandidateAnalysis>();

    RewritePatternSet patterns(&getContext());
    patterns.add<RescaleHoistingPattern>(&getContext());

    GreedyRewriteConfig config;
    config.maxIterations = 256;
    config.maxNumRewrites = GreedyRewriteConfig::kNoLimit;
    config.enableRegionSimplification = true;

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns), config))) {
      llvm::errs() << "RescaleHoistingPass: failed to apply patterns and fold greedily\n";
      // signalPassFailure(); // if failed, terminate the opt
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
    registry.insert<scf::SCFDialect>();
  }

private:
  struct RescaleHoistingPattern
      : public OpRewritePattern<hecate::earth::AddOp> {
    using OpRewritePattern<hecate::earth::AddOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(hecate::earth::AddOp addOp,
                                  PatternRewriter &rewriter) const override {
      auto getUnscaledInfo = [&](Value operand) -> std::pair<Operation *, hecate::earth::RescaleOp> {
        Operation *definingOp = operand.getDefiningOp();
        if (!definingOp) return {nullptr, nullptr};

        // Case 1: Operand is directly from a RescaleOp.
        if (auto rescale = dyn_cast<hecate::earth::RescaleOp>(definingOp)) {
          return {rescale, rescale};
        }

        // Case 2: Operand is from RotateOp -> RescaleOp.
        if (auto rotate = dyn_cast<hecate::earth::RotateOp>(definingOp)) {
          Operation *rescaleCandidateOp = rotate.getOperand().getDefiningOp();
          if (!rescaleCandidateOp) {
            return {nullptr, nullptr};
          }

          if (auto rescale =
                  dyn_cast<hecate::earth::RescaleOp>(rescaleCandidateOp)) {
            return {rotate, rescale};
          }
        }
        return {nullptr, nullptr};
      };

      // Analyze both operands of the AddOp.
      auto [lhsChainOp, lhsRescaleOp] = getUnscaledInfo(addOp.getLhs());
      auto [rhsChainOp, rhsRescaleOp] = getUnscaledInfo(addOp.getRhs());

      // The pattern only matches if both operands have a preceding RescaleOp.
      if (!lhsRescaleOp || !rhsRescaleOp) return failure();
      if (debug) {
        llvm::outs() << "RescaleHoistingPattern: " << addOp.getOperation()->getName().getStringRef() << " is a AddOp\n";
        llvm::outs() << "  -> LHS: " << lhsChainOp->getName().getStringRef() << "  "
                     << "RHS: " << rhsChainOp->getName().getStringRef() << "\n";
      }

      // Before hoisting, the types must be compatible (same scale and level).
      auto lhsUnscaledType = lhsRescaleOp.getOperand().getType().dyn_cast<hecate::earth::HEScaleTypeInterface>();
      auto rhsUnscaledType = rhsRescaleOp.getOperand().getType().dyn_cast<hecate::earth::HEScaleTypeInterface>();
      if (!lhsUnscaledType || !rhsUnscaledType) return failure();
      if (lhsUnscaledType.getScale() != rhsUnscaledType.getScale() ||
          lhsUnscaledType.getLevel() != rhsUnscaledType.getLevel()) return failure();
      
      // Hoist the RescaleOp to the operand of the AddOp.
      // Set Operations before insertion point with previous operand and attributes (Scale and Level).
      auto createNewOperand = [&](Operation *chainOp, hecate::earth::RescaleOp foundRescaleOp) -> Value {
        if (isa<hecate::earth::RescaleOp>(chainOp)) {
          return foundRescaleOp.getOperand();
        }
        if (auto rotateOp = dyn_cast<hecate::earth::RotateOp>(chainOp)) {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(addOp);
          Operation *newRotateOp = rewriter.clone(*rotateOp);
          newRotateOp->setOperand(0, foundRescaleOp.getOperand());
          newRotateOp->getResult(0).setType(foundRescaleOp.getOperand().getType());
          return newRotateOp->getResult(0);
        }
        return nullptr;
      };

      Value newLhs = createNewOperand(lhsChainOp, lhsRescaleOp);
      Value newRhs = createNewOperand(rhsChainOp, rhsRescaleOp);
      if (!newLhs || !newRhs) return failure();

      // Create a new AddOp with the new operands and replace the old AddOp.
      rewriter.setInsertionPoint(addOp);
      auto newAdd = rewriter.create<hecate::earth::AddOp>(addOp.getLoc(), newLhs, newRhs);
      auto newRescale = rewriter.create<hecate::earth::RescaleOp>(addOp.getLoc(), newAdd.getResult());
      rewriter.replaceOp(addOp, newRescale.getResult());
      
      return success();
    }
  };
};
} // namespace