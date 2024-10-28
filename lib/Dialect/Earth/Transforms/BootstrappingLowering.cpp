

#include "hecate/Dialect/Earth/Analysis/CandidateAnalysis.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Common.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/Debug.h"
#include <fstream>
#include <random>

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_BOOTSTRAPPINGLOWERING
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate
  //
#define DEBUG_TYPE "dacapo"

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct BootstrappingLoweringPass
    : public hecate::earth::impl::BootstrappingLoweringBase<
          BootstrappingLoweringPass> {
  BootstrappingLoweringPass() {}
  void runOnOperation() override {
    auto func = getOperation();
    // prepare the bootstrapping constants.

    func.walk([&](hecate::earth::BootstrapOp bop) {
      // replace bop with the actual arithmetic operation
      auto builder = OpBuilder(bop);
      auto origin = bop.getOperand();
      auto dM = hecate::generateConstant(
          hecate::earth::EarthDialect::polynomialDegree);
      auto forwardMat = hecate::mergeMatrix(dM, {5, 6, 6});
      auto backwardIdx = std::vector<int>{6, 6, 5};
      auto backwardMat = hecate::mergeMatrix(dM, backwardIdx);
      for (int i = 0; i < backwardIdx.size(); i++) {
        backwardMat[i] =
            hecate::dInverse(backwardMat[i], (1 << backwardIdx[i]));
      }
      backwardMat = std::vector<hecate::StepMatrix>(backwardMat.rbegin(),
                                                    backwardMat.rend());
      mlir::Value Scaled =
          builder.create<hecate::earth::ScaleCastOp>(bop.getLoc(), origin, 8);
      // Scaled = builder.create<hecate::earth::MultIntOp>(bop.getLoc(), Scaled,
      // 5);

      mlir::Value FFTed = origin;
      for (size_t i = 0; i < forwardMat.size(); i++) {
        auto &&mat = forwardMat[i];
        FFTed = hecate::dvMultGen(builder, FFTed, mat);
        FFTed = builder.create<hecate::earth::RescaleOp>(bop.getLoc(), FFTed);
      }
      mlir::Value Uped =
          builder.create<hecate::earth::LevelRecoverOp>(bop.getLoc(), FFTed, 0);
      mlir::Value InvFFTed = Uped;
      for (size_t i = 0; i < backwardMat.size(); i++) {
        auto &&mat = backwardMat[i];
        InvFFTed = hecate::dvMultGen(builder, InvFFTed, mat);
        if (i != backwardMat.size() - 1) {
          InvFFTed =
              builder.create<hecate::earth::RescaleOp>(bop.getLoc(), InvFFTed);
        }
      }
      auto Conj =
          builder.create<hecate::earth::ConjugateOp>(bop.getLoc(), InvFFTed);
      auto Removed = builder.create<hecate::earth::RescaleOp>(
          bop.getLoc(),
          builder.create<hecate::earth::AddOp>(bop.getLoc(), Conj, InvFFTed));

      mlir::Value Retreived =
          builder.create<hecate::earth::ScaleCastOp>(bop.getLoc(), origin, -8);
      mlir::Value Booted = Retreived;

      bop.replaceAllUsesWith(Booted);
      // rewriter.replaceAllUsesExcept(bop, btp, btp);
    });
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
