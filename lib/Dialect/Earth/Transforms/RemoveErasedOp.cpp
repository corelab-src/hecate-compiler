
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include <fstream>

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_REMOVEERASEDOP
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct RemoveErasedOpPass
    : public hecate::earth::impl::RemoveErasedOpBase<RemoveErasedOpPass> {
  RemoveErasedOpPass() {}

  void runOnOperation() override {

    auto func = getOperation();
    mlir::OpBuilder builder(func);
    llvm::SmallVector<hecate::earth::CastOp, 4> toErase;

    func.walk([&](scf::ForOp ForOp) {
      mlir::Region &loop_body = ForOp.getBodyRegion();
      mlir::Block *loopBodyBlock = ForOp.getBody();

      for (size_t i = 0; i < loop_body.getNumArguments(); i++) {
        size_t forOpArgIdx = i + 2;
        auto operTy = ForOp.getOperand(forOpArgIdx).getType();
        if (auto erTy =
                operTy.dyn_cast<hecate::earth::HEScaleTypeInterface>()) {
          if (erTy.isErased() && !erTy.hasUnknownScale() &&
              !erTy.hasUnknownLevel()) {
            // defOp should be CastOp
            auto defOp = ForOp.getOperand(forOpArgIdx).getDefiningOp();
            ForOp.setOperand(forOpArgIdx, defOp->getOperand(0));
            defOp->erase();
          }
        }
        loop_body.getArgument(i).setType(
            ForOp.getOperand(forOpArgIdx).getType());
      }

      // set yield op types as concrete types
      for (size_t i = 0; i < ForOp.getNumResults(); i++) {
        auto yieldOper = loopBodyBlock->getTerminator()->getOperand(i);
        if (auto erTy = yieldOper.getType()
                            .dyn_cast<hecate::earth::HEScaleTypeInterface>()) {
          if (erTy.isErased() && !erTy.hasUnknownScale() &&
              !erTy.hasUnknownLevel()) {
            // defOp should be CastOp
            auto defOp = yieldOper.getDefiningOp();
            loopBodyBlock->getTerminator()->setOperand(i, defOp->getOperand(0));
            defOp->erase();
          }
        }
        ForOp.getResult(i).setType(
            loopBodyBlock->getTerminator()->getOperand(i).getType());
      }
    });
    func->walk([&](hecate::earth::HEScaleOpInterface sop) {
      if (llvm::isa<hecate::earth::BootstrapOp>(sop) ||
          llvm::isa<hecate::earth::RotateOp>(sop) ||
          llvm::isa<hecate::earth::NegateOp>(sop)) {
        sop->getResult(0).setType(sop->getOperand(0).getType());
        return;
      }
    });

    auto funcType = func.getFunctionType();
    mlir::SmallVector<mlir::Type, 4> retTypes;
    for (auto ret : func.getRegion().front().getTerminator()->getOperands()) {
      retTypes.push_back(ret.getType());
    }
    func.setFunctionType(
        builder.getFunctionType(funcType.getInputs(), retTypes));

    /* func.dump(); */
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
    registry.insert<mlir::scf::SCFDialect>();
  }
};
} // namespace
