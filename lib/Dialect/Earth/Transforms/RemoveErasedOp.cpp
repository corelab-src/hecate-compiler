
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
    func.walk([&](hecate::earth::EraseTypeOp eop) {
      eop.replaceAllUsesWith(eop.getOperand());
      eop.erase();
    });
    func.walk([&](scf::ForOp ForOp) {
      auto inputTy = ForOp.getOperandTypes();
      mlir::Region &loop_body = ForOp.getBodyRegion();
      mlir::Block *loopBodyBlock = ForOp.getBody();
      for (size_t i = 0; i < loop_body.getNumArguments(); i++) {
        loop_body.getArgument(i).setType(ForOp.getOperand(i + 2).getType());
      }
      for (size_t i = 0; i < ForOp.getNumResults(); i++) {
        ForOp.getResult(i).setType(
            loopBodyBlock->getTerminator()->getOperand(i).getType());
      }
      auto funcType = func.getFunctionType();
      mlir::SmallVector<mlir::Type, 4> retTypes;
      for (auto ret : func.getRegion().front().getTerminator()->getOperands()) {
        retTypes.push_back(ret.getType());
      }
      func.setFunctionType(
          builder.getFunctionType(funcType.getInputs(), retTypes));
    });
    /* func.dump(); */
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
    registry.insert<mlir::scf::SCFDialect>();
  }
};
} // namespace
