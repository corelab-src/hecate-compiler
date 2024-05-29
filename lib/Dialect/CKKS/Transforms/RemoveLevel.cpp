

#include "hecate/Dialect/CKKS/IR/CKKSOps.h"
#include "hecate/Dialect/CKKS/IR/PolyTypeInterface.h"
#include "hecate/Dialect/CKKS/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace hecate {
namespace ckks {
#define GEN_PASS_DEF_REMOVELEVEL
#include "hecate/Dialect/CKKS/Transforms/Passes.h.inc"
} // namespace ckks
} // namespace hecate

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct RemoveLevelPass
    : public hecate::ckks::impl::RemoveLevelBase<RemoveLevelPass> {
  RemoveLevelPass() {}

  void runOnOperation() override {
    markAllAnalysesPreserved();
    auto &&func = getOperation();
    mlir::OpBuilder builder(func);

    SmallVector<int64_t, 4> level_in;
    SmallVector<int64_t, 4> level_out;
    for (auto &&arg : func.getArguments()) {
      if (auto tt = arg.getType().dyn_cast<hecate::ckks::PolyTypeInterface>())
        level_in.push_back(tt.getLevel());
      else
        level_in.push_back(0);
    }
    func->setAttr("arg_level", builder.getDenseI64ArrayAttr(level_in));
    for (auto &&restype : func.getResultTypes()) {
      if (auto tt = restype.dyn_cast<hecate::ckks::PolyTypeInterface>())
        level_out.push_back(tt.getLevel());
    }
    func->setAttr("res_level", builder.getDenseI64ArrayAttr(level_out));

    for (auto value : func.getArguments()) {
      if (auto &&tt =
              value.getType().dyn_cast<hecate::ckks::PolyTypeInterface>())
        value.setType(tt.switchLevel(0));
    }
    func.walk([&](Operation *op) {
      for (auto value : op->getResults()) {
        if (auto &&tt =
                value.getType().dyn_cast<hecate::ckks::PolyTypeInterface>())
          value.setType(tt.switchLevel(0));
      }
    });
    func.walk([&](scf::ForOp ForOp) {
      auto inputTy = ForOp.getOperandTypes();
      mlir::Region &loop_body = ForOp.getBodyRegion();
      mlir::Block *loopBodyBlock = ForOp.getBody();
      for (auto argval : loop_body.getArguments()) {
        if (auto &&tt =
                argval.getType().dyn_cast<hecate::ckks::PolyTypeInterface>()) {
          argval.setType(tt.switchLevel(0));
        }
      }
      for (auto retval : ForOp.getResults()) {
        if (auto &&tt =
                retval.getType().dyn_cast<hecate::ckks::PolyTypeInterface>()) {
          retval.setType(tt.switchLevel(0));
        }
      }
    });

    auto funcType = func.getFunctionType();
    llvm::SmallVector<Type, 4> inputTys;
    llvm::SmallVector<Type, 4> outputTys;
    for (auto &&ty : funcType.getInputs()) {
      if (auto &&tt = ty.dyn_cast<hecate::ckks::PolyTypeInterface>())
        inputTys.push_back(tt.switchLevel(0));
      else
        inputTys.push_back(ty);
    }
    for (auto &&ty : funcType.getResults()) {
      if (auto &&tt = ty.dyn_cast<hecate::ckks::PolyTypeInterface>())
        outputTys.push_back(tt.switchLevel(0));
      else
        outputTys.push_back(ty);
    }
    func.setFunctionType(builder.getFunctionType(inputTys, outputTys));
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::ckks::CKKSDialect>();
  }
};
} // namespace
