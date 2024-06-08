
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
#include "mlir/IR/Value.h"
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
    llvm::errs() << __FILE__ << '\n';
    auto func = getOperation();
    auto mod = func->getParentOfType<mlir::ModuleOp>();
    auto rescalingFactor = hecate::earth::EarthDialect::rescalingFactor;

    mlir::OpBuilder builder(func);
    mlir::IRRewriter rewriter(builder);

    llvm::errs() << __FILE__ << " : " << __LINE__ << '\n';
    auto &&bb = func.getBody().getBlocks().front();
    auto &&inputType_attrs = func->getAttr("segment_inputType")
                                 .dyn_cast<mlir::ArrayAttr>()
                                 .getValue();
    auto &&inputs =
        func->getAttrOfType<mlir::DenseI64ArrayAttr>("segment_input")
            .asArrayRef();

    llvm::errs() << __FILE__ << " : " << __LINE__ << '\n';
    SmallVector<mlir::Type, 4> modified_inputTypes;
    SmallVector<int64_t, 4> additional_btp_target;
    auto tp = mlir::RankedTensorType::get(
        llvm::SmallVector<int64_t, 1>{1},
        builder.getType<hecate::earth::CipherType>(rescalingFactor, 0));
    llvm::errs() << __FILE__ << " : " << __LINE__ << '\n';

    for (auto tt : inputType_attrs)
      modified_inputTypes.push_back(tt.dyn_cast<mlir::TypeAttr>().getValue());

    llvm::errs() << __FILE__ << " : " << __LINE__ << '\n';
    for (auto iter = bb.begin(); iter != bb.end(); ++iter) {
      if (auto fop = dyn_cast<mlir::scf::ForOp>(&*iter)) {
        auto &&loopOp = dyn_cast<mlir::LoopLikeOpInterface>(fop.getOperation());
        for (size_t i = 0; i < inputs.size(); i++) {
          auto arg =
              func.getArgument(func.getNumArguments() - inputs.size() + i);
          /* for (auto use : arg.getUsers()) { */
          /*   if (!isa<mlir::scf::ForOp>(use) && */
          /*       !loopOp.isDefinedOutsideOfLoop(use->getResult(0))) { */
          /*     auto initArgNumber = */
          /*         dyn_cast<mlir::BlockArgument>(arg).getArgNumber(); */
          /*     modified_inputTypes[initArgNumber] = tp; */
          /*     additional_btp_target.push_back(inputs[i]); */
          /*     break; */
          /*   } */
          /* } */
        }
        // set RegionIterArgs as func input arguments
        for (size_t i = 0; i < fop.getNumRegionIterArgs(); i++) {
          auto initArg = fop.getOperand(i + fop.getNumControlOperands());
          auto initArgNumber =
              dyn_cast<mlir::BlockArgument>(initArg).getArgNumber();
          modified_inputTypes[initArgNumber] = tp;
          fop.getRegionIterArg(i).replaceAllUsesWith(initArg);
        }

        auto &&bb = fop.getBody();
        for (auto iter = bb->begin(); iter != bb->end(); iter = bb->begin()) {

          mlir::Operation *op = &*iter;
          if (auto yop = dyn_cast<mlir::scf::YieldOp>(op)) {
            for (auto oper : yop.getOperands())
              hecate::setIntegerAttr("is_yielded", oper, 1);
            fop->getResults().replaceAllUsesWith(yop.getOperands());
            fop->erase();
            func->setAttr("segment_inputType",
                          builder.getTypeArrayAttr(modified_inputTypes));

            func->setAttr("additional_btp_target",
                          builder.getDenseI64ArrayAttr(additional_btp_target));
            return;
          }
          loopOp.moveOutOfLoop(op);
        }
      }
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<arith::ArithDialect>();
  }
};
} // namespace
