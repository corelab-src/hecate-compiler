

#include "hecate/Conversion/CKKSToCKKS/SCFToDpsSCF.h"
#include "hecate/Conversion/CKKSCommon/PolyTypeConverter.h"

#include "hecate/Dialect/CKKS/IR/CKKSOps.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "mlir/Conversion/ArithCommon/AttrToLLVMConverter.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <type_traits>

namespace hecate {
#define GEN_PASS_DEF_SCFTODPSSCFCONVERSION
#include "hecate/Conversion/Passes.h.inc"
} // namespace hecate

using namespace mlir;
using namespace hecate;

namespace {

//===----------------------------------------------------------------------===//
// Straightforward Op Lowerings
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Op Lowering Patterns
//===----------------------------------------------------------------------===//

struct ForOpLowering : public OpConversionPattern<mlir::scf::ForOp> {
  using OpConversionPattern<mlir::scf::ForOp>::ConversionPattern;
  ForOpLowering(MLIRContext *ctxt)
      : OpConversionPattern<mlir::scf::ForOp>(ctxt) {}

  LogicalResult
  matchAndRewrite(mlir::scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct YieldOpLowering : public OpConversionPattern<mlir::scf::YieldOp> {
  using OpConversionPattern<mlir::scf::YieldOp>::ConversionPattern;
  YieldOpLowering(MLIRContext *ctxt)
      : OpConversionPattern<mlir::scf::YieldOp>(ctxt) {}

  LogicalResult
  matchAndRewrite(mlir::scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace

//===----------------------------------------------------------------------===//
// ForOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
ForOpLowering::matchAndRewrite(mlir::scf::ForOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  /* op.dump(); */

  for (size_t i = op.getNumControlOperands(); i < op.getNumOperands(); i++) {
    auto tt = op.getOperand(i).getType().dyn_cast<mlir::RankedTensorType>();
    auto dst = rewriter.create<tensor::EmptyOp>(op.getLoc(), tt.getShape(),
                                                tt.getElementType());
    auto copyc = rewriter.create<hecate::ckks::CopyCOp>(op.getLoc(), dst,
                                                        op.getOperand(i));
    op.setOperand(i, copyc);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// YieldOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
YieldOpLowering::matchAndRewrite(mlir::scf::YieldOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  /* auto tt = */
  /*     op.getType().getElementType().dyn_cast<hecate::ckks::PolyTypeInterface>();
   */
  /* auto dst = rewriter.create<tensor::EmptyOp>( */
  /*     op.getLoc(), op.getType().getShape(), tt.switchNumPoly(1)); */
  /* op->insertOperands(0, {dst}); */

  return success();
}
//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct SCFToDpsSCFConversion
    : public hecate::impl::SCFToDpsSCFConversionBase<SCFToDpsSCFConversion> {

  using Base::Base;

  void runOnOperation() override {
    ConversionTarget target(getContext());

    auto func = getOperation();

    mlir::RewritePatternSet patterns(&getContext());

    target.addLegalDialect<hecate::ckks::CKKSDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalDialect<func::FuncDialect>();

    hecate::ckks::populateSCFToDpsSCFConversionPatterns(&getContext(),
                                                        patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//
//
void hecate::ckks::populateSCFToDpsSCFConversionPatterns(
    mlir::MLIRContext *ctxt, mlir::RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<ForOpLowering>(ctxt);
  /* patterns.add<YieldOpLowering>(ctxt); */

  // clang-format on
}

std::unique_ptr<::mlir::OperationPass<::mlir::func::FuncOp>>
hecate::ckks::createSCFToDpsSCFConversionPass() {
  return std::make_unique<SCFToDpsSCFConversion>();
}
