

#include "hecate/Conversion/CKKSToCKKS/ArithToDpsArith.h"
#include "hecate/Conversion/CKKSCommon/PolyTypeConverter.h"

#include "hecate/Dialect/CKKS/IR/CKKSOps.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "mlir/Conversion/ArithCommon/AttrToLLVMConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <type_traits>

namespace hecate {
#define GEN_PASS_DEF_ARITHTODPSARITHCONVERSION
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
struct ConstantOpLowering
    : public OpConversionPattern<mlir::arith::ConstantOp> {
  using OpConversionPattern<mlir::arith::ConstantOp>::ConversionPattern;
  ConstantOpLowering(MLIRContext *ctxt)
      : OpConversionPattern<mlir::arith::ConstantOp>(ctxt) {}

  LogicalResult
  matchAndRewrite(mlir::arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct AddIOpLowering : public OpConversionPattern<mlir::arith::AddIOp> {
  using OpConversionPattern<mlir::arith::AddIOp>::ConversionPattern;
  AddIOpLowering(MLIRContext *ctxt)
      : OpConversionPattern<mlir::arith::AddIOp>(ctxt) {}

  LogicalResult
  matchAndRewrite(mlir::arith::AddIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct SubIOpLowering : public OpConversionPattern<mlir::arith::SubIOp> {
  using OpConversionPattern<mlir::arith::SubIOp>::ConversionPattern;
  SubIOpLowering(MLIRContext *ctxt)
      : OpConversionPattern<mlir::arith::SubIOp>(ctxt) {}

  LogicalResult
  matchAndRewrite(mlir::arith::SubIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct RemSIOpLowering : public OpConversionPattern<mlir::arith::RemSIOp> {
  using OpConversionPattern<mlir::arith::RemSIOp>::ConversionPattern;
  RemSIOpLowering(MLIRContext *ctxt)
      : OpConversionPattern<mlir::arith::RemSIOp>(ctxt) {}

  LogicalResult
  matchAndRewrite(mlir::arith::RemSIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

//===----------------------------------------------------------------------===//
// ConstantOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
ConstantOpLowering::matchAndRewrite(mlir::arith::ConstantOp op,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  /* op.dump(); */
  auto dst =
      rewriter.create<ckks::EmptyIOp>(op.getLoc(), op.getType(), ValueRange());
  /* dst.dump(); */

  auto tt = rewriter.replaceOpWithNewOp<ckks::ConstIOp>(
      op, dst, adaptor.getValue().dyn_cast<IntegerAttr>().getInt());
  /* tt.dump(); */
  /* llvm::errs() << '\n'; */

  return success();
}

//===----------------------------------------------------------------------===//
// AddIOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
AddIOpLowering::matchAndRewrite(mlir::arith::AddIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  /* auto dst = rewriter.create<tensor::EmptyOp>( */
  /*     op.getLoc(), llvm::SmallVector<int64_t, 1>{1}, op.getType()); */
  /* auto dst = */
  /*     rewriter.create<tensor::EmptyOp>(op.getLoc(), op.getType(),
   * ValueRange()); */
  /* op->insertOperands(0, {dst}); */
  /* auto dst = rewriter.create<tensor::EmptyOp>( */
  /*     op.getLoc(), llvm::SmallVector<int64_t, 1>{1}, op.getType()); */

  /* rewriter.replaceOpWithNewOp<ckks::AddIOp>(op, dst, adaptor.getRhs(), */
  /*                                           adaptor.getLhs()); */
  /* op.dump(); */
  auto dst =
      rewriter.create<ckks::EmptyIOp>(op.getLoc(), op.getType(), ValueRange());
  /* dst.dump(); */

  auto tt = rewriter.replaceOpWithNewOp<ckks::AddIOp>(op, dst, adaptor.getRhs(),
                                                      adaptor.getLhs());
  /* tt.dump(); */
  /* llvm::errs() << '\n'; */

  return success();
}
//===----------------------------------------------------------------------===//
// SubIOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
SubIOpLowering::matchAndRewrite(mlir::arith::SubIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  /* auto dst = rewriter.create<tensor::EmptyOp>( */
  /*     op.getLoc(), llvm::SmallVector<int64_t, 1>{1}, op.getType()); */
  /* rewriter.setInsertionPointAfter(op); */
  /* auto insertOp = */
  /*     rewriter.create<tensor::InsertOp>(op.getLoc(), op, dst, ValueRange());
   */
  /* auto dst = */
  /*     rewriter.create<tensor::EmptyOp>(op.getLoc(), op.getType(),
   * ValueRange()); */
  /* auto dst = rewriter.create<tensor::EmptyOp>( */
  /*     op.getLoc(), llvm::SmallVector<int64_t, 1>{1}, op.getType()); */

  /* rewriter.replaceOpWithNewOp<ckks::SubIOp>(op, dst, adaptor.getRhs(), */
  /*                                           adaptor.getLhs()); */
  /* op.dump(); */
  auto dst =
      rewriter.create<ckks::EmptyIOp>(op.getLoc(), op.getType(), ValueRange());
  /* dst.dump(); */

  auto tt = rewriter.replaceOpWithNewOp<ckks::SubIOp>(op, dst, adaptor.getRhs(),
                                                      adaptor.getLhs());
  /* tt.dump(); */
  /* llvm::errs() << '\n'; */

  return success();
}
//===----------------------------------------------------------------------===//
// RemSIOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
RemSIOpLowering::matchAndRewrite(mlir::arith::RemSIOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  /* auto dst = rewriter.create<tensor::EmptyOp>( */
  /*     op.getLoc(), llvm::SmallVector<int64_t, 1>{1}, op.getType()); */
  /* rewriter.setInsertionPointAfter(op); */
  /* auto insertOp = */
  /*     rewriter.create<tensor::InsertOp>(op.getLoc(), op, dst, ValueRange());
   */
  /* auto dst = rewriter.create<tensor::EmptyOp>( */
  /*     op.getLoc(), llvm::SmallVector<int64_t, 1>{1}, op.getType()); */

  /* rewriter.replaceOpWithNewOp<ckks::RemSIOp>(op, dst, adaptor.getRhs(), */
  /*                                            adaptor.getLhs()); */
  auto dst =
      rewriter.create<ckks::EmptyIOp>(op.getLoc(), op.getType(), ValueRange());
  /* dst.dump(); */

  auto tt = rewriter.replaceOpWithNewOp<ckks::RemSIOp>(
      op, dst, adaptor.getRhs(), adaptor.getLhs());

  return success();
}
//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct ArithToDpsArithConversion
    : public hecate::impl::ArithToDpsArithConversionBase<
          ArithToDpsArithConversion> {

  using Base::Base;

  void runOnOperation() override {
    ConversionTarget target(getContext());

    mlir::RewritePatternSet patterns(&getContext());

    target.addIllegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<hecate::ckks::CKKSDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalDialect<func::FuncDialect>();

    hecate::ckks::populateArithToDpsArithConversionPatterns(&getContext(),
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
void hecate::ckks::populateArithToDpsArithConversionPatterns(
    mlir::MLIRContext *ctxt, mlir::RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<ConstantOpLowering> (ctxt);
  patterns.add<AddIOpLowering> (ctxt);
  patterns.add<SubIOpLowering> (ctxt);
  patterns.add<RemSIOpLowering> (ctxt);

  // clang-format on
}
std::unique_ptr<::mlir::OperationPass<::mlir::func::FuncOp>>
hecate::ckks::createArithToDpsArithConversionPass() {
  return std::make_unique<ArithToDpsArithConversion>();
}
