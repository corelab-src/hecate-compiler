#include "hecate/Conversion/EarthToEarth/EarthToEarth.h"
#include "hecate/Conversion/CKKSCommon/PolyTypeConverter.h"

#include "hecate/Dialect/CKKS/IR/CKKSOps.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/Transforms/Common.h"
#include "mlir/Conversion/ArithCommon/AttrToLLVMConverter.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <type_traits>

namespace hecate {
#define GEN_PASS_DEF_EARTHTOEARTHCONVERSION
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

/// Directly lower to LLVM op.
struct PackOpLowering : public OpConversionPattern<hecate::earth::PackOp> {
  using OpConversionPattern<hecate::earth::PackOp>::ConversionPattern;
  PackOpLowering(MLIRContext *ctxt)
      : OpConversionPattern<hecate::earth::PackOp>(ctxt) {}

  LogicalResult
  matchAndRewrite(hecate::earth::PackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct UnPackOpLowering : public OpConversionPattern<hecate::earth::UnPackOp> {
  using OpConversionPattern<hecate::earth::UnPackOp>::ConversionPattern;
  UnPackOpLowering(MLIRContext *ctxt)
      : OpConversionPattern<hecate::earth::UnPackOp>(ctxt) {}

  LogicalResult
  matchAndRewrite(hecate::earth::UnPackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace

//===----------------------------------------------------------------------===//
// PackOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
PackOpLowering::matchAndRewrite(hecate::earth::PackOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  auto &&num_elements = adaptor.getNumElements();

  Value packedRes = adaptor.getOperands()[0];
  SmallVector<mlir::Value, 4> bePacked;
  if (adaptor.getNumInputs() > 1) {
    for (size_t i = 0; i < adaptor.getNumInputs(); i++) {
      SmallVector<double, 4> oneArray, padArray;
      oneArray.resize(num_elements, 1.0);
      padArray.resize(num_elements * adaptor.getNumInputs(), 0.0);
      std::copy(oneArray.begin(), oneArray.end(),
                padArray.begin() + i * num_elements);

      Value padConst = rewriter.create<hecate::earth::ConstantOp>(
          op.getLoc(), llvm::ArrayRef(padArray));

      padConst.setType(
          hecate::earth::getScaleType(padConst)
              .switchLevel(hecate::earth::getScaleType(adaptor.getOperands()[i])
                               .getLevel())
              .switchScale(hecate::earth::getScaleType(adaptor.getOperands()[i])
                               .getScale()));

      Value &&mul = rewriter.create<hecate::earth::MulOp>(
          op.getLoc(), adaptor.getOperands()[i], padConst);
      bePacked.push_back(
          rewriter.create<hecate::earth::RescaleOp>(op.getLoc(), mul));
    }
    auto &&prevOper = bePacked[0];
    for (size_t i = 1; i < adaptor.getNumInputs(); i++) {
      auto &&currentOper = rewriter.create<hecate::earth::AddOp>(
          op.getLoc(), bePacked[i], prevOper);
      prevOper = currentOper;
    }
    packedRes = prevOper;
  }

  rewriter.replaceOp(op, packedRes);
  return success();
}

//===----------------------------------------------------------------------===//
// UnPackOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
UnPackOpLowering::matchAndRewrite(hecate::earth::UnPackOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  auto &&num_elements = adaptor.getNumElements();
  auto num_slot = hecate::earth::EarthDialect::polynomialDegree >> 1;
  auto log_slot = log2(num_slot);
  SmallVector<Value> unPackedRes;
  if (adaptor.getNumOutputs() > 1) {
    for (size_t i = 0; i < adaptor.getNumOutputs(); i++) {
      SmallVector<double, 4> oneArray, padArray;
      oneArray.resize(num_elements, 1.0);
      padArray.resize(num_slot, 0.0);
      std::copy(oneArray.begin(), oneArray.end(),
                padArray.begin() + i * num_elements);
      Value padConst = rewriter.create<hecate::earth::ConstantOp>(
          op.getLoc(), llvm::ArrayRef(padArray));
      padConst.setType(
          hecate::earth::getScaleType(padConst)
              .switchLevel(hecate::earth::getScaleType(adaptor.getInputs()[0])
                               .getLevel())
              .switchScale(hecate::earth::getScaleType(adaptor.getInputs()[0])
                               .getScale()));

      Value &&mul = rewriter.create<hecate::earth::MulOp>(
          op.getLoc(), adaptor.getInputs()[0], padConst);
      Value &&extractedResult =
          rewriter.create<hecate::earth::RescaleOp>(op.getLoc(), mul);
      auto res = extractedResult;
      for (size_t offset = 1; offset < (num_slot / num_elements);
           offset = offset << 1) {
        auto rot = rewriter.create<hecate::earth::RotateOp>(
            op.getLoc(), res, offset * num_elements);
        res = rewriter.create<hecate::earth::AddOp>(op.getLoc(), res, rot);
      }
      unPackedRes.push_back(res);
    }
  } else
    unPackedRes.push_back(adaptor.getInputs()[0]);

  rewriter.replaceOp(op, unPackedRes);
  return success();
}
//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct EarthToEarthConversion
    : public hecate::impl::EarthToEarthConversionBase<EarthToEarthConversion> {
  using Base::Base;

  void runOnOperation() override {
    ConversionTarget target(getContext());

    auto func = getOperation();

    mlir::RewritePatternSet patterns(&getContext());

    target.addIllegalOp<hecate::earth::PackOp>();
    target.addIllegalOp<hecate::earth::UnPackOp>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<hecate::earth::EarthDialect>();

    hecate::earth::populateEarthToEarthConversionPatterns(&getContext(),
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
void hecate::earth::populateEarthToEarthConversionPatterns(
    mlir::MLIRContext *ctxt, mlir::RewritePatternSet &patterns) {
  // clang-format off
  /* patterns.add<ConstantOpLowering, VariableOpLowering> (converter, ctxt, init_level); */
  patterns.add<
    PackOpLowering,
   UnPackOpLowering
 >(ctxt);

  // clang-format on
}

std::unique_ptr<::mlir::OperationPass<::mlir::func::FuncOp>>
hecate::earth::createEarthToEarthConversionPass() {
  return std::make_unique<EarthToEarthConversion>();
}
