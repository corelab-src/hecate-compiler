
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/ForwardManagementInterface.h"
#include "hecate/Dialect/Earth/Transforms/Common.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
//===----------------------------------------------------------------------===//
// ForOpInterface
//===----------------------------------------------------------------------===//

struct ForOpInterface
    : public hecate::earth::ForwardMgmtInterface::ExternalModel<
          ForOpInterface, mlir::scf::ForOp> {
  void processOperandsEVA(mlir::Operation *op, int64_t param) const {
    OpBuilder builder(op);
    mlir::IRRewriter rewriter(builder);

    auto forOp = dyn_cast<mlir::scf::ForOp>(op);
    for (size_t i = forOp.getNumControlOperands(); i < forOp.getNumOperands();
         i++) {
      builder.setInsertionPoint(forOp);
      auto &&oper = dyn_cast<hecate::earth::HEScaleTypeInterface>(
          forOp.getOperand(i).getType());
      if (oper.getScale() +
              hecate::earth::EarthDialect::rescalingFactor * oper.getLevel() <
          (hecate::earth::EarthDialect::bootstrapLevelUpperBound + 1) *
              hecate::earth::EarthDialect::rescalingFactor) {
        if (oper.getScale() < hecate::earth::EarthDialect::rescalingFactor) {
          op->setOperand(i, builder.create<hecate::earth::UpscaleOp>(
                                op->getLoc(), forOp.getOperand(i),
                                hecate::earth::EarthDialect::rescalingFactor -
                                    oper.getScale()));
        } else if (oper.getScale() >
                   hecate::earth::EarthDialect::rescalingFactor) {
          int overLevel = (oper.getScale() - 1) /
                          hecate::earth::EarthDialect::rescalingFactor;
          op->setOperand(i, builder.create<hecate::earth::UpscaleOp>(
                                op->getLoc(), forOp.getOperand(i),
                                (hecate::earth::EarthDialect::rescalingFactor *
                                     (overLevel + 1) -
                                 oper.getScale())));
          for (int j = overLevel; j > 0; j--) {
            op->setOperand(i, builder.create<hecate::earth::RescaleOp>(
                                  op->getLoc(), forOp.getOperand(i)));
          }
        }
      }
      auto level_diff =
          hecate::earth::EarthDialect::bootstrapLevelUpperBound -
          hecate::earth::EarthDialect::bootstrapLevelLowerBound -
          hecate::earth::getScaleType(forOp.getOperand(i)).getLevel();
      auto mop = builder.create<hecate::earth::ModswitchOp>(
          forOp.getLoc(), forOp.getOperand(i), level_diff);
      forOp.setOperand(i, mop);

      forOp.getRegionIterArg(i - forOp.getNumControlOperands())
          .setType(op->getOperand(i).getType());
    }
    for (auto arg : forOp.getRegionIterArgs()) {
      builder.setInsertionPointAfterValue(arg);
      auto btp = builder.create<hecate::earth::BootstrapOp>(arg.getLoc(), arg);
      rewriter.replaceAllUsesExcept(arg, btp, btp);
    }
    return;
  }

  void processResultsEVA(Operation *op, int64_t param) const {
    OpBuilder builder(op);
    mlir::IRRewriter rewriter(builder);
    auto forOp = dyn_cast<mlir::scf::ForOp>(op);
    auto yieldOp =
        dyn_cast<mlir::scf::YieldOp>(forOp.getBody()->getTerminator());

    /* for (size_t yieldIdx = 0; yieldIdx < forOp.getNumResults(); yieldIdx++) {
     */
    /*   size_t initArgIdx = yieldIdx + forOp.getNumControlOperands(); */
    /*   auto aop = dyn_cast<hecate::earth::HEScaleTypeInterface>( */
    /*       forOp.getOperand(initArgIdx).getType()); */
    /*   auto yop = dyn_cast<hecate::earth::HEScaleTypeInterface>( */
    /*       yieldOp.getOperand(yieldIdx).getType()); */
    /*   if (aop.getLevel() > yop.getLevel()) { */
    /*     auto level_diff = aop.getLevel() - yop.getLevel(); */
    /*     builder.setInsertionPoint(yieldOp); */
    /*     yieldOp.setOperand( */
    /*         yieldIdx, */
    /*         builder.create<hecate::earth::ModswitchOp>( */
    /*             yieldOp.getLoc(), yieldOp.getOperand(yieldIdx), level_diff));
     */
    /*   } else if (aop.getLevel() < yop.getLevel()) { */
    /*     auto level_diff = yop.getLevel() - aop.getLevel(); */
    /*     builder.setInsertionPoint(forOp); */
    /*     forOp.setOperand( */
    /*         initArgIdx, */
    /*         builder.create<hecate::earth::ModswitchOp>( */
    /*             forOp.getLoc(), forOp.getOperand(initArgIdx), level_diff));
     */
    /*     forOp.getRegionIterArg(yieldIdx).setType( */
    /*         op->getOperand(initArgIdx).getType()); */
    /*   } */
    /* } */
    for (size_t i = 0; i < forOp.getNumResults(); i++) {
      /* forOp.getResult(i).setType( */
      /*     forOp.getBody()->getTerminator()->getOperand(i).getType()); */
      forOp.getResult(i).setType(
          forOp.getOperand(forOp.getNumControlOperands() + i).getType());
    }

    return;
  }
  void processOperandsPARS(Operation *op, int64_t param) const {
    processOperandsEVA(op, param);
    return;
  }

  void processResultsPARS(Operation *op, int64_t param) const {
    processResultsEVA(op, param);
    return;
  }

  void processOperandsSNR(Operation *op, int64_t param) const { return; }
  void processResultsSNR(Operation *op, int64_t param) const { return; }

  bool overThreshold(Operation *op, float thr) const { return false; }
  bool isBootstrappable(Operation *op) const { return true; }
  bool isValidated(Operation *op) const {
    auto forOp = dyn_cast<mlir::scf::ForOp>(op);
    return llvm::all_of(forOp.getInitArgs(), [](mlir::Value arg) {
      auto lScale = hecate::earth::getScaleType(arg);
      auto rf = hecate::earth::EarthDialect::rescalingFactor;
      return lScale.getLevel() * rf + lScale.getScale() <=
             (hecate::earth::EarthDialect::bootstrapLevelUpperBound -
              hecate::earth::EarthDialect::bootstrapLevelLowerBound + 1) *
                 rf;
    });
  }
};

struct ForOpTypeInferInterface
    : public mlir::InferTypeOpInterface::ExternalModel<ForOpTypeInferInterface,
                                                       mlir::scf::ForOp> {
  static bool isCompatibleReturnTypes(::mlir::TypeRange lhs,
                                      ::mlir::TypeRange rhs) {
    return !rhs.back()
                .dyn_cast<hecate::earth::HEScaleTypeInterface>()
                .isCipher();
  }

  static ::mlir::LogicalResult inferReturnTypes(
      ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
      ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
      ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
    auto forOp =
        mlir::scf::ForOpAdaptor(operands, attributes, properties, regions);
    auto yieldOp = dyn_cast<mlir::scf::YieldOp>(
        forOp.getRegion().getBlocks().front().getTerminator());
    for (auto arg : forOp.getInitArgs()) {
      auto lScale = hecate::earth::getScaleType(arg);
      if (lScale.getLevel() <=
          hecate::earth::EarthDialect::bootstrapLevelUpperBound -
              hecate::earth::EarthDialect::bootstrapLevelLowerBound) {
        inferredReturnTypes.push_back(arg.getType());
      } else
        return ::mlir::failure();
    }
    return ::mlir::success();
  }

  static ::mlir::LogicalResult refineReturnTypes(
      ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
      ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
      ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
      ::llvm::SmallVectorImpl<::mlir::Type> &returnTypes) {
    return ::mlir::failure();
  }
};

//===----------------------------------------------------------------------===//
// YieldOpInterface
//===----------------------------------------------------------------------===//

struct YieldOpInterface
    : public hecate::earth::ForwardMgmtInterface::ExternalModel<
          YieldOpInterface, mlir::scf::YieldOp> {
  void processOperandsEVA(Operation *op, int64_t param) const {
    OpBuilder builder(op);
    auto yieldOp = dyn_cast<mlir::scf::YieldOp>(op);
    for (size_t i = 0; i < yieldOp.getNumOperands(); i++) {
      auto &&oper = dyn_cast<hecate::earth::HEScaleTypeInterface>(
          yieldOp.getOperand(i).getType());
      if (oper.getScale() +
              hecate::earth::EarthDialect::rescalingFactor * oper.getLevel() <
          (hecate::earth::EarthDialect::bootstrapLevelUpperBound + 1) *
              hecate::earth::EarthDialect::rescalingFactor) {
        if (oper.getScale() < hecate::earth::EarthDialect::rescalingFactor) {
          op->setOperand(i, builder.create<hecate::earth::UpscaleOp>(
                                op->getLoc(), yieldOp.getOperand(i),
                                hecate::earth::EarthDialect::rescalingFactor -
                                    oper.getScale()));
        } else if (oper.getScale() >
                   hecate::earth::EarthDialect::rescalingFactor) {
          int overLevel = (oper.getScale() - 1) /
                          hecate::earth::EarthDialect::rescalingFactor;
          op->setOperand(i, builder.create<hecate::earth::UpscaleOp>(
                                op->getLoc(), yieldOp.getOperand(i),
                                (hecate::earth::EarthDialect::rescalingFactor *
                                     (overLevel + 1) -
                                 oper.getScale())));
          for (int j = overLevel; j > 0; j--) {
            op->setOperand(i, builder.create<hecate::earth::RescaleOp>(
                                  op->getLoc(), yieldOp.getOperand(i)));
          }
        }
      }
      auto level_diff =
          hecate::earth::EarthDialect::bootstrapLevelUpperBound -
          hecate::earth::EarthDialect::bootstrapLevelLowerBound -
          hecate::earth::getScaleType(yieldOp.getOperand(i)).getLevel();
      auto mop = builder.create<hecate::earth::ModswitchOp>(
          yieldOp.getLoc(), yieldOp.getOperand(i), level_diff);
      yieldOp.setOperand(i, mop);
    }

    return;
  }
  void processResultsEVA(Operation *op, int64_t param) const { return; }

  void processOperandsPARS(Operation *op, int64_t param) const {
    processOperandsEVA(op, param);
    return;
  }
  void processResultsPARS(Operation *op, int64_t param) const {
    processResultsEVA(op, param);
    return;
  }

  void processOperandsSNR(Operation *op, int64_t param) const { return; }
  void processResultsSNR(Operation *op, int64_t param) const { return; }

  bool overThreshold(Operation *op, float thr) const { return false; }
  bool isBootstrappable(Operation *op) const { return false; }
  bool isValidated(Operation *op) const { return true; }
};

void hecate::earth::registerSCFOpInterfaceExternalModels(
    mlir::DialectRegistry &registry) {
  registry.insert<scf::SCFDialect>();
  registry.addExtension(+[](MLIRContext *ctx, scf::SCFDialect *dialect) {
    scf::ForOp::attachInterface<ForOpInterface>(*ctx);
    /* scf::ForOp::attachInterface<ForOpTypeInferInterface>(*ctx); */
    scf::YieldOp::attachInterface<YieldOpInterface>(*ctx);
  });
}
