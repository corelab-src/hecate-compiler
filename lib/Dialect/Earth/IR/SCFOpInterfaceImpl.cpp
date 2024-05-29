
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/ForwardManagementInterface.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

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
      auto oper = dyn_cast<hecate::earth::HEScaleTypeInterface>(
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

    for (size_t yieldIdx = 0; yieldIdx < forOp.getNumResults(); yieldIdx++) {
      size_t initArgIdx = yieldIdx + forOp.getNumControlOperands();
      auto aop = dyn_cast<hecate::earth::HEScaleTypeInterface>(
          forOp.getOperand(initArgIdx).getType());
      auto yop = dyn_cast<hecate::earth::HEScaleTypeInterface>(
          yieldOp.getOperand(yieldIdx).getType());
      auto level_diff = hecate::earth::EarthDialect::bootstrapLevelUpperBound -
                        hecate::earth::EarthDialect::bootstrapLevelLowerBound -
                        yop.getLevel();

      builder.setInsertionPoint(yieldOp);
      yieldOp.setOperand(
          yieldIdx,
          builder.create<hecate::earth::ModswitchOp>(
              yieldOp.getLoc(), yieldOp.getOperand(yieldIdx), level_diff));
      level_diff = hecate::earth::EarthDialect::bootstrapLevelUpperBound -
                   hecate::earth::EarthDialect::bootstrapLevelLowerBound -
                   aop.getLevel();

      builder.setInsertionPoint(forOp);
      forOp.setOperand(
          initArgIdx,
          builder.create<hecate::earth::ModswitchOp>(
              forOp.getLoc(), forOp.getOperand(initArgIdx), level_diff));
      forOp.getRegionIterArg(yieldIdx).setType(
          op->getOperand(initArgIdx).getType());
    }
    for (size_t i = 0; i < forOp.getNumResults(); i++) {
      forOp.getResult(i).setType(
          forOp.getBody()->getTerminator()->getOperand(i).getType());
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
  bool isBootstrappable(Operation *op) const { return false; }
  bool isValidated(Operation *op) const { return false; }
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
      auto oper = dyn_cast<hecate::earth::HEScaleTypeInterface>(
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
    scf::YieldOp::attachInterface<YieldOpInterface>(*ctx);
  });
}
