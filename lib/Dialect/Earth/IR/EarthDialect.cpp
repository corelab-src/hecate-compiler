

#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "nlohmann/json.hpp"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"
#include <fstream>

using namespace mlir;

#define GET_TYPEDEF_CLASSES
#include "hecate/Dialect/Earth/IR/EarthOpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "hecate/Dialect/Earth/IR/EarthOps.cpp.inc"

#include "hecate/Dialect/Earth/IR/EarthOpsDialect.cpp.inc"

#include "hecate/Dialect/Earth/IR/EarthCanonicalizerPattern.inc"

struct ScaleTypeTensorModel
    : public hecate::earth::HEScaleTypeInterface::ExternalModel<
          ScaleTypeTensorModel, mlir::RankedTensorType> {
  bool isCipher(Type t) const {
    if (auto scaleType = t.dyn_cast<mlir::RankedTensorType>()
                             .getElementType()
                             .dyn_cast<hecate::earth::HEScaleTypeInterface>()) {
      return scaleType.isCipher();
    } else {
      return false;
    }
  }
  hecate::earth::HEScaleTypeInterface toCipher(Type t) const {
    auto tt = t.dyn_cast<RankedTensorType>();
    return dyn_cast<hecate::earth::HEScaleTypeInterface>(RankedTensorType::get(
        tt.getShape(), tt.getElementType()
                           .dyn_cast<hecate::earth::HEScaleTypeInterface>()
                           .toCipher()));
  }
  hecate::earth::HEScaleTypeInterface toPlain(Type t) const {
    auto tt = t.dyn_cast<RankedTensorType>();
    return dyn_cast<hecate::earth::HEScaleTypeInterface>(RankedTensorType::get(
        tt.getShape(), tt.getElementType()
                           .dyn_cast<hecate::earth::HEScaleTypeInterface>()
                           .toPlain()));
  }
  hecate::earth::HEScaleTypeInterface toErased(Type t) const {
    auto tt = t.dyn_cast<RankedTensorType>();
    return dyn_cast<hecate::earth::HEScaleTypeInterface>(RankedTensorType::get(
        tt.getShape(), tt.getElementType()
                           .dyn_cast<hecate::earth::HEScaleTypeInterface>()
                           .toErased()));
  }
  hecate::earth::HEScaleTypeInterface switchScale(Type t,
                                                  unsigned scale) const {
    auto tt = t.dyn_cast<RankedTensorType>();
    return dyn_cast<hecate::earth::HEScaleTypeInterface>(RankedTensorType::get(
        tt.getShape(), tt.getElementType()
                           .dyn_cast<hecate::earth::HEScaleTypeInterface>()
                           .switchScale(scale)));
  }
  hecate::earth::HEScaleTypeInterface switchLevel(Type t,
                                                  unsigned level) const {
    auto tt = t.dyn_cast<RankedTensorType>();
    return dyn_cast<hecate::earth::HEScaleTypeInterface>(RankedTensorType::get(
        tt.getShape(), tt.getElementType()
                           .dyn_cast<hecate::earth::HEScaleTypeInterface>()
                           .switchLevel(level)));
  }
  unsigned getScale(Type t) const {
    if (auto scaleType = t.dyn_cast<mlir::RankedTensorType>()
                             .getElementType()
                             .dyn_cast<hecate::earth::HEScaleTypeInterface>()) {
      return scaleType.getScale();
    } else {
      return 0;
    }
  }
  unsigned getLevel(Type t) const {
    if (auto scaleType = t.dyn_cast<mlir::RankedTensorType>()
                             .getElementType()
                             .dyn_cast<hecate::earth::HEScaleTypeInterface>()) {
      return scaleType.getLevel();
    } else {
      return 0;
    }
  }
};

struct ForOpMgmtInterfaceModel
    : public hecate::earth::ForwardMgmtInterface::ExternalModel<
          ForOpMgmtInterfaceModel, mlir::scf::ForOp> {
  void processOperandsEVA(mlir::Operation *op, int64_t param) const {
    OpBuilder builder(op);
    mlir::IRRewriter rewriter(builder);

    auto forOp = dyn_cast<mlir::scf::ForOp>(op);
    for (size_t i = forOp.getNumControlOperands(); i < forOp.getNumOperands();
         i++) {
      builder.setInsertionPoint(forOp);
      auto oper = dyn_cast<hecate::earth::HEScaleOpInterface>(
                      forOp.getOperand(i).getDefiningOp())
                      .getScaleType();
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

    for (auto tt : forOp.getYieldedValues())
      llvm::errs() << tt.getType();
    for (size_t yieldIdx = 0; yieldIdx < forOp.getNumResults(); yieldIdx++) {
      size_t initArgIdx = yieldIdx + forOp.getNumControlOperands();
      auto aop = dyn_cast<hecate::earth::HEScaleOpInterface>(
                     forOp.getOperand(initArgIdx).getDefiningOp())
                     .getScaleType();
      auto yop = dyn_cast<hecate::earth::HEScaleOpInterface>(
                     yieldOp.getOperand(yieldIdx).getDefiningOp())
                     .getScaleType();
      if (aop.getLevel() > yop.getLevel()) {
        auto level_diff = aop.getLevel() - yop.getLevel();
        yieldOp.setOperand(
            yieldIdx,
            builder.create<hecate::earth::ModswitchOp>(
                yieldOp.getLoc(), yieldOp.getOperand(yieldIdx), level_diff));
      } else if (aop.getLevel() < yop.getLevel()) {
        auto level_diff = yop.getLevel() - aop.getLevel();
        forOp.setOperand(
            initArgIdx,
            builder.create<hecate::earth::ModswitchOp>(
                forOp.getLoc(), forOp.getOperand(initArgIdx), level_diff));
        forOp.getRegionIterArg(yieldIdx).setType(
            op->getOperand(initArgIdx).getType());
      }
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

struct YieldOpMgmtInterfaceModel
    : public hecate::earth::ForwardMgmtInterface::ExternalModel<
          YieldOpMgmtInterfaceModel, mlir::scf::YieldOp> {
  void processOperandsEVA(Operation *op, int64_t param) const {
    OpBuilder builder(op);
    auto yieldOp = dyn_cast<mlir::scf::YieldOp>(op);
    for (size_t i = 0; i < yieldOp.getNumOperands(); i++) {
      auto oper = dyn_cast<hecate::earth::HEScaleOpInterface>(
                      yieldOp.getOperand(i).getDefiningOp())
                      .getScaleType();
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
    scf::ForOp::attachInterface<ForOpMgmtInterfaceModel>(*ctx);
    scf::YieldOp::attachInterface<YieldOpMgmtInterfaceModel>(*ctx);
  });
}

void hecate::earth::EarthDialect::initialize() {
  // Registers all the Types into the EVADialect class
  addTypes<
#define GET_TYPEDEF_LIST
#include "hecate/Dialect/Earth/IR/EarthOpsTypes.cpp.inc"
      >();

  // Registers all the Operations into the EVADialect class
  addOperations<
#define GET_OP_LIST
#include "hecate/Dialect/Earth/IR/EarthOps.cpp.inc"
      >();
  mlir::RankedTensorType::attachInterface<ScaleTypeTensorModel>(*getContext());
}

std::map<std::string, llvm::SmallVector<int64_t>> latencyTable;

std::map<std::string, llvm::SmallVector<double>> noiseTable;

int64_t hecate::earth::EarthDialect::polynomialDegree;
int64_t hecate::earth::EarthDialect::rescalingFactor;
int64_t hecate::earth::EarthDialect::bootstrapLevelLowerBound;
int64_t hecate::earth::EarthDialect::bootstrapLevelUpperBound;
int64_t hecate::earth::EarthDialect::levelUpperBound;
int64_t hecate::earth::EarthDialect::levelLowerBound;

template <typename... Args>
void addOperationNamesTo(SmallVectorImpl<StringRef> &names) {
  (void)std::initializer_list<int>{
      0, (names.push_back(Args::getOperationName()), 0)...};
}

std::map<std::string, llvm::SmallVector<int64_t>>
hecate::earth::EarthDialect::getLatencyTable() {
  return latencyTable;
}
std::map<std::string, llvm::SmallVector<double>>
hecate::earth::EarthDialect::getNoiseTable() {
  return noiseTable;
}

void hecate::earth::EarthDialect::setCKKSParameters(llvm::StringRef filename) {
  // it should be read from file but currently fixed
  hecate::earth::EarthDialect::levelLowerBound = 1;  // 0 should be forbidden
  hecate::earth::EarthDialect::levelUpperBound = 13; // L =14
                                                     // inclusion
  hecate::earth::EarthDialect::polynomialDegree = 1LL << 15;

  SmallVector<StringRef, 4> names;
  addOperationNamesTo<
#define GET_OP_LIST
#include "hecate/Dialect/Earth/IR/EarthOps.cpp.inc"
      >(names);

  std::ifstream iff(filename.str());
  nlohmann::json config = nlohmann::json::parse(iff);
  hecate::earth::EarthDialect::rescalingFactor = config["rescalingFactor"];
  hecate::earth::EarthDialect::levelLowerBound = config["levelLowerBound"];
  hecate::earth::EarthDialect::levelUpperBound = config["levelUpperBound"];
  hecate::earth::EarthDialect::bootstrapLevelLowerBound =
      config["bootstrapLevelLowerBound"];
  hecate::earth::EarthDialect::bootstrapLevelUpperBound =
      config["bootstrapLevelUpperBound"];
  hecate::earth::EarthDialect::polynomialDegree = config["polynomialDegree"];

  for (auto &&name : names) {
    for (auto &&suffix : {"_single", "_double"}) {
      auto &latTab = latencyTable[name.str() + suffix];
      auto &noTab = noiseTable[name.str() + suffix];
      latTab.push_back(0);
      noTab.push_back(0);
      if (config["latencyTable"].contains(name.str() + suffix) &&
          config["latencyTable"][name.str() + suffix].size()) {
        for (auto &&data : config["latencyTable"][name.str() + suffix]) {
          latTab.push_back(data.get<int64_t>());
        }
      }
      latTab.resize(levelUpperBound + 1, latTab.back());

      if (config["noiseTable"].contains(name.str() + suffix) &&
          config["noiseTable"][name.str() + suffix].size()) {
        for (auto &&data : config["noiseTable"][name.str() + suffix]) {
          noTab.push_back(data.get<double>());
        }
      }
      noTab.resize(levelUpperBound + 1, noTab.back());

      latencyTable[name.str() + suffix] = latTab;
      noiseTable[name.str() + suffix] = noTab;
    }
  }
}

::mlir::LogicalResult hecate::earth::ConstantOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto op = ConstantOpAdaptor(operands, attributes, properties, regions);

  inferredReturnTypes.push_back(mlir::RankedTensorType::get(
      llvm::SmallVector<int64_t, 1>{1}, PlainType::get(context, 0, 0)));
  return ::mlir::success();
}

::mlir::LogicalResult hecate::earth::VariableOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto op = VariableOpAdaptor(operands, attributes, properties, regions);

  inferredReturnTypes.push_back(mlir::RankedTensorType::get(
      llvm::SmallVector<int64_t, 1>{1}, ErasedType::get(context, 0, 0)));
  return ::mlir::success();
}

::mlir::LogicalResult hecate::earth::EraseTypeOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto op = EraseTypeOpAdaptor(operands, attributes, properties, regions);
  auto lScale = earth::getScaleType(op.getValue());
  inferredReturnTypes.push_back(mlir::RankedTensorType::get(
      llvm::SmallVector<int64_t, 1>{1},
      ErasedType::get(context, lScale.getScale(), lScale.getLevel())));
  return ::mlir::success();
}
void hecate::earth::EraseTypeOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<RescaleUpscalePattern>(context);
}
void hecate::earth::RescaleOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<RescaleUpscalePattern>(context);
}
void hecate::earth::RotateOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  /* patterns.add<RotateOffsetModuloPattern>(context); */
}
/* ::mlir::LogicalResult hecate::earth::RotateOp::inferReturnTypes( */
/*     ::mlir::MLIRContext *context, ::std::optional<::mlir::Location>
 * location, */
/*     ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, */
/*     ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, */
/*     ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) { */
/*   auto op = RotateOpAdaptor(operands, attributes, regions); */
/*   auto lScale = earth::getScaleType(op.getValue()); */
/*   inferredReturnTypes.push_back(lScale); */
/*   return ::mlir::success(); */
/* } */

::mlir::LogicalResult hecate::earth::RescaleOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto op = RescaleOpAdaptor(operands, attributes, properties, regions);
  auto lScale = earth::getScaleType(op.getValue());
  inferredReturnTypes.push_back(
      lScale.switchLevel(lScale.getLevel() + 1)
          .switchScale(lScale.getScale() -
                       hecate::earth::EarthDialect::rescalingFactor));

  return ::mlir::success();
}

void hecate::earth::ModswitchOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<ModswitchModswitchPattern, ModswitchConstantPattern,
               ZeroModswitchPattern>(context);
}

::mlir::LogicalResult hecate::earth::ModswitchOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto op = ModswitchOpAdaptor(operands, attributes, properties, regions);
  auto lScale = earth::getScaleType(op.getValue());
  if (op.getDownFactor() >= 0) {
    inferredReturnTypes.push_back(
        lScale.switchLevel(lScale.getLevel() + op.getDownFactor()));
    return ::mlir::success();
  } else
    return ::mlir::failure();
}

void hecate::earth::UpscaleOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns
      .add<UpscaleUpscalePattern, UpscaleConstantPattern, ZeroUpscalePattern>(
          context);
}

::mlir::LogicalResult hecate::earth::UpscaleOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto op = UpscaleOpAdaptor(operands, attributes, properties, regions);
  auto lScale = earth::getScaleType(op.getValue());
  if (op.getUpFactor() >= 0) {
    inferredReturnTypes.push_back(
        lScale.switchScale(lScale.getScale() + op.getUpFactor()));
    return ::mlir::success();
  } else
    return ::mlir::failure();
}

void hecate::earth::BootstrapOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  /* patterns.add<exceedBootstrapBound>(context); */
}

::mlir::LogicalResult hecate::earth::BootstrapOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto op = BootstrapOpAdaptor(operands, attributes, properties, regions);
  auto lScale = earth::getScaleType(op.getValue());
  // accumulated scale > maximum scale limit
  if (lScale.getLevel() <=
      hecate::earth::EarthDialect::bootstrapLevelUpperBound -
          hecate::earth::EarthDialect::bootstrapLevelLowerBound) {
    if (lScale.getScale() == 0) {
      inferredReturnTypes.push_back(lScale.switchLevel(0));
    } else {
      inferredReturnTypes.push_back(lScale.switchLevel(op.getTargetLevel()));
    }
    return ::mlir::success();
  } else
    return ::mlir::failure();
}

void hecate::earth::AddOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<AddZeroPattern>(context);
}

::mlir::LogicalResult hecate::earth::AddOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto op = AddOpAdaptor(operands, attributes, properties, regions);
  auto lScale = earth::getScaleType(op.getLhs());
  auto lTensor = earth::getTensorType(op.getLhs());
  auto rScale = earth::getScaleType(op.getRhs());
  auto rTensor = earth::getTensorType(op.getRhs());
  // nulltype rScale means indexType
  if (!rScale || (lScale.getLevel() == rScale.getLevel() &&
                  lScale.getScale() == rScale.getScale() &&
                  lTensor.getShape()[0] == rTensor.getShape()[0])) {
    inferredReturnTypes.push_back(lScale.toCipher());
    return ::mlir::success();
  } else {
    return ::mlir::failure();
  }
}

void hecate::earth::MulOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<MulZeroPattern, MulOnePattern, NegMulPattern>(context);
}

::mlir::LogicalResult hecate::earth::MulOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto op = MulOpAdaptor(operands, attributes, properties, regions);
  auto lScale = earth::getScaleType(op.getLhs());
  auto lTensor = earth::getTensorType(op.getLhs());
  auto rScale = earth::getScaleType(op.getRhs());
  auto rTensor = earth::getTensorType(op.getRhs());
  if (!rScale || (lScale.getLevel() == rScale.getLevel() &&
                  lTensor.getShape()[0] == rTensor.getShape()[0] &&
                  ((EarthDialect::bootstrapLevelUpperBound)*EarthDialect::
                       rescalingFactor >=
                   lScale.getLevel() * EarthDialect::rescalingFactor +
                       lScale.getScale()))) {
    inferredReturnTypes.push_back(
        lScale.switchScale(lScale.getScale() + rScale.getScale()).toCipher());
    return ::mlir::success();
  } else {
    return ::mlir::failure();
  }
}

mlir::RankedTensorType hecate::earth::getTensorType(mlir::Value v) {
  return v.getType().dyn_cast<mlir::RankedTensorType>();
}
hecate::earth::HEScaleTypeInterface hecate::earth::getScaleType(mlir::Value v) {
  return v.getType().dyn_cast<hecate::earth::HEScaleTypeInterface>();
}
