#include "hecate/Dialect/Earth/Transforms/Common.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Support/Support.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"

using namespace mlir;

void hecate::earth::refineLevel(mlir::OpBuilder builder, mlir::Operation *op,
                                int64_t waterline, int64_t output_val,
                                int64_t min_level) {
  int64_t max_required_level =
      hecate::earth::EarthDialect::bootstrapLevelUpperBound - min_level;
  if (max_required_level < 0) {
    max_required_level =
        hecate::earth::EarthDialect::levelUpperBound - min_level;
  }

  builder.setInsertionPoint(op);

  int64_t acc_scale_max = 0;
  int64_t rescalingFactor = hecate::earth::EarthDialect::rescalingFactor;
  for (auto v : op->getOperands()) {
    auto st = v.getType().dyn_cast<hecate::earth::HEScaleTypeInterface>();
    auto acc_scale = st.getLevel() * rescalingFactor + st.getScale();
    acc_scale_max = std::max(acc_scale_max, acc_scale);
  }
  for (size_t i = 0; i < op->getNumOperands(); i++) {
    auto v = op->getOperand(i);
    if (hecate::getIntegerAttr("is_bypassed", v) > 0) {
      continue;
    }
    auto st = v.getType().dyn_cast<hecate::earth::HEScaleTypeInterface>();
    auto acc_scale =
        st.getLevel() * rescalingFactor + st.getScale() + output_val;
    auto max_acc_scale = max_required_level * rescalingFactor;
    int64_t level_diff = (max_acc_scale - acc_scale) / rescalingFactor;
    op->setOperand(i, builder.create<hecate::earth::ModswitchOp>(
                          op->getLoc(), v, level_diff));
  }
}

void hecate::earth::refineReturnValues(mlir::func::FuncOp func,
                                       mlir::OpBuilder builder,
                                       SmallVector<mlir::Type, 4> inputTypes,
                                       int64_t waterline, int64_t output_val) {
  using namespace mlir;

  // Compute required init level.
  int64_t max_required_level =
      hecate::earth::EarthDialect::bootstrapLevelUpperBound;
  // -      hecate::earth::EarthDialect::levelLowerBound;
  if (max_required_level < 0)
    max_required_level = hecate::earth::EarthDialect::levelUpperBound;
  // -                         hecate::earth::EarthDialect::levelLowerBound;

  // Get the return op.
  auto rop = dyn_cast<func::ReturnOp>(func.getBlocks().front().getTerminator());
  assert(rop && "Function expected to end with func.return");

  // Small helpers.
  auto getBool = [&](StringRef name) -> bool {
    if (auto a = func->getAttrOfType<BoolAttr>(name))
      return a.getValue();
    return false;
  };

  auto applySegmentBypassesIfAny = [&](func::ReturnOp ret) {
    if (!getBool("is_mid_segment"))
      return;
    if (!func->hasAttr("segment_returnBypasses"))
      return;

    auto arr = func->getAttrOfType<ArrayAttr>("segment_returnBypasses");
    if (!arr)
      return;

    const size_t n = std::min(static_cast<size_t>(ret->getNumOperands()),
                              static_cast<size_t>(arr.size()));
    for (size_t i = 0; i < n; ++i) {
      bool isBypass = false;
      if (auto b = arr[i].dyn_cast<BoolAttr>())
        isBypass = b.getValue();
      auto v = ret->getOperand(i);
      hecate::setIntegerAttr("is_bypassed", v, isBypass);
    }
  };

  const bool isPacked = getBool("is_packed");
  const bool isMidSegment = getBool("is_mid_segment");
  const bool isMidSection = getBool("is_mid_section");

  // Apply bypass tags where applicable (only when mid-segment).
  applySegmentBypassesIfAny(rop);

  // Decide parameters for refineLevel() once, then call it.
  int64_t outValForRefine = output_val;
  int64_t minLevelForRefine = hecate::earth::EarthDialect::levelLowerBound;

  if (isPacked) {
    outValForRefine = 0;
    minLevelForRefine =
        isMidSegment ? hecate::earth::EarthDialect::bootstrapLevelLowerBound - 1
                     : hecate::earth::EarthDialect::bootstrapLevelLowerBound;
  } else if (isMidSection || isMidSegment) {
    outValForRefine = 0;
    minLevelForRefine =
        hecate::earth::EarthDialect::bootstrapLevelLowerBound - 1;
  } else {
    outValForRefine = output_val;
    minLevelForRefine = hecate::earth::EarthDialect::levelLowerBound;
  }

  hecate::earth::refineLevel(builder, rop, waterline, outValForRefine,
                             minLevelForRefine);

  // Remap the return types.
  func.setFunctionType(
      builder.getFunctionType(inputTypes, rop.getOperandTypes()));

  // Attach attributes.
  func->setAttr("init_level", builder.getI64IntegerAttr(max_required_level));

  SmallVector<int64_t, 4> scales_in;
  SmallVector<int64_t, 4> scales_out;
  scales_in.reserve(func.getNumArguments());
  scales_out.reserve(func.getNumResults());

  for (auto &&arg : func.getArguments()) {
    if (auto tt = arg.getType().dyn_cast<hecate::earth::HEScaleTypeInterface>())
      scales_in.push_back(tt.getScale());
    else
      scales_in.push_back(0);
  }
  func->setAttr("arg_scale", builder.getDenseI64ArrayAttr(scales_in));

  for (auto &&restype : func.getResultTypes()) {
    if (auto tt = restype.dyn_cast<hecate::earth::HEScaleTypeInterface>())
      scales_out.push_back(tt.getScale());
  }
  func->setAttr("res_scale", builder.getDenseI64ArrayAttr(scales_out));
}

SmallVector<mlir::Type, 4>
hecate::earth::getInputValueTypes(mlir::func::FuncOp func,
                                  mlir::OpBuilder builder, int64_t waterline,
                                  int64_t output_val) {
  SmallVector<mlir::Type, 4> inputTypes;
  // Set function argument types
  if (!func->hasAttr("segment_inputType")) {
    for (auto &&argval : func.getArguments()) {
      mlir::Type ty = argval.getType();
      if (auto tt = argval.getType().dyn_cast<RankedTensorType>()) {
        ty = RankedTensorType::get(
            tt.getShape(), tt.getElementType()
                               .dyn_cast<hecate::earth::HEScaleTypeInterface>()
                               .switchScale(waterline));
      }
      inputTypes.push_back(ty);
    }
  } else {
    auto &&inputType_attrs = func->getAttr("segment_inputType")
                                 .dyn_cast<mlir::ArrayAttr>()
                                 .getValue();
    /* llvm::errs() << "func size: " << func.getNumArguments() << '\n'; */
    /* llvm::errs() << "input size: " << inputType_attrs.size() << '\n'; */
    for (size_t i = 0; i < func.getNumArguments(); i++) {
      auto &&argval = func.getArgument(i);
      mlir::Type ty = argval.getType();

      if (auto input_type =
              inputType_attrs[i]
                  .dyn_cast<mlir::TypeAttr>()
                  .getValue()
                  .dyn_cast<hecate::earth::HEScaleTypeInterface>())
        ty = input_type;
      /* llvm::errs() << ty << '\n'; */
      inputTypes.push_back(ty);
    }
  }
  return inputTypes;
}

/* llvm::SmallVector<mlir::Type, 4> */
void hecate::earth::refineInputValues(mlir::func::FuncOp func,
                                      mlir::OpBuilder builder,
                                      SmallVector<mlir::Type, 4> &inputTypes,
                                      int64_t waterline, int64_t output_val) {
  // Set function argument types
  if (!func->hasAttr("segment_inputType")) {
    for (auto &&argval : func.getArguments()) {
      if (auto tt = argval.getType().dyn_cast<RankedTensorType>()) {
        argval.setType(RankedTensorType::get(
            tt.getShape(), tt.getElementType()
                               .dyn_cast<hecate::earth::HEScaleTypeInterface>()
                               .switchScale(waterline)));
      }
      inputTypes.push_back(argval.getType());
    }
  } else {
    auto &&inputType_attrs = func->getAttr("segment_inputType")
                                 .dyn_cast<mlir::ArrayAttr>()
                                 .getValue();
    for (size_t i = 0; i < func.getNumArguments(); i++) {
      auto &&argval = func.getArgument(i);
      if (auto input_type =
              inputType_attrs[i]
                  .dyn_cast<mlir::TypeAttr>()
                  .getValue()
                  .dyn_cast<hecate::earth::HEScaleTypeInterface>())
        argval.setType(input_type);
      inputTypes.push_back(argval.getType());
    }
  }
  func.walk([&](scf::ForOp ForOp) {
    auto inputTy = ForOp.getOperandTypes();
    mlir::Region &loop_body = ForOp.getBodyRegion();
    mlir::Block *loopBodyBlock = ForOp.getBody();
    for (auto argval : loop_body.getArguments()) {
      if (auto sop = argval.getType().dyn_cast<mlir::RankedTensorType>()) {
        argval.setType(mlir::RankedTensorType::get(
            argval.getType().dyn_cast<mlir::RankedTensorType>().getShape(),
            argval.getType()
                .dyn_cast<mlir::RankedTensorType>()
                .getElementType()
                .dyn_cast<hecate::earth::HEScaleTypeInterface>()
                .switchScale(hecate::earth::EarthDialect::rescalingFactor)));
      }
    }
  });

  return;
}

void hecate::earth::inferTypeForward(hecate::earth::ForwardMgmtInterface sop) {
  Operation *oop = sop.getOperation();
  if (auto iop = dyn_cast<mlir::InferTypeOpInterface>(oop)) {
    SmallVector<Type, 4> retTypes;
    if (iop.inferReturnTypes(oop->getContext(), oop->getLoc(),
                             oop->getOperands(), oop->getAttrDictionary(),
                             oop->getPropertiesStorage(), oop->getRegions(),
                             retTypes)
            .succeeded()) {
      oop->getResults().back().setType(retTypes.back());
    }
  }
}

llvm::SmallVector<mlir::Value, 4> hecate::earth::attachOpid(mlir::Block *bb) {
  llvm::SmallVector<mlir::Value, 4> values;
  // attach the opid to operation
  values.push_back(NULL);
  /* bb->walk([&](hecate::earth::HEScaleOpInterface sop) { */
  for (auto iter = bb->begin(); iter != bb->end(); ++iter) {
    mlir::Operation *op = &*iter;
    if ((llvm::isa<hecate::earth::UpscaleOp>(op) ||
         llvm::isa<hecate::earth::RescaleOp>(op) ||
         llvm::isa<hecate::earth::BootstrapOp>(op) ||
         llvm::isa<hecate::earth::DummyOp>(op) ||
         llvm::isa<hecate::earth::ModswitchOp>(op))) {
      /* assert(0 && "Currently not supported"); */
      continue;
    }
    for (auto &&val : op->getResults()) {
      values.push_back(val);
    }
    if (auto forOp = dyn_cast<mlir::scf::ForOp>(op)) {
      auto &&bb = forOp.getBody();
      auto &&v = attachOpid(bb);
      values.append(v.begin() + 1, v.end());
    }
  }
  return values;
}

llvm::DenseMap<int64_t, mlir::Value>
hecate::earth::getOpidToValueMap(mlir::Block *bb) {
  llvm::DenseMap<int64_t, mlir::Value> valueMap;
  // attach the opid to operation
  /* values.push_back(NULL); */
  valueMap[0] = NULL;
  /* bb->walk([&](hecate::earth::HEScaleOpInterface sop) { */
  for (auto iter = bb->begin(); iter != bb->end(); ++iter) {
    mlir::Operation *op = &*iter;
    if ((llvm::isa<hecate::earth::UpscaleOp>(op) ||
         llvm::isa<hecate::earth::RescaleOp>(op) ||
         llvm::isa<hecate::earth::BootstrapOp>(op) ||
         llvm::isa<hecate::earth::DummyOp>(op) ||
         llvm::isa<hecate::earth::ModswitchOp>(op))) {
      /* assert(0 && "Currently not supported"); */
      continue;
    }
    for (auto &&val : op->getResults()) {
      auto opid = hecate::getIntegerAttr("opid", val);
      valueMap[opid] = val;
    }
    if (auto forOp = dyn_cast<mlir::scf::ForOp>(op)) {
      auto &&bb = forOp.getBody();
      auto &&v = getOpidToValueMap(bb);
      valueMap.insert(v.begin(), v.end());
    }
  }
  return valueMap;
}
