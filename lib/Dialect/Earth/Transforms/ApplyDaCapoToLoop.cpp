
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
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include <filesystem>
#include <fstream>
#include <queue>
#include <tuple>

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_APPLYDACAPOTOLOOP
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct ApplyDaCapoToLoopPass
    : public hecate::earth::impl::ApplyDaCapoToLoopBase<ApplyDaCapoToLoopPass> {
  ApplyDaCapoToLoopPass() {}
  ApplyDaCapoToLoopPass(hecate::earth::ApplyDaCapoToLoopOptions ops) {
    this->waterline = ops.waterline;
    this->output_val = ops.output_val;
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto mod_func = func->getParentOfType<mlir::ModuleOp>();
    /* llvm::errs() << "enter APPLY DACAPO \n"; */
    auto &ca = getAnalysis<hecate::CandidateAnalysis>();
    /* func.dump(); */

    mlir::OpBuilder builder(func);
    mlir::IRRewriter rewriter(builder);

    auto mod = mlir::ModuleOp::create(func.getLoc());
    PassManager pm_dacapo(mod.getContext()), pm_nested(mod.getContext());
    pm_dacapo.addNestedPass<func::FuncOp>(
        hecate::earth::createCandidateSelection({waterline, output_val}));
    pm_dacapo.addNestedPass<func::FuncOp>(
        hecate::earth::createDaCapoPlanner({waterline, output_val}));
    pm_dacapo.addNestedPass<func::FuncOp>(
        hecate::earth::createBootstrapPlacement());
    pm_dacapo.addNestedPass<func::FuncOp>(
        hecate::earth::createProactiveRescaling({waterline, output_val}));
    pm_dacapo.addNestedPass<func::FuncOp>(
        hecate::earth::createWrapUpLoopBody());

    pm_nested.addNestedPass<func::FuncOp>(
        hecate::earth::createCodeSegmentation());
    pm_nested.addNestedPass<func::FuncOp>(
        hecate::earth::createPromoteLoopBody());
    pm_nested.addNestedPass<func::FuncOp>(
        hecate::earth::createApplyDaCapoToLoop());

    PassManager pm_boot(mod_func.getContext());
    pm_boot.addNestedPass<func::FuncOp>(
        hecate::earth::createBootstrapPlacement());

    // {from, to}, {inputType, returnType, bypassTypes}
    SmallVector<int64_t, 2> btp_target;
    // opid, {offset,  inputType}
    DenseMap<int64_t, std::tuple<uint64_t, SmallVector<Type, 4>>> forOpTable;

    SmallVector<Type, 4> initialTypes;
    initialTypes =
        hecate::earth::getInputValueTypes(func, builder, waterline, output_val);

    auto dup = func.clone();
    auto &&bb = &dup.getBody().getBlocks().front();
    size_t &&loop_offset = 0;
    for (auto iter = bb->begin(); iter != bb->end(); ++iter) {
      if (auto forOp = dyn_cast<mlir::scf::ForOp>(*iter)) {
        // record the erased ForOp
        auto &&forOpid = hecate::getIntegerAttr("opid", forOp.getResult(0));

        forOpTable[forOpid] = {loop_offset, {}};
        loop_offset += forOp.getBody()->getOperations().size();
        forOp.getBody()->clear();
        // generate the YieldOp for ForOp Type Checking
        builder.setInsertionPointToStart(forOp.getBody());

        builder.create<mlir::scf::YieldOp>(forOp.getLoc(),
                                           forOp.getRegionIterArgs());
      }
    }

    /* llvm::errs() << "BEFORE DACAPO PROCESS \n"; */
    /* dup.dump(); */

    mod.push_back(dup);
    if (failed(pm_dacapo.run(mod))) {
      llvm::errs() << "Apply DaCapo failed" << '\n';
    }

    /* llvm::errs() << "AFTER DACAPO PROCESS \n"; */
    /* dup.dump(); */
    auto &&segment_btp_target =
        dup->getAttrOfType<mlir::DenseI64ArrayAttr>("btp_target").asArrayRef();
    btp_target.append(segment_btp_target.begin(), segment_btp_target.end());

    for (auto targetForOp : forOpTable) {
      auto forOpid = targetForOp.getFirst();
      auto values = hecate::earth::getOpidToValueMap(&dup.getRegion().front());

      SmallVector<Type, 4> inputTypes(initialTypes);
      for (auto inputId : ca.getValueInfo(forOpid - 1)->getLiveOuts()) {
        /* values[inputId].dump(); */
        inputTypes.push_back(values[inputId].getType());
      }
      forOpTable[forOpid] = {0, inputTypes};
    }

    dup.erase();
    for (auto targetForOp : forOpTable) {
      auto ddup = func.clone();
      auto forOpid = targetForOp.getFirst();
      auto &&forOp = std::get<0>(targetForOp.getSecond());
      auto &&inputType = std::get<1>(targetForOp.getSecond());

      ddup->setAttr("cutted_edge",
                    builder.getDenseI64ArrayAttr({forOpid - 1, forOpid}));
      ddup->setAttr("segment_input",
                    builder.getDenseI64ArrayAttr(
                        ca.getValueInfo(forOpid - 1)->getLiveOuts()));
      ddup->setAttr("segment_inputType", builder.getTypeArrayAttr(inputType));
      ddup->setAttr("is_mid_section", builder.getBoolAttr(true));

      ddup->setAttr("segment_return",
                    builder.getDenseI64ArrayAttr(
                        ca.getValueInfo(forOpid)->getLiveOuts()));
      mod.push_back(ddup);
      /* llvm::errs() << "INPUT TYPES\n"; */
      /* for (auto tt : inputType) */
      /*   llvm::errs() << tt << '\n'; */
      /* llvm::errs() << "BEFORE PROCESS FOROP : " << forOpid << '\n'; */
      if (failed(pm_nested.run(mod))) {
        llvm::errs() << "Apply DaCapo failed" << '\n';
      }

      /* llvm::errs() << "AFTER PROCESS FOROP : " << forOpid << '\n'; */
      segment_btp_target =
          ddup->getAttrOfType<mlir::DenseI64ArrayAttr>("btp_target")
              .asArrayRef();
      btp_target.append(segment_btp_target.begin(), segment_btp_target.end());

      ddup.erase();
    }

    func->setAttr("btp_target", builder.getDenseI64ArrayAttr(btp_target));
    /* func->setAttr("return_type", builder.getTypeArrayAttr( */
    /*                                  std::get<0>(edgeTypes[ca.getRetOpid()])));
     */
    /* mod.push_back(func); */
    if (failed(pm_boot.run(mod_func))) {
      llvm::errs() << "Apply DaCapo failed" << '\n';
    }

    if (ca.getIdMap().size()) {
      auto idMap = ca.getIdMap();
      SmallVector<int64_t, 2> convertedTarget;
      for (auto tt : btp_target) {
        convertedTarget.push_back(idMap[tt]);
      }
      func->setAttr("btp_target",
                    builder.getDenseI64ArrayAttr(convertedTarget));
    }

    /* llvm::errs() << "END APPLY DACAPO \n"; */
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<arith::ArithDialect>();
  }
};
} // namespace
