
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
    this->be_unroll = ops.be_unroll;
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto mod_func = func->getParentOfType<mlir::ModuleOp>();
    /* llvm::errs() << "enter APPLY DACAPO \n"; */
    auto &ca = getAnalysis<hecate::CandidateAnalysis>();
    /* func.dump(); */
    /* func.dump(); */

    llvm::errs() << __FILE__ << " : " << __LINE__ << '\n';
    mlir::OpBuilder builder(func);
    mlir::IRRewriter rewriter(builder);

    auto mod = mlir::ModuleOp::create(func.getLoc());
    PassManager pm_dacapo(mod.getContext()), pm_nested(mod.getContext());
    pm_dacapo.addNestedPass<func::FuncOp>(
        hecate::earth::createBypassDetection());
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
    if (be_unroll)
      pm_dacapo.addNestedPass<func::FuncOp>(
          hecate::earth::createUnrollFactorAnalysis());

    pm_nested.addNestedPass<func::FuncOp>(
        hecate::earth::createCodeSegmentation());
    pm_nested.addNestedPass<func::FuncOp>(
        hecate::earth::createPromoteLoopBody());
    pm_nested.addNestedPass<func::FuncOp>(
        hecate::earth::createApplyDaCapoToLoop(
            {waterline, output_val, be_unroll}));

    PassManager pm_boot(mod_func.getContext());
    pm_boot.addNestedPass<func::FuncOp>(
        hecate::earth::createBootstrapPlacement());

    // {from, to}, {inputType, returnType, bypassTypes}
    SmallVector<int64_t, 2> btp_target;
    // opid, {is_packed,  inputType}
    DenseMap<int64_t, std::tuple<uint64_t, SmallVector<Type, 4>, scf::ForOp>>
        forOpTable;

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

        bool is_packed =
            forOp->getAttrOfType<mlir::BoolAttr>("is_packed").getValue();
        forOpTable[forOpid] = {is_packed, {}, forOp};
        loop_offset += forOp.getBody()->getOperations().size();
        forOp.getBody()->clear();
        // generate the YieldOp for ForOp Type Checking
        builder.setInsertionPointToStart(forOp.getBody());

        builder.create<mlir::scf::YieldOp>(forOp.getLoc(),
                                           forOp.getRegionIterArgs());
      }
    }

    dup->setAttr("is_mid_segment", builder.getBoolAttr(false));
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

    if (dup->hasAttr("unroll_factor")) {
      auto &&unroll_factor =
          dup->getAttrOfType<mlir::IntegerAttr>("unroll_factor").getInt();
      func->setAttr("unroll_factor", builder.getI64IntegerAttr(unroll_factor));
    }
    for (auto targetForOp : forOpTable) {
      auto forOpid = targetForOp.getFirst();
      auto values = hecate::earth::getOpidToValueMap(&dup.getRegion().front());

      SmallVector<Type, 4> inputTypes(initialTypes);
      llvm::errs() << __FILE__ << " : " << __LINE__ << '\n';
      for (auto inputId : ca.getValueInfo(forOpid - 1)->getLiveOuts()) {
        /* values[inputId].dump(); */
        inputTypes.push_back(values[inputId].getType());
      }
      llvm::errs() << __FILE__ << " : " << __LINE__ << '\n';
      forOpTable[forOpid] = {std::get<0>(targetForOp.getSecond()), inputTypes,
                             std::get<2>(targetForOp.getSecond())};
    }

    dup.erase();

    for (auto targetForOp : forOpTable) {
      auto ddup = func.clone();
      auto forOpid = targetForOp.getFirst();
      auto is_packed = std::get<0>(targetForOp.getSecond());
      auto &&inputType = std::get<1>(targetForOp.getSecond());
      /* auto &&forOp = std::get<2>(targetForOp.getSecond()); */

      ddup->setAttr("cutted_edge",
                    builder.getDenseI64ArrayAttr({forOpid - 1, forOpid}));
      ddup->setAttr("segment_input",
                    builder.getDenseI64ArrayAttr(
                        ca.getValueInfo(forOpid - 1)->getLiveOuts()));
      ddup->setAttr("segment_inputType", builder.getTypeArrayAttr(inputType));
      ddup->setAttr("is_mid_section", builder.getBoolAttr(true));
      ddup->setAttr("be_unroll", builder.getBoolAttr(be_unroll));

      ddup->setAttr("segment_return",
                    builder.getDenseI64ArrayAttr(
                        ca.getValueInfo(forOpid)->getLiveOuts()));
      mod.push_back(ddup);
      /* llvm::errs() << "INPUT TYPES\n"; */
      /* for (auto tt : inputType) */
      /*   llvm::errs() << tt << '\n'; */
      /* llvm::errs() << "BEFORE PROCESS FOROP : " << forOpid << '\n'; */
      llvm::errs() << __FILE__ << " : " << __LINE__ << '\n';
      if (failed(pm_nested.run(mod))) {
        llvm::errs() << "Apply DaCapo failed" << '\n';
      }
      llvm::errs() << __FILE__ << " : " << __LINE__ << '\n';

      /* llvm::errs() << "AFTER PROCESS FOROP : " << forOpid << '\n'; */
      /* ddup.dump(); */
      segment_btp_target =
          ddup->getAttrOfType<mlir::DenseI64ArrayAttr>("btp_target")
              .asArrayRef();
      btp_target.append(segment_btp_target.begin(), segment_btp_target.end());

      if (ddup->hasAttr("unroll_factor")) {
        auto &&unroll_factor =
            ddup->getAttrOfType<mlir::IntegerAttr>("unroll_factor").getInt();
        auto forOp = dyn_cast<mlir::scf::ForOp>(
            ca.getValueInfo(forOpid)->getValue().getDefiningOp());
        forOp->setAttr("unroll_factor",
                       builder.getI64IntegerAttr(unroll_factor));
        SmallVector<int64_t, 2> innerLoopUnrollFactor;
        ddup.walk([&](mlir::scf::ForOp fop) {
          auto &&inner_unroll_factor =
              fop->getAttrOfType<mlir::IntegerAttr>("unroll_factor").getInt();
          innerLoopUnrollFactor.push_back(inner_unroll_factor);
        });
        int cnt = 0;
        forOp.getBody()->walk([&](mlir::scf::ForOp fop) {
          auto &&inner_unroll_factor = innerLoopUnrollFactor[cnt++];
          fop->setAttr("unroll_factor",
                       builder.getI64IntegerAttr(inner_unroll_factor));
        });
      }

      llvm::errs() << __FILE__ << " : " << __LINE__ << '\n';
      /* forOp->setAttr("unroll_factor",
       * builder.getI64IntegerAttr(unroll_factor)); */

      ddup.erase();
    }

    llvm::errs() << __FILE__ << " : " << __LINE__ << '\n';
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
    /* llvm::errs() << __FILE__ << " : " << __LINE__ << '\n'; */
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<arith::ArithDialect>();
  }
};
} // namespace
