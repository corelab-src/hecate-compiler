
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

  mlir::SmallVector<std::pair<int64_t, int64_t>, 4> cuttedEdges;

  mlir::SmallVector<scf::ForOp, 4> scfForQueue;
  void scheduleForOp(mlir::scf::ForOp &op) {
    auto &&bb = op.getBody();
    for (auto iter = bb->begin(); iter != bb->end(); ++iter) {
      if (auto forOp = dyn_cast<mlir::scf::ForOp>(*iter)) {
        scheduleForOp(forOp);
      }
    }
    scfForQueue.push_back(op);
    return;
  }

  void decideCuttedEdges(mlir::Block *bb, int64_t &startp, int64_t &foropp) {
    for (auto iter = bb->begin(); iter != bb->end(); ++iter) {
      if (auto forOp = dyn_cast<mlir::scf::ForOp>(*iter)) {
        auto &&from = hecate::getIntegerAttr("opid", forOp.getResult(0));
        cuttedEdges.push_back({startp, from - 1});
        cuttedEdges.push_back({from - 1, from});
        startp = from;
        /* auto before_foropp = foropp; */
        /* foropp = from; */
        /* decideCuttedEdges(forOp.getBody(), startp, foropp); */
        /* foropp = before_foropp; */
      } else if (auto yop = dyn_cast<mlir::scf::YieldOp>(*iter)) {
        cuttedEdges.push_back({startp, foropp});
        startp = foropp;
      }
    }

    return;
  }

  void findForOp(mlir::Block *bb) {
    for (auto iter = bb->begin(); iter != bb->end(); ++iter) {
      if (auto forOp = dyn_cast<mlir::scf::ForOp>(*iter)) {
        auto &&from = hecate::getIntegerAttr("opid", forOp.getResult(0));
        cuttedEdges.push_back({from - 1, from});
      }
    }
    return;
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto mod_func = func->getParentOfType<mlir::ModuleOp>();
    llvm::errs() << "enter APPLY DACAPO \n";
    auto &ca = getAnalysis<hecate::CandidateAnalysis>();
    func.dump();

    mlir::OpBuilder builder(func);
    mlir::IRRewriter rewriter(builder);

    auto mod = mlir::ModuleOp::create(func.getLoc());
    PassManager pm_dacapo(mod.getContext()), pm_nested(mod.getContext());
    auto rescalingFactor = hecate::earth::EarthDialect::rescalingFactor;
    /* pm_dacapo.addNestedPass<func::FuncOp>( */
    /*     hecate::earth::createCodeSegmentation()); */
    pm_dacapo.addNestedPass<func::FuncOp>(
        hecate::earth::createBypassDetection({waterline, 0.5}));
    pm_dacapo.addNestedPass<func::FuncOp>(
        hecate::earth::createCandidateSelection({waterline, output_val}));
    pm_dacapo.addNestedPass<func::FuncOp>(
        hecate::earth::createDaCapoPlanner({waterline, output_val}));
    pm_dacapo.addNestedPass<func::FuncOp>(
        hecate::earth::createBootstrapPlacement());
    pm_dacapo.addNestedPass<func::FuncOp>(
        hecate::earth::createProactiveRescaling({waterline, output_val}));
    pm_dacapo.addNestedPass<func::FuncOp>(
        hecate::earth::createEarlyModswitch());
    pm_dacapo.addNestedPass<func::FuncOp>(
        hecate::earth::createWrapUpLoopBody());

    pm_nested.addNestedPass<func::FuncOp>(
        hecate::earth::createCodeSegmentation());
    pm_nested.addNestedPass<func::FuncOp>(
        hecate::earth::createPromoteLoopBody());
    pm_nested.addNestedPass<func::FuncOp>(
        hecate::earth::createApplyDaCapoToLoop());

    PassManager pm_boot(mod_func.getContext()),
        pm_bypass(mod_func.getContext());
    pm_boot.addNestedPass<func::FuncOp>(
        hecate::earth::createBootstrapPlacement());

    pm_bypass.addNestedPass<func::FuncOp>(
        hecate::earth::createBypassDetection({waterline, 0.5}));

    llvm::errs() << "CUTTED EDGES" << '\n';
    for (auto tt : cuttedEdges)
      llvm::errs() << tt.first << "  " << tt.second << '\n';
    llvm::errs() << '\n';

    // {from, to}, {inputType, returnType, bypassTypes}
    SmallVector<int64_t, 2> btp_target;
    DenseMap<int64_t, std::tuple<SmallVector<Type, 4>, SmallVector<Type, 4>>>
        edgeTypes;
    // opid, {offset,  inputType}
    DenseMap<int64_t, std::tuple<uint64_t, SmallVector<Type, 4>>> forOpTable;

    SmallVector<Type, 4> initialTypes;
    /* func.dump(); */
    initialTypes =
        hecate::earth::getInputValueTypes(func, builder, waterline, output_val);

    if (failed(pm_bypass.run(mod_func))) {
      llvm::errs() << "Apply DaCapo failed" << '\n';
    }

    /* edgeTypes[0] = {{}, {}}; */
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
    llvm::errs() << __FILE__ << " : " << __LINE__ << '\n';

    mod.push_back(dup);
    if (failed(pm_dacapo.run(mod))) {
      llvm::errs() << "Apply DaCapo failed" << '\n';
    }

    llvm::errs() << "AFTER DACAPO PROCESS \n";
    dup.dump();
    auto &&segment_btp_target =
        dup->getAttrOfType<mlir::DenseI64ArrayAttr>("btp_target").asArrayRef();
    btp_target.append(segment_btp_target.begin(), segment_btp_target.end());

    for (auto targetForOp : forOpTable) {
      auto forOpid = targetForOp.getFirst();
      /* auto values = hecate::earth::attachOpid(&dup.getRegion().front()); */
      auto values = hecate::earth::getOpidToValueMap(&dup.getRegion().front());

      SmallVector<Type, 4> inputTypes(initialTypes);
      llvm::errs() << __FILE__ << " : " << __LINE__ << '\n';
      llvm::errs() << "forOpid " << forOpid << '\n';

      llvm::errs() << "FOR VALUE TYPE MAP\n";
      /* for (auto tt : values) */
      /*   llvm::errs() << tt.first << " : " << tt.second << '\n'; */
      for (auto inputId : ca.getValueInfo(forOpid - 1)->getLiveOuts()) {
        /* inputId -= offset; */
        llvm::errs() << "INPUTID : " << inputId << '\n';
        values[inputId].dump();
        inputTypes.push_back(values[inputId].getType());
      }
      llvm::errs() << __FILE__ << " : " << __LINE__ << '\n';
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
      if (failed(pm_nested.run(mod))) {
        llvm::errs() << "Apply DaCapo failed" << '\n';
      }

      llvm::errs() << "AFTER PROCESS FOROP : " << forOpid << '\n';
      ddup.dump();
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
      llvm::errs() << "CONVERT TARGET\n";
      auto idMap = ca.getIdMap();
      SmallVector<int64_t, 2> convertedTarget;
      for (auto tt : btp_target) {
        convertedTarget.push_back(idMap[tt]);
      }
      func->setAttr("btp_target",
                    builder.getDenseI64ArrayAttr(convertedTarget));
    }

    llvm::errs() << "END APPLY DACAPO \n";
    /* dup.erase(); */
    /* SmallVector<Type, 4> inputType(initialTypes); */
    /* dup->setAttr("segment_inputType", builder.getTypeArrayAttr(inputType));
     */
    /* llvm::errs() << "SEGMENT AFTER INPUTTYPE\n"; */
    /* for (auto tt : inputType) */
    /*   llvm::errs() << tt << " " << '\n'; */
  }

  void runOnOperation2() {
    auto func = getOperation();
    auto mod_func = func->getParentOfType<mlir::ModuleOp>();
    llvm::errs() << "enter APPLY DACAPO \n";
    auto &ca = getAnalysis<hecate::CandidateAnalysis>();

    mlir::OpBuilder builder(func);
    mlir::IRRewriter rewriter(builder);

    auto mod = mlir::ModuleOp::create(func.getLoc());
    PassManager pm_dacapo(mod.getContext()), pm_nested(mod.getContext());
    auto rescalingFactor = hecate::earth::EarthDialect::rescalingFactor;
    /* pm_dacapo.addNestedPass<func::FuncOp>( */
    /*     hecate::earth::createCodeSegmentation()); */
    pm_dacapo.addNestedPass<func::FuncOp>(
        hecate::earth::createBypassDetection({waterline, 0.5}));
    pm_dacapo.addNestedPass<func::FuncOp>(
        hecate::earth::createCandidateSelection({waterline, output_val}));
    pm_dacapo.addNestedPass<func::FuncOp>(
        hecate::earth::createDaCapoPlanner({waterline, output_val}));

    pm_nested.addNestedPass<func::FuncOp>(
        hecate::earth::createCodeSegmentation());
    pm_nested.addNestedPass<func::FuncOp>(
        hecate::earth::createPromoteLoopBody());
    pm_nested.addNestedPass<func::FuncOp>(
        hecate::earth::createApplyDaCapoToLoop());
    pm_nested.addNestedPass<func::FuncOp>(
        hecate::earth::createWrapUpLoopBody());

    PassManager pm_boot(mod_func.getContext()),
        pm_bypass(mod_func.getContext());
    pm_boot.addNestedPass<func::FuncOp>(
        hecate::earth::createBootstrapPlacement());

    pm_bypass.addNestedPass<func::FuncOp>(
        hecate::earth::createBypassDetection({waterline, 0.5}));
    /* pm_boot.addPass(mlir::createCSEPass()); */
    /* pm_boot.addPass(mlir::createCanonicalizerPass()); */
    if (failed(pm_bypass.run(mod_func))) {
      llvm::errs() << "Apply DaCapo failed" << '\n';
    }

    int64_t startp = 0;
    int64_t foropp = -1;
    /* llvm::errs() << "START CUTTED EDGES" << '\n'; */

    decideCuttedEdges(&func.getBody().getBlocks().front(), startp, foropp);
    cuttedEdges.push_back({startp, ca.getRetOpid()});
    /* func.dump(); */

    llvm::errs() << "CUTTED EDGES" << '\n';
    for (auto tt : cuttedEdges)
      llvm::errs() << tt.first << "  " << tt.second << '\n';
    llvm::errs() << '\n';
    // {from, to}, {inputType, returnType, bypassTypes}
    SmallVector<int64_t, 2> btp_target;
    DenseMap<int64_t, std::tuple<SmallVector<Type, 4>, SmallVector<Type, 4>>>
        edgeTypes;

    SmallVector<Type, 4> initialTypes;
    initialTypes =
        hecate::earth::getInputValueTypes(func, builder, waterline, output_val);

    /* SmallVector<bool, 2> initialBypassTypes; */
    edgeTypes[0] = {{}, {}};
    llvm::errs() << "INITIAL TYPES\n";
    for (auto tt : initialTypes)
      llvm::errs() << tt << '\n';

    for (auto edge : cuttedEdges) {
      auto &&from = edge.first;
      auto &&to = edge.second;

      auto &&formerReturnType = std::get<0>(edgeTypes[from]);
      SmallVector<Type, 4> inputType(initialTypes);
      /* Smallvector<bool, 2> inputBypasses(); */

      inputType.append(formerReturnType.begin(), formerReturnType.end());
      llvm::errs() << "TTTTTT INPUTYPE\n";
      for (auto ttt : inputType)
        llvm::errs() << ttt << '\n';
      auto &&segmentInputs = ca.getValueInfo(from)->getLiveOuts();
      auto &&toValue = ca.getValueInfo(to)->getValue();
      llvm::errs() << "Processing " << from << " -> " << to << '\n';

      llvm::errs() << "BTP TARGETs\n";
      for (auto tt : ca.getTargets(from))
        llvm::errs() << tt << " " << '\n';
      llvm::errs() << "LIVEOUT\n";
      for (auto tt : ca.getValueInfo(from)->getLiveOuts())
        llvm::errs() << tt << " " << '\n';
      auto dup = func.clone();
      dup->setAttr("cutted_edge", builder.getDenseI64ArrayAttr({from, to}));
      dup->setAttr("segment_input",
                   builder.getDenseI64ArrayAttr(segmentInputs));
      auto tp = mlir::RankedTensorType::get(
          llvm::SmallVector<int64_t, 1>{1},
          builder.getType<hecate::earth::CipherType>(rescalingFactor, 0));
      auto &&btpTargets = ca.getTargets(from);
      dup->setAttr("segment_inputType", builder.getTypeArrayAttr(inputType));

      llvm::errs() << "SEGMENT AFTER INPUTTYPE\n";
      for (auto tt : inputType)
        llvm::errs() << tt << " " << '\n';

      dup->setAttr("segment_return", builder.getDenseI64ArrayAttr(
                                         ca.getValueInfo(to)->getLiveOuts()));
      if (func->hasAttr("is_mid_section")) {
        auto &&tt =
            func->getAttrOfType<mlir::BoolAttr>("is_mid_section").getValue();
        dup->setAttr("is_mid_section", builder.getBoolAttr(tt));
      } else if (to == ca.getRetOpid()) {
        dup->setAttr("is_mid_section", builder.getBoolAttr(false));
      } else {
        dup->setAttr("is_mid_section", builder.getBoolAttr(true));
      }
      auto &&is_mid_sec =
          dup->getAttrOfType<mlir::BoolAttr>("is_mid_section").getValue();

      llvm::errs() << "IS_MID_SECTION: " << is_mid_sec << '\n';

      if (!func->hasAttr("section_returnBypasses")) {
        dup->setAttr("section_returnBypasses",
                     builder.getBoolArrayAttr(ca.getBypassTypeOfLiveOuts(to)));
        llvm::errs() << "SECTION RETURN BYPASSES ";
        for (auto tt : ca.getBypassTypeOfLiveOuts(to))
          llvm::errs() << tt << " ";
        llvm::errs() << '\n';
      }
      /* llvm::SmallVector<bool, 2> bypassTypes; */
      /* bypassTypes.append(inputBypasses); */
      /* bypassTypes.append(ca.getBypassTypeOfLiveOuts(to)); */
      /* bestPlan[to] = {cost, plan, retTypes, dup.clone(), bypassTypes}; */

      SmallVector<Type, 4> tmpInputType;

      if (toValue && isa<mlir::scf::ForOp>(toValue.getDefiningOp())) {
        auto forOp = dyn_cast<mlir::scf::ForOp>(toValue.getDefiningOp());

        /* for (size_t idx = 0; idx < segmentInputs.size(); idx++) { */
        /*   size_t typeIdx = idx + initialTypes.size(); */
        /*   auto input = segmentInputs[idx]; */
        /*   auto shouldBtp = */
        /*       std::find(btpTargets.begin(), btpTargets.end(), input); */
        /*   if (shouldBtp != btpTargets.end()) { */
        /*     llvm::errs() << "CHANGED SEGMENT: " << idx << '\n'; */
        /*     llvm::errs() << "CHANGED TYPE IDX: " << typeIdx << '\n'; */
        /*     llvm::errs() << inputType[typeIdx] << " ----> " << tp << '\n'; */
        /*     inputType[typeIdx] = tp; */
        /*   } */
        /* } */

        /* dup->setAttr("is_mid_section", builder.getBoolAttr(true)); */

        /* SmallVector<bool, 2> forOpBypassTypes; */
        /* for (auto no_bypass : forOp.getInitArgs()) */
        /*   forOpBypassTypes.push_back(false); */
        /* dup->setAttr("section_returnBypasses", */
        /*              builder.getBoolArrayAttr(forOpBypassTypes)); */
        /* llvm::errs() << "FOR SECTION RETURN BYPASSES "; */
        /* for (auto tt : forOpBypassTypes) */
        /*   llvm::errs() << tt << " "; */
        /* llvm::errs() << '\n'; */

        mod.push_back(dup);
        llvm::errs() << __LINE__ << '\n';
        if (failed(pm_nested.run(mod))) {
          llvm::errs() << "Apply DaCapo failed" << '\n';
        }
        llvm::errs() << __LINE__ << '\n';
        auto &&additional_btp_target =
            dup->getAttrOfType<mlir::DenseI64ArrayAttr>("additional_btp_target")
                .asArrayRef();
        llvm::errs() << "ADDITIONAL BTP TARGETS \n";
        for (auto tt : additional_btp_target)
          llvm::errs() << tt << " ";
        llvm::errs() << "\n\n";
        btp_target.append(additional_btp_target.begin(),
                          additional_btp_target.end());

      } else {
        llvm::errs() << __LINE__ << '\n';
        mod.push_back(dup);
        if (failed(pm_dacapo.run(mod))) {
          llvm::errs() << "Apply DaCapo failed" << '\n';
        }

        llvm::errs() << __LINE__ << '\n';
      }
      llvm::errs() << __LINE__ << '\n';
      /* llvm::errs() << "=================\n"; */
      auto &&retType = std::get<0>(edgeTypes[to]);
      auto &&segment_btp_target =
          dup->getAttrOfType<mlir::DenseI64ArrayAttr>("btp_target")
              .asArrayRef();
      llvm::errs() << __LINE__ << '\n';
      auto &&segment_return_type =
          dup->getAttr("return_type").dyn_cast<mlir::ArrayAttr>().getValue();
      llvm::errs() << __LINE__ << '\n';

      /* llvm::errs() << "SEGMENT_BTP_TARGET" << '\n'; */
      /* for (auto opid : segment_btp_target) { */
      /*   llvm::errs() << opid << '\n'; */
      /* } */
      btp_target.append(segment_btp_target.begin(), segment_btp_target.end());
      llvm::errs() << __LINE__ << '\n';
      /* llvm::errs() << "SEG_RET_TYPE\n "; */
      /* std::get<0>(edgeTypes[to]) */
      /*     .append(initialTypes.begin(), initialTypes.end()); */
      for (auto ret : segment_return_type) {
        std::get<0>(edgeTypes[to])
            .push_back(ret.dyn_cast<mlir::TypeAttr>().getValue());
      }
      llvm::errs() << __LINE__ << '\n';
      /* llvm::errs() << "RETURN TYPE SIZE : " <<
       * std::get<0>(edgeTypes[to]).size() */
      /*              << '\n'; */
      dup.erase();
      // After scf loop body to segment
    }

    /* llvm::errs() << "FINAL BOOTSTRAPPING LOCATION\n"; */
    /* for (auto target : btp_target) */
    /*   llvm::errs() << target << '\n'; */
    llvm::errs() << __LINE__ << '\n';
    func->setAttr("btp_target", builder.getDenseI64ArrayAttr(btp_target));
    func->setAttr("return_type", builder.getTypeArrayAttr(
                                     std::get<0>(edgeTypes[ca.getRetOpid()])));
    /* mod.push_back(func); */
    if (failed(pm_boot.run(mod_func))) {
      llvm::errs() << "Apply DaCapo failed" << '\n';
    }

    if (ca.getIdMap().size()) {
      llvm::errs() << "CONVERT TARGET\n";
      auto idMap = ca.getIdMap();
      SmallVector<int64_t, 2> convertedTarget;
      for (auto tt : btp_target) {
        convertedTarget.push_back(idMap[tt]);
      }
      func->setAttr("btp_target",
                    builder.getDenseI64ArrayAttr(convertedTarget));
    }
    llvm::errs() << __LINE__ << '\n';

    llvm::errs() << __FILE__ << '\n';
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<arith::ArithDialect>();
  }
};
} // namespace
