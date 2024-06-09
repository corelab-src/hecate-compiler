
#include "hecate/Dialect/Earth/Analysis/CandidateAnalysis.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include <fstream>

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_CANDIDATESELECTION
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate

using namespace mlir;
#define DEBUG_TYPE "Debug"

namespace {
/// Pass to bufferize Arith ops.
struct CandidateSelectionPass
    : public hecate::earth::impl::CandidateSelectionBase<
          CandidateSelectionPass> {
  CandidateSelectionPass() {}
  CandidateSelectionPass(hecate::earth::CandidateSelectionOptions ops) {
    this->waterline = ops.waterline;
    this->output_val = ops.output_val;
  }

  void runOnOperation() override {

    /* llvm::errs() << __FILE__ << " : " << __LINE__ << '\n'; */
    auto func = getOperation();
    auto &ca = getAnalysis<hecate::CandidateAnalysis>();

    // Organize the validLiveOuts
    for (auto a : ca.getEdges()) {
      auto v = ca.getValueInfo(a);
      mlir::SmallVector<int64_t, 4> validTargets;
      for (auto bp : v->getLiveOuts()) {
        auto vp = ca.getValueInfo(bp);
        if (!vp->isBypassEdge(a)) {
          validTargets.push_back(bp);
        }
      }
      v->setValidLiveOuts(validTargets);
      ca.sortValidCandidates(a);
    }
    /*
    for (auto a : ca.getEdges()) {
      auto v = ca.getValueInfo(a);
      llvm::errs() << a << " : ";
      for (auto vl : v->getValidLiveOuts()) {
        llvm::errs() << vl << " ";
      }
      llvm::errs() << '\n';
    }
    */

    mlir::OpBuilder builder(func);
    auto mod = mlir::ModuleOp::create(func.getLoc());
    PassManager pm(mod.getContext());
    pm.addNestedPass<func::FuncOp>(hecate::earth::createBootstrapPlacement());
    pm.addNestedPass<func::FuncOp>(
        hecate::earth::createProactiveRescaling({waterline, output_val}));

    for (size_t i = 1; i <= ca.getMaxNumOuts(); i++) {
      auto dup = func.clone();
      mlir::OpBuilder builder(dup);
      dup->setAttr("btp_target",
                   builder.getDenseI64ArrayAttr(ca.sortTargets(i)));
      mod.push_back(dup);
      if (pm.run(mod).succeeded()) {
        func->setAttr("selected_set", builder.getI64IntegerAttr(i));
        ca.finalizeCandidates(i);
        /* llvm::errs() << "selected _set : " << i << '\n'; */
        dup.erase();
        break;
      }
      dup.erase();
    }
    markAnalysesPreserved<hecate::CandidateAnalysis>();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
