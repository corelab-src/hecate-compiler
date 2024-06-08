
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
#include "mlir/IR/Value.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include <filesystem>
#include <fstream>
#include <queue>

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_WRAPUPLOOPBODY
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct WrapUpLoopBodyPass
    : public hecate::earth::impl::WrapUpLoopBodyBase<WrapUpLoopBodyPass> {
  WrapUpLoopBodyPass() {}
  WrapUpLoopBodyPass(hecate::earth::WrapUpLoopBodyOptions ops) {
    this->waterline = ops.waterline;
    this->output_val = ops.output_val;
  }

  void runOnOperation() override {
    llvm::errs() << __FILE__ << " : " << __LINE__ << '\n';
    auto func = getOperation();
    markAnalysesPreserved<hecate::CandidateAnalysis>();
    auto &ca = getAnalysis<hecate::CandidateAnalysis>();
    auto values = hecate::earth::getOpidToValueMap(&func.getRegion().front());

    mlir::OpBuilder builder(func);
    mlir::IRRewriter rewriter(builder);

    for (auto bechanged : ca.getIdMap()) {
      auto oldId = bechanged.first;
      auto newId = bechanged.second;
      SmallVector<int64_t, 2> convertedTarget;
      auto &&val = values[oldId];
      hecate::setIntegerAttr("opid", val, newId);
    }
    llvm::errs() << __FILE__ << " : " << __LINE__ << '\n';
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<arith::ArithDialect>();
  }
};
} // namespace
