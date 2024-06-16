
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "hecate/Dialect/Earth/Transforms/Common.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include <random>

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_COVERAGERECORDER
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate

using namespace mlir;

#define DEBUG_TYPE "dacapo"

namespace {
/// Pass to bufferize Arith ops.
struct CoverageRecorderPass
    : public hecate::earth::impl::CoverageRecorderBase<CoverageRecorderPass> {
  CoverageRecorderPass() {}
  CoverageRecorderPass(hecate::earth::CoverageRecorderOptions ops) {
    this->waterline = ops.waterline;
    this->threshold = ops.threshold;
  }

  void runOnOperation() override {

    /* llvm::errs() << __FILE__ << " : " << __LINE__ << '\n'; */
    auto func = getOperation();
    auto dup = func.clone();
    auto &&from = dup->getAttrOfType<mlir::DenseI64ArrayAttr>("cutted_edge")
                      .asArrayRef()
                      .front();
    bool is_packed = false;

    if (func->hasAttr("is_packed")) {
      is_packed = func->getAttrOfType<mlir::BoolAttr>("is_packed").getValue();
    }

    auto &&block = dup.getBody().front();
    auto &&operations = block.getOperations();
    mlir::OpBuilder builder(dup);

    // Set function argument types
    auto &&inputType_attrs = dup->getAttr("segment_inputType")
                                 .dyn_cast<mlir::ArrayAttr>()
                                 .getValue();
    /* llvm::errs() << "FROM : " << from << '\n'; */
    /* llvm::errs() << "func size: " << func.getNumArguments() << '\n'; */
    /* llvm::errs() << "input size: " << inputType_attrs.size() << '\n'; */
    for (size_t i = 0; i < dup.getNumArguments(); i++) {
      auto argval = dup.getArgument(i);
      auto input_type = inputType_attrs[i]
                            .dyn_cast<mlir::TypeAttr>()
                            .getValue()
                            .dyn_cast<hecate::earth::HEScaleTypeInterface>();
      argval.setType(input_type);
    }

    // Calculate the coverages
    int64_t coverage = -1, bootCoverage = -1;
    for (auto &&op : operations) {
      if (auto sop = dyn_cast<hecate::earth::ForwardMgmtInterface>(op)) {
        auto opid = hecate::getIntegerAttr("opid", sop->getResult(0));
        if (!isa<hecate::earth::BootstrapOp>(sop) && opid < from)
          continue;

        if (isa<mlir::scf::ForOp>(op)) {
          if (!sop.isValidated()) {
            coverage = bootCoverage;
            break;
          }
        }
        if (is_packed && isa<hecate::earth::PackOp>(op)) {
          if (!sop.isValidated()) {
            bootCoverage = coverage;
            break;
          }
        }

        // PARS Scale Management
        builder.setInsertionPointAfter(sop.getOperation());
        sop.processOperandsPARS(waterline);
        inferTypeForward(sop);
        sop.processResultsPARS(waterline);
        /////////////////////////////////////////
        // Find Bootstrapping Coverage
        if (bootCoverage < 0 && !sop.isBootstrappable()) {
          bootCoverage = opid;
          if (coverage > 0 && bootCoverage > 0)
            break;
          continue;
        }
        // Find Coverage
        if (!sop.isValidated()) {
          coverage = opid;
          if (coverage > 0 && bootCoverage > 0)
            break;
        }
      }
    }

    /* llvm::errs() << "from " << from << "coverage " << coverage << " " */
    /*              << bootCoverage << '\n'; */
    /* dup.dump(); */
    dup.erase();
    func->setAttr("coverages",
                  builder.getDenseI64ArrayAttr({coverage, bootCoverage}));
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
