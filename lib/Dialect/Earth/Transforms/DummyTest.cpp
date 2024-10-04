
#include "hecate/Dialect/Earth/Tools/GenPoly.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Common.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/Debug.h"
#include <fstream>
#include <random>

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_DUMMYTEST
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate
  //
#define DEBUG_TYPE "dacapo"

using namespace mlir;

namespace {
/// Dummy Test Pass for testing functions.
struct DummyTestPass
    : public hecate::earth::impl::DummyTestBase<
          DummyTestPass> {
  DummyTestPass() {}
  void runOnOperation() override {
    auto func = getOperation();
    mlir::OpBuilder builder(func);
    mlir::IRRewriter rewriter(builder);
    /*
    auto &poly_test = getAnalysis<hecate::PolynomialAnalysis>();
    poly_test.GenPoly_Test();
    poly_test.GSBS();
    */
    hecate::PolynomialAnalysis poly_test;
    poly_test.GenPoly_Test();
    poly_test.GSBS();
    hecate::PolynomialAnalysis poly_test2;
    poly_test2.GenPoly_Test(13);
    poly_test2.GSBS();
    poly_test.GSBS();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
