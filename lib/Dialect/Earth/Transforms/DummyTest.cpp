
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
#include <iostream>

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
    
    hecate::GenPoly poly_test;
    //hecate::GenPoly poly_test("treeStr.txt", "coeffStr.txt", 8, 1.0);
    //poly_test.GenPoly();
    
    std::vector<double> inputs;
    inputs.push_back(0.1);
    inputs.push_back(0.2);
    inputs.push_back(0.3);
    std::vector<double> outputs;
    outputs = poly_test.GSBS_check(inputs);
    
    std::cout<<"Done"<<std::endl;
    
    auto &&block = func.getBody().front();
    auto &&operations = block.getOperations();
    for(auto &&op : operations) {
      // Temporarily convert mulop operation(0) to creating GSBS for test
      if (auto mop = dyn_cast<hecate::earth::MulOp>(op)) {
        builder.setInsertionPoint(mop);
        auto newop = poly_test.GSBS_createHEOps(builder, mop.getLoc(), mop.getOperand(0));
        mop->setOperand(0, newop);
      }
    }
    /*
    func.walk([&](hecate::earth::MulOp op) {
      op->getLoc().dump();
      poly_test.GSBS_createHEOps(builder, op->getLoc(), op.getResult());
    });
    */
    std::cout<<"Done"<<std::endl;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
