
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "hecate/Support/Support.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include <fstream>

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_ELIDECONSTANT
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct ElideConstantPass
    : public hecate::earth::impl::ElideConstantBase<ElideConstantPass> {
  ElideConstantPass() {}
  ElideConstantPass(hecate::earth::ElideConstantOptions ops) {
    this->name = ops.name;
  }

  void runOnOperation() override {
    auto func = getOperation();

    auto num_constants = 0;
    if (auto attr = func->getAttr("num_constants")) {
      num_constants = attr.dyn_cast<mlir::IntegerAttr>().getInt();
    }
    hecate::ConstData constData;

    SmallVector<SmallVector<double, 4>, 4> save_data;

    mlir::OpBuilder builder(func.getOperation());

    func.walk([&](hecate::earth::ConstantOp cop) {
      if (auto datas = cop.getValue().dyn_cast<DenseElementsAttr>()) {
        constData.push_back(std::vector<double>(datas.value_begin<double>(),
                                                datas.value_end<double>()));
        cop.setValueAttr(
            builder.getI64IntegerAttr(constData.size() + num_constants - 1));
      }
    });

    name = name + (func.getName() + ".cst").str();
    constData.save(name, num_constants);

    func->setAttr("num_constants",
                  builder.getI64IntegerAttr(num_constants + save_data.size()));
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
