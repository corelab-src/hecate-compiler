
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "hecate/Support/Support.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include <fstream>
#include <zlib.h>

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

    // Compressed Code Should be Updated to Support/ConstData.cpp
    // llvm::errs() << name << "\n";
    // std::ofstream of(name, std::ios::binary);
    // int64_t a;
    // a = save_data.size();
    // of.write((char *)&a, sizeof(int64_t));
    // SmallVector<double> serializedData;
    // for (auto k : save_data) {
    //   a = k.size();
    //   of.write((char *)&a, sizeof(int64_t));
    //   for (auto d : k) {
    //     serializedData.push_back(d);
    //   }
    // }
    // // compress the constant data
    // uLongf compressedSize =
    //     compressBound(serializedData.size() * sizeof(double));
    // std::vector<Bytef> buffer(compressedSize);
    // compress(buffer.data(), &compressedSize, (Bytef *)(serializedData.data()),
    //          serializedData.size() * sizeof(double));
    // of.write((char *)(buffer.data()), compressedSize);

    // of.close();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
