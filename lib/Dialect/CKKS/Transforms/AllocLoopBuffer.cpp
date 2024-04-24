

#include <map>

#include "hecate/Dialect/CKKS/IR/CKKSOps.h"
#include "hecate/Dialect/CKKS/IR/PolyTypeInterface.h"
#include "hecate/Dialect/CKKS/Transforms/Passes.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
/* #include "mlir/Interfaces/DestinationStyleOpInterface.h" */
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace hecate {
namespace ckks {
#define GEN_PASS_DEF_ALLOCLOOPBUFFER
#include "hecate/Dialect/CKKS/Transforms/Passes.h.inc"
} // namespace ckks
} // namespace hecate

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct AllocLoopBufferPass
    : public hecate::ckks::impl::AllocLoopBufferBase<AllocLoopBufferPass> {
  AllocLoopBufferPass() {}

  void runOnOperation() override {
    auto &&func = getOperation();
    mlir::OpBuilder builder(func);
    mlir::Liveness l(func);
    func.walk([&](scf::ForOp fop) {
      SmallVector<Value, 4> garbage;
      auto loop = dyn_cast<LoopLikeOpInterface>(fop.getOperation());
      auto &&bb = fop.getBody();
      for (auto iter = bb->begin(); iter != bb->end(); ++iter) {
        Operation *oop = &*iter;
        if (auto op = dyn_cast<mlir::DestinationStyleOpInterface>(oop)) {
          for (int i = 0; i < op.getNumDpsInputs(); i++) {
            auto v = op.getDpsInputOperand(i);
            if (auto tt = hecate::ckks::getPolyType(v->get())) {
              if (tt.getNumPoly() == 1)
                continue;
              if (l.isDeadAfter(v->get(), op) &&
                  (!loop.isDefinedOutsideOfLoop(v->get())) &&
                  (!isa<BlockArgument>(v->get())) &&
                  (garbage.empty() || v->get() != garbage.back())) {
                garbage.push_back(v->get());
              }
            }
          }
          for (int i = 0; i < op.getNumDpsInits(); i++) {
            auto v = op.getDpsInitOperand(i);
            if (auto tt = hecate::ckks::getPolyType(v->get())) {
              if (tt.getNumPoly() == 1)
                continue;
              if (!garbage.empty()) {
                op.getDpsInitOperand(i)->set(garbage.pop_back_val());
              }
            }
          }
        }
      }
    });
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::ckks::CKKSDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    /* registry.addExtension(+[](MLIRContext *ctx, scf::SCFDialect *dialect) {
     */
    /*   scf::ForOp::attachInterface<mlir::DestinationStyleOpInterface>(*ctx);
     */
    /* }); */
  }
};
} // namespace
