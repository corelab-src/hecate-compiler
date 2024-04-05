
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
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
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include <filesystem>
#include <fstream>
#include <queue>

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_LOOPROTATION
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct LoopRotationPass
    : public hecate::earth::impl::LoopRotationBase<LoopRotationPass> {
  LoopRotationPass() {}

  void bfs(mlir::SmallVector<mlir::Value, 1> inputs) {
    for (auto arg : inputs) {
      auto argDefType = arg.getDefiningOp()
                            ->getOperand(0)
                            .getType()
                            .dyn_cast<hecate::earth::HEScaleTypeInterface>();
      if (!argDefType.isCipher()) {
        arg.replaceAllUsesWith(arg.getDefiningOp()->getOperand(0));
        arg.getDefiningOp()->erase();
        continue;
      }
      std::queue<mlir::Value> values;
      values.push(arg);
      while (!values.empty()) {
        auto val = values.front();
        values.pop();
        for (auto user : val.getUsers()) {
        }
      }
    }
  }

  static bool isInnermostAffineForOp(affine::AffineForOp op) {
    return !op.getBody()
                ->walk([&](affine::AffineForOp nestedForOp) {
                  return WalkResult::interrupt();
                })
                .wasInterrupted();
  }

  static void
  gatherInnermostLoops(func::FuncOp f,
                       SmallVectorImpl<affine::AffineForOp> &loops) {
    f.walk([&](affine::AffineForOp forOp) {
      if (isInnermostAffineForOp(forOp))
        loops.push_back(forOp);
    });
  }

  void runOnOperation() override {

    auto func = getOperation();
    auto mod = func->getParentOfType<mlir::ModuleOp>();

    mlir::OpBuilder builder(func);

    Type transAnyType = transform::AnyOpType::get(builder.getContext());
    SmallVector<mlir::Type, 2> inputTypes, retTypes;
    inputTypes.push_back(transAnyType);

    PassManager pm(mod.getContext());
    OpPassManager &nestedModulePM = pm.nest<mlir::ModuleOp>();

    transform::PreloadLibraryPassOptions opts;
    std::string path = std::filesystem::path(getenv("HECATE")).string() +
                       "/examples/optimized/halo/" +
                       std::string("loop.transform.mlir");
    opts.transformLibraryPaths = path;
    pm.addPass(transform::createPreloadLibraryPass(opts));
    pm.addPass(transform::createInterpreterPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<func::FuncOp>(hecate::earth::createRemoveErasedOp());
    pm.addNestedPass<func::FuncOp>(hecate::earth::createPrivatizeConstant());

    if (failed(pm.run(mod))) {
      llvm::errs() << "loop transform failed" << '\n';
      func.dump();
    }
    func.walk([&](scf::ForOp ForOp) {
      mlir::Region &loop_body = ForOp.getBodyRegion();
      mlir::Block *loopBodyBlock = ForOp.getBody();
      mlir::IRRewriter rewriter(builder);
      SmallVector<Operation *, 8> toRotate;
      for (auto args : loopBodyBlock->getArguments()) {
        if (llvm::all_of(args.getUsers(), [&](auto &&v) {
              return hecate::earth::getScaleType(v->getResult(0)).isCipher();
            })) {
        }
      }
    });
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<transform::TransformDialect>();
    registry.insert<affine::AffineDialect>();
    registry.insert<arith::ArithDialect>();
    /* addExtensions<transform::SCFTransformDialectExtension>(registry); */
    /* registry.addExtensions<transform::SCFTransformDialectExtension>(); */
    /* transform::registerTransformDialectExtension(registry); */
    scf::registerTransformDialectExtension(registry);
  }
};
} // namespace
