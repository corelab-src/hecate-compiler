#include "frontend.h"

#define DEBUG 0
using namespace mlir;
namespace hecate {
extern "C" {
loopID createLoop(Context *ctxt, size_t *rng, valueID *indvar, valueID *inputs,
                  size_t len, size_t num_elements, char *filename,
                  size_t line) {
  auto &&builder = *ctxt->builder;
  auto location =
      mlir::FileLineColLoc::get(builder.getStringAttr(filename), line, 0);
  mlir::Type erasedTy =
      RankedTensorType::get(llvm::SmallVector<int64_t, 1>{1},
                            builder.getType<hecate::earth::ErasedType>(0, 0));

  // Set initial inputs for the loop
  llvm::SmallVector<mlir::Value> inputarr;
  for (size_t i = 0; i < len; i++) {
    auto v = ctxt->valueMap[inputs[i]];
    auto castOp = builder.create<hecate::earth::CastOp>(location, erasedTy, v);
#if DEBUG
    v.dump();
    castOp.dump();
#endif
    inputarr.push_back(castOp);
    ctxt->valueMap.push_back(castOp);
  }

  // rng = {lower, upper, step}
  auto lowerBound =
      builder.create<mlir::arith::ConstantIndexOp>(location, rng[0]);
  Value upperBound;
  if (isa<mlir::IndexType>((ctxt->valueMap[rng[1]]).getType()))
    upperBound = ctxt->valueMap[rng[1]];
  else
    upperBound = builder.create<mlir::arith::ConstantIndexOp>(location, rng[1]);
  auto step = builder.create<mlir::arith::ConstantIndexOp>(location, rng[2]);
  auto loop = builder.create<mlir::scf::ForOp>(location, lowerBound, upperBound,
                                               step, inputarr);
#if DEBUG
  lowerBound.dump();
  upperBound.dump();
  step.dump();
  loop.dump();
#endif
  for (auto res : loop.getResults()) {
    hecate::setIntegerAttr("num_elements", res, num_elements);
  }

  loop->setAttr("is_packed", builder.getBoolAttr(false));
  ctxt->loopMap.push_back(loop);

  builder.setInsertionPointToStart(loop.getBody());
  ctxt->valueMap.push_back(loop.getInductionVar());
  indvar[0] = ctxt->valueMap.size() - 1;

  mlir::Region &loop_body = loop.getBodyRegion();
  mlir::Block &loop_block = loop_body.front();
  for (size_t i = 0; i < len; i++) {
    auto carriedVar = loop_block.getArgument(i + 1);
    ctxt->valueMap.push_back(carriedVar);
    indvar[i + 1] = ctxt->valueMap.size() - 1;
  }
  return ctxt->loopMap.size() - 1;
}

valueID setInductionVar(Context *ctxt, loopID loopID) {
  auto &&loop = ctxt->loopMap[loopID];
  Value iv = loop.getInductionVar();
  ctxt->valueMap.push_back(iv);
  return ctxt->valueMap.size() - 1;
}

loopID createLoopParse(Context *ctxt, valueID *bounds, valueID *inputs,
                       size_t len, size_t num_elements, char *filename,
                       size_t line) {

  auto &&builder = *ctxt->builder;
  auto location =
      mlir::FileLineColLoc::get(builder.getStringAttr(filename), line, 0);
  llvm::SmallVector<mlir::Value> inputarr;
  mlir::Type erasedTy =
      RankedTensorType::get(llvm::SmallVector<int64_t, 1>{1},
                            builder.getType<hecate::earth::ErasedType>(0, 0));

  // Erase Input Types for initial inputs
  for (size_t i = 0; i < len; i++) {
    auto v = ctxt->valueMap[inputs[i]];
    // auto er = builder.create<hecate::earth::EraseTypeOp>(
    // location, ctxt->valueMap[inputs[i]]);
    auto castOp = builder.create<hecate::earth::CastOp>(location, erasedTy, v);
    inputarr.push_back(castOp);
    ctxt->valueMap.push_back(castOp);
  }

  // Set the bounds for the loop
  Value lowerBound = ctxt->valueMap[bounds[0]];
  Value upperBound = ctxt->valueMap[bounds[1]];
  Value step = ctxt->valueMap[bounds[2]];
  auto loop = builder.create<mlir::scf::ForOp>(location, lowerBound, upperBound,
                                               step, inputarr);
  for (auto res : loop.getResults()) {
    hecate::setIntegerAttr("num_elements", res, num_elements);
  }

  loop->setAttr("is_packed", builder.getBoolAttr(false));
  ctxt->loopMap.push_back(loop);
  // auto &&loopOp = ctxt->loopMap[lid];
  return ctxt->loopMap.size() - 1;
}

void initLoop(Context *ctxt, loopID lid, valueID *block_args) {
  auto &&loopOp = ctxt->loopMap[lid];
  ctxt->builder->setInsertionPointToStart(loopOp.getBody());
  mlir::Region &loopBodyRegion = loopOp.getBodyRegion();
  mlir::Block &loopBodyBlock = loopBodyRegion.front();
  {
    // Induction variable poisitioned at 0, so we start from i + 1
    for (size_t i = 0; i < loopBodyBlock.getNumArguments(); i++) {
      auto carriedVar = loopBodyBlock.getArgument(i + 1);
      ctxt->valueMap.push_back(carriedVar);
      block_args[i] = ctxt->valueMap.size() - 1;
    }
  }
}

void setLoopCarriedVars(Context *ctxt, valueID loopID, valueID *arg, size_t len,
                        char *filename, size_t line) {
  auto &&builder = *ctxt->builder;
  auto &&rewriter = *ctxt->rewriter;
  auto &&loop = ctxt->loopMap[loopID];
  auto location =
      mlir::FileLineColLoc::get(builder.getStringAttr(filename), line, 0);
  mlir::Region &loop_body = loop.getBodyRegion();
  mlir::Block &loop_block = loop_body.front();

  // set loop argument
  SmallPtrSet<mlir::Operation *, 2> excepted;
  for (size_t i = 0; i < len; i++) {
    auto v = ctxt->valueMap[arg[i]];
    auto carriedVar = loop_block.getArgument(i + 1);
    for (auto use : v.getUsers()) {
      if (loop.isDefinedOutsideOfLoop(use->getResult(0))) {
        excepted.insert(use);
      }
    }
    v.replaceAllUsesExcept(carriedVar, excepted);
  }
}

void setYield(Context *ctxt, valueID loopID, valueID *ret, size_t len,
              char *filename, size_t line) {
  auto &&loop = ctxt->loopMap[loopID];
  auto &&builder = *ctxt->builder;
  auto location =
      mlir::FileLineColLoc::get(builder.getStringAttr(filename), line, 0);
  mlir::Region &loop_body = loop.getBodyRegion();
  mlir::Block &loop_block = loop_body.front();
  llvm::SmallVector<mlir::Value> rets;
  mlir::Type erasedTy =
      RankedTensorType::get(llvm::SmallVector<int64_t, 1>{1},
                            builder.getType<hecate::earth::ErasedType>(0, 0));

  // Set Induction Variable
  Value iv = loop.getInductionVar();
  for (size_t i = 0; i < len; i++) {
    auto v = ctxt->valueMap[ret[i]];
    auto castOp = builder.create<hecate::earth::CastOp>(location, erasedTy, v);
#if DEBUG
    v.dump();
    castOp.dump();
#endif
    rets.push_back(castOp);
  }
  auto loopYield = builder.create<mlir::scf::YieldOp>(location, rets);
#if DEBUG
  loopYield.dump();
#endif

  for (size_t i = 0; i < len; i++) {
    ctxt->valueMap.push_back(loop->getResult(i));
    ret[i] = ctxt->valueMap.size() - 1;
  }
  builder.setInsertionPointToEnd(loop->getBlock());
}
}
} // namespace hecate
