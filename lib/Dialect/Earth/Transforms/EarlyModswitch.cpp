
//===- Bufferize.cpp - Bufferization for Arith ops ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hecate/Dialect/Earth/Analysis/CandidateAnalysis.h"
#include "hecate/Dialect/Earth/Analysis/ScaleManagementUnit.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_EARLYMODSWITCH
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate
  //
#define DEBUG_TYPE "hecate_em"

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct EarlyModswitchPass
    : public hecate::earth::impl::EarlyModswitchBase<EarlyModswitchPass> {
  EarlyModswitchPass() {}

  void runOnOperation() override {
    /* llvm::errs() << __FILE__ << " : " << __LINE__ << '\n'; */
    auto func = getOperation();
    markAnalysesPreserved<hecate::ScaleManagementUnit>();
    markAnalysesPreserved<hecate::CandidateAnalysis>();

    mlir::OpBuilder builder(func);
    mlir::IRRewriter rewriter(builder);

    SmallVector<mlir::Type, 4> inputTypes;
    auto &&bb = func.getBody().getBlocks().front();
    for (auto iter = bb.rbegin(); iter != bb.rend(); ++iter) {
      if (auto op = dyn_cast<hecate::earth::HEScaleOpInterface>(*iter)) {
        applyEarlyModswitch(func, op);
      } else if (auto forOp = dyn_cast<mlir::scf::ForOp>(*iter)) {
        applyEarlyModswitchToLoop(func, forOp);
      }
    }
    /* llvm::errs() << __FILE__ << " : " << __LINE__ << '\n'; */
  }

  void applyEarlyModswitch(func::FuncOp func,
                           hecate::earth::HEScaleOpInterface &op) {

    mlir::OpBuilder builder(func);
    mlir::IRRewriter rewriter(builder);
    // Skip the bootstrapping operation
    if (auto oop = dyn_cast<hecate::earth::BootstrapOp>(op.getOperation())) {
      builder.setInsertionPoint(oop);
      auto v = oop->getOperand(0);
      auto level_diff = hecate::earth::EarthDialect::bootstrapLevelUpperBound -
                        hecate::earth::EarthDialect::bootstrapLevelLowerBound -
                        op.getOperandLevel(0);
      oop->setOperand(0, builder.create<hecate::earth::ModswitchOp>(
                             op->getLoc(), v, level_diff));
      return; // Go to next operation
    }

    // Gather the users and finds the minimum "downFactor"
    /* for (auto res : op->getResults()) { */
    uint64_t minModFactor = -1;
    for (auto res : op->getResults()) {
      for (auto &&oper : res.getUses()) {
        if (auto oop = dyn_cast<hecate::earth::ModswitchOp>(oper.getOwner())) {
          minModFactor = std::min(minModFactor, oop.getDownFactor());
        } else {
          minModFactor = 0;
        }
      }

      // Check that every user needs the "downFactor"ed level
      if (!minModFactor) {
        continue; // Go to next operation
      }
    }

    // Move the modswitch
    if (auto oop = dyn_cast<hecate::earth::ModswitchOp>(op.getOperation())) {
      // Modswitch movement can be absorbed into modswitch
      oop.setDownFactor(oop.getDownFactor() + minModFactor);
      auto tt = oop.getType().dyn_cast<RankedTensorType>();
      oop.getResult().setType(RankedTensorType::get(
          tt.getShape(),
          tt.getElementType()
              .dyn_cast<hecate::earth::HEScaleTypeInterface>()
              .switchLevel(op.getRescaleLevel() + minModFactor)));
    } else {
      // Modswitch is moved to the operands
      for (int i = 0; i < op->getNumOperands(); i++) {
        auto oper = op->getOperand(i);
        builder.setInsertionPoint(op);
        auto newOper = builder.create<hecate::earth::ModswitchOp>(
            op->getLoc(), oper, minModFactor);
        op->setOperand(i, newOper);
      }
      for (auto res : op->getResults()) {
        res.setType(hecate::earth::getScaleType(res).switchLevel(
            hecate::earth::getScaleType(res).getLevel() + minModFactor));
      }
    }

    // Change the user modswitch downFactors
    for (auto res : op->getResults()) {
      for (auto &&oper : res.getUsers()) {
        if (auto oop = dyn_cast<hecate::earth::ModswitchOp>(oper)) {
          oop.setDownFactor(oop.getDownFactor() - minModFactor);
        }
      }
    }
  }

  void applyEarlyModswitchToLoop(func::FuncOp func, mlir::scf::ForOp &forOp) {

    mlir::OpBuilder builder(func);
    mlir::IRRewriter rewriter(builder);
    auto &&forBody = forOp.getBody();
    for (auto iiter = forBody->rbegin(); iiter != forBody->rend(); ++iiter) {
      if (auto op = dyn_cast<hecate::earth::HEScaleOpInterface>(*iiter)) {
        applyEarlyModswitch(func, op);
      } else if (auto forOp = dyn_cast<mlir::scf::ForOp>(*iiter)) {
        applyEarlyModswitchToLoop(func, forOp);
      }
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
    registry.insert<scf::SCFDialect>();
  }
};
} // namespace
