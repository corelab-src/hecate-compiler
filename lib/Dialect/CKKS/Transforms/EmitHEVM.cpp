
#include "hecate/Dialect/CKKS/IR/CKKSOps.h"
#include "hecate/Dialect/CKKS/IR/PolyTypeInterface.h"
#include "hecate/Dialect/CKKS/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <filesystem>
#include <fstream>

#include "hecate/Support/HEVMHeader.h"

namespace hecate {
namespace ckks {
#define GEN_PASS_DEF_EMITHEVM
#define HELoopOpcode 11
#include "hecate/Dialect/CKKS/Transforms/Passes.h.inc"
#include <map>
#include <vector>
} // namespace ckks
} // namespace hecate

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct EmitHEVMPass : public hecate::ckks::impl::EmitHEVMBase<EmitHEVMPass> {
  EmitHEVMPass() {}
  EmitHEVMPass(hecate::ckks::EmitHEVMOptions ops) { this->prefix = ops.prefix; }
  std::vector<HEVMLoopOp> loops;
  std::map<int, std::vector<HEVMOperation>> loop_insts;

  void runOnOperation() override {
    markAllAnalysesPreserved();
    auto &&func = getOperation();
    HEVMHeader header;
    ConfigBody config_body;
    header.hevm_header_size = sizeof(HEVMHeader);
    header.config_header.arg_length = func.getNumArguments();
    header.config_header.res_length = func.getNumResults();

    config_body.config_body_length = sizeof(ConfigBody);
    SmallVector<uint64_t, 4> config_body_ints;
    std::vector<HEVMOperation> insts;

    int64_t cipher_registers = 0;
    int64_t plain_registers = 0;
    int64_t integer_registers = 0;
    /* int64_t msg_registers = 0; */
    llvm::DenseMap<mlir::Value, int64_t> cipher_register_file;
    llvm::DenseMap<mlir::Value, int64_t> plain_register_file;
    llvm::DenseMap<mlir::Value, int64_t> integer_register_file;
    /* llvm::DenseMap<mlir::Value, int64_t> msg_register_file; */
    for (auto &&arg : func.getArguments()) {
      if (auto tt = arg.getType().dyn_cast<hecate::ckks::PolyTypeInterface>()) {
        if (tt.getNumPoly() == 1) {
          plain_register_file.insert({arg, plain_registers++});
        } else {
          cipher_register_file.insert({arg, cipher_registers++});
        }
      } else if (auto tt = arg.getType().dyn_cast<mlir::IndexType>()) {
        integer_register_file.insert({arg, integer_registers++});
      }
    }

    auto &&bb = func.getBody().getBlocks().front();
    for (auto iter = bb.begin(); iter != bb.end(); ++iter) {
      generateOpsVector(&*iter, insts, cipher_registers, plain_registers,
                        integer_registers, cipher_register_file,
                        plain_register_file, integer_register_file);
    }

    SmallVector<int64_t> ret_dst;
    auto retOp =
        dyn_cast<func::ReturnOp>(func.getBlocks().front().getTerminator());
    for (auto arg : retOp.getOperands()) {
      ret_dst.push_back(cipher_register_file[arg]);
    }

    auto arg_scale_array =
        func->getAttrOfType<DenseI64ArrayAttr>("arg_scale").asArrayRef();
    auto arg_level_array =
        func->getAttrOfType<DenseI64ArrayAttr>("arg_level").asArrayRef();
    auto res_scale_array =
        func->getAttrOfType<DenseI64ArrayAttr>("res_scale").asArrayRef();
    auto res_level_array =
        func->getAttrOfType<DenseI64ArrayAttr>("res_level").asArrayRef();

    config_body.num_operations = insts.size();
    config_body.num_loops = loops.size();
    config_body.num_ctxt_buffer = cipher_registers;
    config_body.num_ptxt_buffer = plain_registers;
    config_body.num_int_buffer = integer_registers;
    config_body.init_level =
        func->getAttrOfType<IntegerAttr>("init_level").getInt();

    config_body_ints.append(arg_scale_array.begin(), arg_scale_array.end());
    config_body_ints.append(arg_level_array.begin(), arg_level_array.end());
    config_body_ints.append(res_scale_array.begin(), res_scale_array.end());
    config_body_ints.append(res_level_array.begin(), res_level_array.end());
    config_body_ints.append(ret_dst.begin(), ret_dst.end());

    config_body.config_body_length +=
        config_body_ints.size() * sizeof(uint64_t);

    std::filesystem::path printpath(prefix.getValue());

    printpath = std::string(printpath) + "." + func.getName().str() + ".hevm";

    std::ofstream of(printpath, std::ios::binary);
    of.write((char *)&header, sizeof(HEVMHeader));
    of.write((char *)&config_body, sizeof(ConfigBody));
    of.write((char *)config_body_ints.data(),
             config_body_ints.size() * sizeof(uint64_t));
    of.write((char *)insts.data(), insts.size() * sizeof(HEVMOperation));
    of.write((char *)loops.data(), loops.size() * sizeof(HEVMLoopOp));
    for (size_t i = 0; i < loops.size(); i++) {
      /* of.write((char *)&loops[i], sizeof(HEVMLoopOp)); */
      of.write((char *)loop_insts[i].data(),
               loop_insts[i].size() * sizeof(HEVMOperation));
    }

    of.close();
  }

  void
  generateOpsVector(mlir::Operation *op, std::vector<HEVMOperation> &insts,
                    int64_t &cipher_registers, int64_t &plain_registers,
                    int64_t &integer_registers,
                    llvm::DenseMap<mlir::Value, int64_t> &cipher_register_file,
                    llvm::DenseMap<mlir::Value, int64_t> &plain_register_file,
                    llvm::DenseMap<mlir::Value, int64_t> &integer_register_file

  ) {
    if (auto alloc = dyn_cast<mlir::tensor::EmptyOp>(op)) {
      auto tt = alloc.getType().dyn_cast<hecate::ckks::PolyTypeInterface>();
      HEVMOperation heops;
      heops.opcode = -1;
      insts.push_back(heops);
      if (tt.getNumPoly() == 1) {
        plain_register_file.insert({alloc, plain_registers++});
      } else {
        cipher_register_file.insert({alloc, cipher_registers++});
      }
    } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      HEVMLoopOp heop;
      std::vector<HEVMOperation> loop_ops;
      heop.config_body.lb = integer_register_file[op->getOperand(0)];
      heop.config_body.ub = integer_register_file[op->getOperand(1)];
      heop.config_body.step = integer_register_file[op->getOperand(2)];
      heop.opcode = HELoopOpcode;
      heop.dst = loops.size();
      insts.push_back(heop);
      loops.push_back(heop);

      SmallVector<Value, 2> initArgs;
      for (auto &&initArg : forOp.getInitArgs()) {
        initArgs.push_back(initArg);
      }

      integer_register_file.insert(
          {forOp.getInductionVar(), integer_registers++});
      for (size_t i = 0; i < forOp.getNumRegionIterArgs(); i++) {
        auto arg = forOp.getRegionIterArg(i);
        auto initArg = initArgs[i];
        auto tt = arg.getType().dyn_cast<hecate::ckks::PolyTypeInterface>();
        if (tt.getNumPoly() == 1) {
          plain_register_file.insert({arg, plain_register_file[initArg]});
        } else {
          cipher_register_file.insert({arg, cipher_register_file[initArg]});
        }
      }

      auto &&bb = forOp.getBody();
      for (auto iter = bb->begin(); iter != bb->end(); ++iter) {
        generateOpsVector(&*iter, loop_ops, cipher_registers, plain_registers,
                          integer_registers, cipher_register_file,
                          plain_register_file, integer_register_file);
      }

      auto yieldOp = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      for (size_t i = 0; i < yieldOp.getNumOperands(); i++) {
        auto arg = forOp.getRegionIterArg(i);
        auto yielded = yieldOp.getOperand(i);
        auto ret = forOp.getResult(i);
        auto tt = ret.getType().dyn_cast<hecate::ckks::PolyTypeInterface>();
        if (tt.getNumPoly() == 1) {
          plain_register_file.insert({ret, plain_register_file[arg]});
          plain_register_file[yielded] = plain_register_file[arg];

        } else {
          cipher_register_file.insert({ret, cipher_register_file[arg]});
          cipher_register_file[yielded] = cipher_register_file[arg];
        }
      }
      loops[heop.dst].config_body.num_operations = loop_ops.size();
      loop_insts[heop.dst] = loop_ops;

    } else if (auto ops = dyn_cast<hecate::ckks::HEVMOpInterface>(op)) {
      HEVMOperation heops = ops.getHEVMOperation(
          plain_register_file, cipher_register_file, integer_register_file);
      if (heops.opcode == 0) {
        plain_register_file.insert({op->getResult(0), heops.dst});
      } else if (heops.opcode < 100) {
        cipher_register_file.insert({op->getResult(0), heops.dst});
      } else if (heops.opcode == 200) {
        cipher_register_file.insert({op->getResult(0), heops.dst});
      } else {
        integer_register_file.insert({op->getResult(0), integer_registers++});
        heops = ops.getHEVMOperation(plain_register_file, cipher_register_file,
                                     integer_register_file);
      }
      insts.push_back(heops);
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::ckks::CKKSDialect>();
    hecate::ckks::registerArithOpInterfaceExternalModels(registry);
    /* hecate::ckks::registerSCFOpInterfaceExternalModels(registry); */
  }
};
} // namespace
