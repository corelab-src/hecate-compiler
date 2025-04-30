
#ifndef HECATE_HEVMHEADER
#define HECATE_HEVMHEADER

#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <vector>
extern "C" {

struct Body;

struct HEVMHeader {
  uint32_t magic_number = 0x4845564D;
  uint32_t hevm_header_size; // 8 + config header size byte
  struct ConfigHeader {
    uint64_t arg_length;
    uint64_t res_length;
  } config_header;
};
struct ConfigBody { // All of entry is 64 bit
  uint64_t config_body_length;
  uint64_t num_operations;
  uint64_t num_loops;
  uint64_t num_ctxt_buffer;
  uint64_t num_ptxt_buffer;
  uint64_t num_int_buffer;
  uint64_t init_level;
  /* uint64_t arg_scale[arg_length]; */
  /* uint64_t arg_level[arg_length]; */
  /* uint64_t res_scale[res_length]; */
  /* uint64_t res_level[res_length]; */
  /* uint64_t res_dst[res_length]; */
};
struct HEVMOperation {
  uint16_t opcode;
  uint16_t dst;
  uint16_t lhs;
  uint16_t rhs;
}; // 64 bit
   //
struct HEVMLoopOp : HEVMOperation {
  /* uint16_t lb = dst; */
  /* uint16_t ub = lhs; */
  /* uint16_t step = rhs; */
  struct LoopConfig {
    uint16_t lb;
    uint16_t ub;
    uint16_t step;
    uint16_t num_operations;
    /* uint64_t num_ctxt_buffer; */
    /* uint64_t num_ptxt_buffer; */
  } config_body;
};
} // namespace "C"
#endif
