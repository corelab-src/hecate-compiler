#ifndef HECATE_SUPPORT_BACKENDINTERFACE
#define HECATE_SUPPORT_BACKENDINTERFACE

#include "hecate/Support/ConstData.h"
#include "hecate/Support/HEVMHeader.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <variant>
#include <vector>
#pragma once

// TODO: Better way to define opcodes for opcode scalaibility
#define OPCODE_LIST(OP)                                                        \
  OP(ENCODE, 0, "Encode")                                                      \
  OP(ROTATE, 1, "Rotate")                                                      \
  OP(NEGATE, 2, "Negate")                                                      \
  OP(RESCALE, 3, "Rescale")                                                    \
  OP(MODSWITCH, 4, "Modswtich")                                                \
  OP(UPSCALE, 5, "Upscale")                                                    \
  OP(ADDCC, 6, "AddCC")                                                        \
  OP(ADDCP, 7, "AddCP")                                                        \
  OP(MULCC, 8, "MulCC")                                                        \
  OP(MULCP, 9, "MulCP")                                                        \
  OP(BOOTSTRAP, 10, "Bootstrap")

enum class opcode_t : uint16_t {
#define DEFINE_ENUM(name, val, str) name = val,
  OPCODE_LIST(DEFINE_ENUM)
#undef DEFINE_ENUM
};

inline const char *getOpName(opcode_t op) {
  switch (op) {
#define CASE_STRING(name, val, str)                                            \
  case opcode_t::name:                                                         \
    return str;
    OPCODE_LIST(CASE_STRING)
#undef CASE_STRING
  default:
    return "EMPTY";
    // assert(0 && "Opcode not found in map");
  }
}
#undef OPCODE_LIST

namespace hecate {
using msg_t = std::vector<double>;

// Options for printing debug information
struct DebugOptions {
  bool printRange = false;
  bool printOpTypes = false;
  bool printOpStats = false;
  // bool printMemoryUsage = false;
};

// Options for running environment
struct ExecutionSettings {
  bool usePreencode = false;
  // std::string hwTarget = "CPU"; // or "GPU"
};

// Options for running the HEVM
struct RuntimeConfig {
  DebugOptions debug;
  ExecutionSettings settings;
};

class HEVMInterface {
public:
  HEVMInterface(uint64_t L, uint64_t N);
  virtual ~HEVMInterface() = default;

  ConstData constData;
  HEVMHeader header;
  ConfigBody config;

  // Runtime configuration
  RuntimeConfig runConfig;
  void setRuntimeConfig(RuntimeConfig &config);
  std::vector<int> op_count;
  std::vector<uint64_t> op_time;
  uint64_t key_memory_usage = 0;
  uint64_t total_memory_usage = 0;
  uint64_t N;
  uint64_t L;
  uint64_t slot_size;

  // Virtual functions to be implemented by derived classes
  virtual void encode(int16_t dst, int16_t src, int8_t level, int8_t scale) = 0;
  virtual void encode_online(int16_t dst) = 0;
  virtual void rotate(int16_t dst, int16_t src, int16_t offset) = 0;
  virtual void negate(int16_t dst, int16_t src) = 0;
  virtual void rescale(int16_t dst, int16_t src) = 0;
  virtual void modswitch(int16_t dst, int16_t src, int16_t downFactor) = 0;
  virtual void upscale(int16_t dst, int16_t src, int16_t upFactor) = 0;
  virtual void addcc(int16_t dst, int16_t lhs, int16_t rhs) = 0;
  virtual void addcp(int16_t dst, int16_t lhs, int16_t rhs) = 0;
  virtual void mulcc(int16_t dst, int16_t lhs, int16_t rhs) = 0;
  virtual void mulcp(int16_t dst, int16_t lhs, int16_t rhs) = 0;
  virtual void bootstrap(int16_t dst, int64_t src, uint64_t targetLevel) = 0;

  void run(std::vector<HEVMOperation> &heops);

  // virtual function to support getting type methods granularity
  virtual msg_t decrypt(int64_t dst) = 0;
  // get scale form should be log2 of the scale
  virtual double getCipherScale(int16_t dst) = 0;
  virtual double getPlainScale(int16_t dst) = 0;
  virtual int getCipherLevel(int16_t dst) = 0;
  virtual int getPlainLevel(int16_t dst) = 0;

  // Debugging functions
  std::vector<msg_t> visibleCiphers;
  std::vector<msg_t> visiblePlains;
  void loadVisibleBackend();
  msg_t runVisible(const HEVMOperation &op);
  void printOperandsType(const HEVMOperation &op, int &num_op);
  void printResultsType(const HEVMOperation &op);
  void printPerformanceStats();
  virtual size_t getCurrentMemoryUsage() = 0;
  void checkPrecision(const msg_t &v1, const msg_t &v2);

  // Helper Functions to format output
  std::string padLeft(const std::string &s, size_t width);
  std::string padRight(const std::string &s, size_t width);
  template <typename T> std::string toString(T val);

private:
};
} // namespace hecate
#endif
