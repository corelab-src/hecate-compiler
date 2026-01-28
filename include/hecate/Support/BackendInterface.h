#ifndef HECATE_SUPPORT_BACKENDINTERFACE
#define HECATE_SUPPORT_BACKENDINTERFACE

#include "hecate/Support/ConstData.h"
#include "hecate/Support/HEVMHeader.h"
#include "hecate/Support/RangeTracker.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <stdexcept>
#include <variant>
#include <vector>
#pragma once

enum class opcode_t : uint16_t {
  ENCODE = 0,
  ROTATE = 1,
  NEGATE = 2,
  RESCALE = 3,
  MODSWITCH = 4,
  UPSCALE = 5,
  ADDCC = 6,
  ADDCP = 7,
  MULCC = 8,
  MULCP = 9,
  BOOTSTRAP = 10,
  LOOP = 11,
  CONSTINT = 100,
  ADDINT = 101,
  SUBINT = 102,
  REMINT = 103,
  COPYC = 200,
};

// global map for opcode to name
inline const std::map<opcode_t, std::string> &opcode_name_map() {
  static const std::map<opcode_t, std::string> m = {
      {opcode_t::ENCODE, "Encode"},       {opcode_t::ROTATE, "Rotate"},
      {opcode_t::NEGATE, "Negate"},       {opcode_t::RESCALE, "Rescale"},
      {opcode_t::MODSWITCH, "Modswitch"}, {opcode_t::UPSCALE, "Upscale"},
      {opcode_t::ADDCC, "AddCC"},         {opcode_t::ADDCP, "AddCP"},
      {opcode_t::MULCC, "MulCC"},         {opcode_t::MULCP, "MulCP"},
      {opcode_t::BOOTSTRAP, "Bootstrap"}, {opcode_t::LOOP, "Loop"},
      {opcode_t::CONSTINT, "ConstInt"},   {opcode_t::ADDINT, "AddInt"},
      {opcode_t::SUBINT, "SubInt"},       {opcode_t::REMINT, "RemInt"},
      {opcode_t::COPYC, "CopyC"},
  };
  return m;
}

inline std::string getOpName(opcode_t op) {
  const auto &m = opcode_name_map();
  if (auto it = m.find(op); it != m.end())
    return it->second;
  return "EMPTY";
}

namespace hecate {
using msg_t = std::vector<double>;

// Options for printing debug information
struct DebugOptions {
  bool printOpStats = false;
  bool printOpTypes = false;
  bool printRange = false;
  // bool printMemoryUsage = false;
};

// Options for running environment
struct ExecutionSettings {
  bool usePreencode = false;
  std::string libName = "";
  // std::string hwTarget = "CPU"; // or "GPU"
};

// Options for running the HEVM
struct RuntimeConfig {
  DebugOptions debug;
  ExecutionSettings settings;
};

class HEVMInterface {
public:
  HEVMInterface(uint64_t N, uint64_t L);
  virtual ~HEVMInterface() = default;

  ConstData constData;
  HEVMHeader header;
  ConfigBody config;
  std::string benchName;
  std::vector<HEVMOperation> ops;
  std::vector<HEVMLoopOp> loops;
  std::vector<std::vector<HEVMOperation>> loop_insts;
  std::vector<uint64_t> arg_scale;
  std::vector<uint64_t> arg_level;
  std::vector<uint64_t> res_scale;
  std::vector<uint64_t> res_level;
  std::vector<uint64_t> res_dst;
  std::vector<int> integers;
  std::vector<double> scalep;
  std::vector<uint64_t> levelp;

  std::vector<std::vector<int64_t>> arg_shape;
  // Runtime configuration
  RuntimeConfig runConfig;

  virtual void loadHeader(std::istream &iff) = 0;
  void loadConstants(char *name);
  void loadHEVM(char *name, RuntimeConfig &run_options);
  void setRuntimeConfig(RuntimeConfig &config);
  std::vector<int> op_count;
  std::vector<uint64_t> op_time;
  uint64_t key_memory_pool_usage = 0;
  uint64_t data_memory_pool_usage = 0;
  uint64_t total_memory_pool_usage = 0;
  uint64_t N;
  uint64_t L;
  uint64_t slot_size;

  // Virtual functions to be implemented by derived classes
  virtual void encode(int16_t dst, int16_t src, int8_t level, int8_t scale) = 0;
  virtual void encode_online(uint16_t dst) = 0;
  virtual void encode_online(uint16_t dst, int depth) = 0;
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
  virtual void copyCipher(int16_t dst, int16_t src) = 0;

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
  void printOperandsType(const HEVMOperation &op);
  void printResultsType(const HEVMOperation &op);
  void printValueRange(const HEVMOperation &op);
  void printPerformanceStats();
  void printFinalResults();
  virtual size_t getCurrentMemoryUsage() = 0;
  void checkPrecision(const msg_t &v1, const msg_t &v2);
  RangeTracker rangeTracker;

  // Helper Functions to format output
  std::string padLeft(const std::string &s, size_t width);
  std::string padRight(const std::string &s, size_t width);
  template <typename T> std::string toString(T val);

private:
  int num_op_ = -1;
};
} // namespace hecate
#endif
