#include <hecate/Support/BackendInterface.h>

// Debug Configurations
namespace hecate {

HEVMInterface::HEVMInterface(uint64_t N, uint64_t L)
    : N(N), L(L), slot_size(N >> 1) {
  // Bootstrap opcode should be defined in the last
  op_count.resize(static_cast<int>(opcode_t::BOOTSTRAP) + 1, 0);
  op_time.resize(static_cast<int>(opcode_t::BOOTSTRAP) + 1, 0);
}

void HEVMInterface::setRuntimeConfig(RuntimeConfig &RunOptions) {
  runConfig = RunOptions;
  if (runConfig.debug.printOpTypes) {
    visibleCiphers.resize(config.num_ctxt_buffer,
                          std::vector<double>(slot_size, 0.0));
    visiblePlains.resize(config.num_ptxt_buffer,
                         std::vector<double>(slot_size, 0.0));
  }
}

// run the HEVM operations based on the opcode
void HEVMInterface::run(std::vector<HEVMOperation> &heops) {
  bool printTypes = runConfig.debug.printOpTypes;
  bool printStats = runConfig.debug.printOpStats;

  int i = (header.hevm_header_size + config.config_body_length) / 8;
  int j = 0;
  std::chrono::high_resolution_clock::time_point start, end;
  for (HEVMOperation &op : heops) {
    if (printTypes)
      printOperandsType(op, j);
    if (printStats)
      start = std::chrono::high_resolution_clock::now();

    opcode_t opcode = static_cast<opcode_t>(op.opcode);
    switch (opcode) {
    case opcode_t::ENCODE: { // Encode
      encode(op.dst, op.lhs, op.rhs >> 10, op.rhs & 0x3FF);
      break;
    }
    case opcode_t::ROTATE: {
      rotate(op.dst, op.lhs, op.rhs);
      break;
    }
    case opcode_t::NEGATE: {
      negate(op.dst, op.lhs);
      break;
    }
    case opcode_t::RESCALE: { // RescaleC
      rescale(op.dst, op.lhs);
      break;
    }
    case opcode_t::MODSWITCH: { // ModswtichC
      modswitch(op.dst, op.lhs, op.rhs);
      break;
    }
    case opcode_t::UPSCALE: { // UpscaleC
      upscale(op.dst, op.lhs, op.rhs);
      break;
    }
    case opcode_t::ADDCC: { // AddCC
      addcc(op.dst, op.lhs, op.rhs);
      break;
    }
    case opcode_t::ADDCP: { // AddCP
      addcp(op.dst, op.lhs, op.rhs);
      break;
    }
    case opcode_t::MULCC: { // MulCC
      mulcc(op.dst, op.lhs, op.rhs);
      break;
    }
    case opcode_t::MULCP: { // MulCP
      mulcp(op.dst, op.lhs, op.rhs);
      break;
    }
    case opcode_t::BOOTSTRAP: { // Bootstrap
      // HEaaN::CudaTools::cudaDeviceSynchronize();
      bootstrap(op.dst, op.lhs, op.rhs);
      break;
    }
    // case 11: { // loop
    //   hevm_->heloop(op.dst);
    //   break;
    // }
    // case 200: {
    //   hevm_->copyCipher(op.dst, op.lhs);
    //   break;
    // }
    // case 100: {
    //   hevm_->arithConstant(op.dst, op.lhs);
    //   break;
    // }
    // case 101: {
    //   hevm_->arithAddI(op.dst, op.lhs, op.rhs);
    //   break;
    // }
    // case 102: {
    //   hevm_->arithSubI(op.dst, op.lhs, op.rhs);
    //   break;
    // }
    // case 103: {
    //   arithRemSI(op.dst, op.lhs, op.rhs);
    //   break;
    // }
    default: {
      break;
    }
    }
    // std::cout << getOpName(opcode) << " "
    // << getCurrentMemoryUsage() / std::pow(10, 9) << "GB" << '\n';
    if (printStats) {
      end = std::chrono::high_resolution_clock::now();
      auto time_diff =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count();
      op_count[op.opcode]++;
      op_time[op.opcode] += time_diff;
    }
    if (printTypes)
      printResultsType(op);
  }
}

// Debug the operands of an operation and print the type information
// TODO: Use a more structured way to handle the output
void HEVMInterface::printOperandsType(const HEVMOperation &op, int &num_op) {
  // Find the opcode in the map
  auto opcode = static_cast<opcode_t>(op.opcode);
  auto op_name = getOpName(opcode);
  std::cout << std::unitbuf; // Flush the output buffer immediately
  std::cout << "%" << num_op++ << " " << op_name << "\n";
  if (op_name == "EMPTY") {
    return;
  }
  if (opcode == opcode_t::ENCODE) {
    std::cout << "msgs[" << op.lhs << "]";
    return;
  }

  std::cout << "(ciphers[" << op.lhs << "] <" << getCipherLevel(op.lhs) << " * "
            << getCipherScale(op.lhs) << ">)";

  switch (opcode) {
  case opcode_t::ADDCC:
  case opcode_t::MULCC:
    std::cout << ", (ciphers[" << op.rhs << "] <" << getCipherLevel(op.rhs)
              << " * " << getCipherScale(op.rhs) << ">)";
    break;

  case opcode_t::ADDCP:
  case opcode_t::MULCP:
    if (!runConfig.settings.usePreencode) {
      std::cout << ", (plains[" << op.rhs << "] <" << int16_t(op.rhs >> 10)
                << " * " << uint64_t(op.rhs & 0x3FF) << ">)";
    } else {
      std::cout << ", (plains[" << op.rhs << "] <" << getPlainLevel(op.rhs)
                << " * " << getPlainScale(op.rhs) << ">)";
    }
    break;

  case opcode_t::ROTATE:
    std::cout << " @offset(" << int16_t(op.rhs) << ")";
    break;

  case opcode_t::MODSWITCH:
    std::cout << " @downFactor(" << op.rhs << ")";
    break;

  case opcode_t::UPSCALE:
    std::cout << " @upFactor(" << op.rhs << ")";
    break;

  case opcode_t::BOOTSTRAP:
    std::cout << " @targetLevel(" << op.rhs << ")";
    break;

  default:
    break;
  }
}

// Run the operation with visible data
msg_t HEVMInterface::runVisible(const HEVMOperation &op) {
  auto opcode = static_cast<opcode_t>(op.opcode);
  switch (opcode) {
  case opcode_t::ENCODE: {
    msg_t const_msg(slot_size, 1.0);
    if (op.lhs == (unsigned short)-1) {
      visiblePlains[op.dst] = const_msg;
    } else {
      size_t data_size = constData[op.lhs].size();
      for (size_t i = 0; i < slot_size; i++) {
        visiblePlains[op.dst][i] = constData[op.lhs][i % data_size];
      }
    }
    return visiblePlains[op.dst];
  } break;
  case opcode_t::ADDCC:
    for (int i = 0; i < slot_size; ++i) {
      visibleCiphers[op.dst][i] =
          visibleCiphers[op.lhs][i] + visibleCiphers[op.rhs][i];
    }
    break;

  case opcode_t::MULCC:
    for (int i = 0; i < slot_size; ++i) {
      visibleCiphers[op.dst][i] =
          visibleCiphers[op.lhs][i] * visibleCiphers[op.rhs][i];
    }
    break;

  case opcode_t::ADDCP: {
    for (int i = 0; i < slot_size; ++i) {
      visibleCiphers[op.dst][i] =
          visibleCiphers[op.lhs][i] + visiblePlains[op.rhs][i];
    }
    break;
  }
  case opcode_t::MULCP: {
    for (int i = 0; i < slot_size; ++i) {
      visibleCiphers[op.dst][i] =
          visibleCiphers[op.lhs][i] * visiblePlains[op.rhs][i];
    }
    break;
  }
  case opcode_t::NEGATE:
    for (int i = 0; i < slot_size; ++i) {
      visibleCiphers[op.dst][i] = -visibleCiphers[op.lhs][i];
    }
    break;

  case opcode_t::ROTATE: {
    visibleCiphers[op.dst] = visibleCiphers[op.lhs];
    int offset = ((op.rhs % slot_size) + slot_size) % slot_size;
    std::rotate(visibleCiphers[op.dst].begin(),
                visibleCiphers[op.dst].begin() + offset,
                visibleCiphers[op.dst].end());
    break;
  }
  case opcode_t::RESCALE:
  case opcode_t::MODSWITCH:
  case opcode_t::BOOTSTRAP:
    visibleCiphers[op.dst] = visibleCiphers[op.lhs];
    break;

  default:
    break;
  }

  return visibleCiphers[op.dst];
}

// Debug the results of an operation and print the results
void HEVMInterface::printResultsType(const HEVMOperation &op) {
  // Find the opcode in the map
  auto opcode = static_cast<opcode_t>(op.opcode);
  auto op_name = getOpName(opcode);
  msg_t plain_result = runVisible(op);
  msg_t he_result;
  if (op_name == "EMPTY") {
    std::cout << std::endl;
    return;
  } else if (opcode == opcode_t::ENCODE) {
    std::cout << std::endl;
    std::cout << std::endl;
    return;

  } else {
    std::cout << " --> (ciphers[" << op.dst << "] <" << getCipherLevel(op.dst)
              << " * " << getCipherScale(op.dst) << ">) " << std::endl;
    he_result = decrypt(op.dst);
  }
  checkPrecision(plain_result, he_result);
  std::cout << std::endl;
}

// Check the precision of two vectors and print the results
void HEVMInterface::checkPrecision(const msg_t &v1, const msg_t &v2) {
  double sumSquares = 0.0;
  double maxDiff = -std::numeric_limits<double>::infinity();
  size_t maxDiffIndex = 0;

  double minPrecision = std::numeric_limits<double>::infinity();
  size_t minPrecisionIndex = 0;

  double maxPrecision = -std::numeric_limits<double>::infinity();
  size_t maxPrecisionIndex = 0;

  for (size_t i = 0; i < v1.size(); ++i) {
    double diff = v1[i] - v2[i];
    double absDiff = std::abs(diff);
    sumSquares += diff * diff;

    if (absDiff > maxDiff) {
      maxDiff = absDiff;
      maxDiffIndex = i;
    }
    // double ref = std::max(std::abs(v1[i]), std::abs(v2[i]));
    double correctBits;

    if (absDiff > 0) {
      correctBits = -std::log2(absDiff);
    } else {
      correctBits = 53.0; // Perfect match or both zero
    }

    if (correctBits < minPrecision) {
      minPrecision = correctBits;
      minPrecisionIndex = i;
    }

    if (correctBits > maxPrecision) {
      maxPrecision = correctBits;
      maxPrecisionIndex = i;
    }
  }

  double rms = std::sqrt(sumSquares / v1.size());

  std::cout << "RMS difference: " << rms << std::endl;

  std::cout << "Maximum precision: " << maxPrecision << " bits (at index "
            << maxPrecisionIndex << ")" << std::endl;
  std::cout << "Minimum precision: " << minPrecision << " bits (at index "
            << minPrecisionIndex << ")" << std::endl;
  std::cout << "  plain_res[" << minPrecisionIndex
            << "] = " << v1[minPrecisionIndex] << "\n";
  std::cout << "  he_res[" << minPrecisionIndex
            << "] = " << v2[minPrecisionIndex] << "\n";
}

std::string HEVMInterface::padLeft(const std::string &s, size_t width) {
  if (s.length() >= width)
    return s;
  return s + std::string(width - s.length(), ' ');
}

std::string HEVMInterface::padRight(const std::string &s, size_t width) {
  if (s.length() >= width)
    return s;
  return std::string(width - s.length(), ' ') + s;
}

template <typename T> std::string HEVMInterface::toString(T val) {
  return std::to_string(val);
}

void HEVMInterface::printPerformanceStats() {
  const int nameWidth = 15;
  const int countWidth = 10;
  const int timeWidth = 15;
  const int percentWidth = 15;

  std::cout << "==================================================\n";
  std::cout << padLeft("Operation", nameWidth) << padLeft("Count", countWidth)
            << padLeft("Time(\u03BCs)", timeWidth) // microseconds
            << padLeft("Percent", percentWidth)
            // << "Average(ns)"
            << "\n";
  std::cout << "--------------------------------------------------\n";
  double total_time = 0.0;
  int total_cnt = 0;
  for (size_t i = 0; i < op_count.size(); ++i) {
    total_time += op_time[i];
    total_cnt += op_count[i];
  }

  auto printEntry = [&](const std::string &name, int cnt, double time) {
    std::cout << padLeft(name, nameWidth) << padLeft(toString(cnt), countWidth)
              << padLeft(toString((long long)time), timeWidth)
              << padLeft(toString(time * 100.0 / total_time), percentWidth)
              // << padLeft(toString(time / cnt), timeWidth)
              << "\n";
  };

  // Print "opname, count, time, and percentage
  for (size_t i = 0; i < op_count.size(); ++i) {
    printEntry(getOpName(static_cast<opcode_t>(i)), op_count[i], op_time[i]);
  }
  std::cout << "--------------------------------------------------\n";
  std::cout << padLeft("total", nameWidth)
            << padLeft(toString(total_cnt), countWidth)
            << (total_time / 1000000.0) << "ms\n";
  std::cout << "--------------------------------------------------\n";
  std::cout << "config.num_ptxt: " << config.num_ptxt_buffer << '\n';
  std::cout << "config.num_ctxt: " << config.num_ctxt_buffer << '\n';
  std::cout << "key_memory_usage: " << key_memory_usage / std::pow(10, 9)
            << "GB" << '\n';
  total_memory_usage = getCurrentMemoryUsage();
  std::cout << "Data Memory Usage: "
            << (total_memory_usage - key_memory_usage) / std::pow(10, 9) << "GB"
            << '\n';
  std::cout << "Total Memory Usage: " << total_memory_usage / std::pow(10, 9)
            << "GB" << '\n';
  std::cout << "==================================================\n";
}

} // namespace hecate
