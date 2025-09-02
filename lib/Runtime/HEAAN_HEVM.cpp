#include <HEaaN/Context.hpp>
#include <HEaaN/Message.hpp>
#include <HEaaN/Plaintext.hpp>
#include <HEaaN/device/CudaTools.hpp>
#include <HEaaN/device/Device.hpp>
#include <any>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>

#include <HEaaN/HEaaN.hpp>
#include <HEaaN/ParameterPreset.hpp>
#include <cmath>
#include <map>

#include <type_traits>
#include <vector>

#include "hecate/Support/BackendInterface.h"
#include "hecate/Support/ConstData.h"
#include "hecate/Support/HEVMHeader.h"

hecate::RuntimeConfig run_config{
    .debug = {.printOpStats = true, .printOpTypes = false, .printRange = false},
    .settings = {.usePreencode = false, .libName = "heaan"}};

struct HEAAN_HEVM : virtual hecate::HEVMInterface {
  using HEVMInterface::HEVMInterface;
  std::vector<HEaaN::Ciphertext> ciphers;
  std::vector<double> scalec;
  std::vector<HEaaN::Plaintext> plains;
  std::vector<double> scalep;
  std::vector<uint64_t> levelp;
  std::vector<HEaaN::Message *> msgs;
  std::map<uint16_t, HEaaN::Message> msgMap;
  std::map<uint64_t, HEaaN::Plaintext> upscale_const;

  HEaaN::Context context;
  std::unique_ptr<HEaaN::KeyPack> keypack;
  std::unique_ptr<HEaaN::SecretKey> seckey;
  std::unique_ptr<HEaaN::Encryptor> encryptor;
  std::unique_ptr<HEaaN::HomEvaluator> evaluator;
  std::unique_ptr<HEaaN::Bootstrapper> bootstrapper;
  std::unique_ptr<HEaaN::Decryptor> decryptor;
  std::unique_ptr<HEaaN::EnDecoder> endecoder;

  std::vector<int64_t> rotKeyOffset = {
      1,     2,     3,     4,     5,     6,     7,     8,     14,    16,
      24,    32,    64,    96,    128,   160,   192,   224,   256,   512,
      768,   1024,  2048,  3072,  4096,  5120,  6144,  7168,  8192,  16384,
      24576, 32768, 40960, 49152, 57344, 61440, 63488, 64512, 64768, 65024,
      65280, 65408, 65472, 65504, 65512, 65520, 65528, 65532, 65534, 65535,
  };
  /* std::vector<int64_t> rotKeyOffset = { */
  /*     1,     2,     3,     4,     5,     6,     7,     8,     16, */
  /*     24,    32,    64,    96,    128,   160,   192,   224,   256, */
  /*     512,   768,   1024,  2048,  3072,  4096,  5120,  6144,  7168, */
  /*     8192,  9216,  10240, 11264, 12288, 13312, 14336, 15360, 15616, */
  /*     15872, 16192, 16224, 16256, 16288, 16320, 16352, 16360, 16368, */
  /*     16376, 16377, 16378, 16379, 16380, 16381, 16382, 16383}; */

  bool debug = false;
  bool togpu = true;
  // bool preencode = true;

  static void create_context(char *dir) {

    auto strdir = std::string(dir);

    /* HEaaN::ParameterPreset preset = HEaaN::ParameterPreset::ST19; */
    HEaaN::ParameterPreset preset = HEaaN::ParameterPreset::FVa;
    auto context = HEaaN::makeContext(preset, {0});
    auto num_full_slot = getLogFullSlots(context);
    HEaaN::SecretKey sk(context);
    {
      std::ofstream f(strdir + "/sec.heaan", std::ios::out | std::ios::binary);
      sk.save(f);
      f.close();
    }
    HEaaN::KeyPack kp(context);
    HEaaN::KeyGenerator keygen(context, sk, kp);
    keygen.genCommonKeys();
    keygen.genRotKeysForBootstrap(num_full_slot);
    {
      keygen.save(strdir);
      HEaaN::saveContextToFile(context, strdir + "/context.heaan");
    }
  }
  size_t getCurrentMemoryUsage() override {
    return HEaaN::CudaTools::getCudaMemoryInfo().second -
           HEaaN::CudaTools::getCudaMemoryInfo().first;
  }
  // void printMemoryUsage() override {
  //   auto MemUse = HEaaN::CudaTools::getCudaMemoryInfo().second -
  //                 HEaaN::CudaTools::getCudaMemoryInfo().first;
  //   auto TotalMemCapacity = HEaaN::CudaTools::getCudaMemoryInfo().second;
  //   std::cout << "GPU memory usage: " << MemUse / std::pow(10, 9) << "GB";
  //   std::cout << " (" << double(MemUse * 100) / double(TotalMemCapacity) <<
  //   "%)"
  //             << "\n";
  // }

  void loadHEAAN(char *dir) {
    HEaaN::setUVM(HEaaN::getCurrentCudaDevice(), false);
    auto strdir = std::string(dir);
    {
      context = HEaaN::makeContextFromFile(strdir + "/context.heaan", {0});
      seckey =
          std::make_unique<HEaaN::SecretKey>(context, strdir + "/sec.heaan");
      keypack = std::make_unique<HEaaN::KeyPack>(context, strdir);
      keypack->loadEncKey();
      keypack->loadMultKey();
      for (auto offset : rotKeyOffset) {
        keypack->loadLeftRotKey(offset);
      }
    }
    encryptor = std::make_unique<HEaaN::Encryptor>(context);
    decryptor = std::make_unique<HEaaN::Decryptor>(context);
    endecoder = std::make_unique<HEaaN::EnDecoder>(context);
    evaluator = std::make_unique<HEaaN::HomEvaluator>(context, *keypack);
    bootstrapper = std::make_unique<HEaaN::Bootstrapper>(*evaluator);
    if (togpu) {
      // printMemoryUsage();
      seckey->to(HEaaN::getCurrentCudaDevice());
      // printMemoryUsage();
      keypack->to(HEaaN::getCurrentCudaDevice());
      // printMemoryUsage();
      bootstrapper->makeBootConstants(HEaaN::getLogFullSlots(context));
      // printMemoryUsage();
      bootstrapper->loadBootConstants(HEaaN::getLogFullSlots(context),
                                      HEaaN::getCurrentCudaDevice());
      // printMemoryUsage();
      // memory_usage = HEaaN::CudaTools::getCudaMemoryInfo().second -
      // HEaaN::CudaTools::getCudaMemoryInfo().first;
      key_memory_usage = getCurrentMemoryUsage();
    }
  }

  void loadClient(char *dir) {
    auto strdir = std::string(dir);
    context = HEaaN::makeContextFromFile(strdir + "/context.heaan", {0});

    encryptor = std::make_unique<HEaaN::Encryptor>(context);
    decryptor = std::make_unique<HEaaN::Decryptor>(context);
    endecoder = std::make_unique<HEaaN::EnDecoder>(context);
  }

  void loadServer(char *dir) {
    auto strdir = std::string(dir);
    context = HEaaN::makeContextFromFile(strdir + "/context.heaan", {0});
    keypack = std::make_unique<HEaaN::KeyPack>(context, strdir);

    endecoder = std::make_unique<HEaaN::EnDecoder>(context);
    evaluator = std::make_unique<HEaaN::HomEvaluator>(context, *keypack);
    bootstrapper = std::make_unique<HEaaN::Bootstrapper>(*evaluator);
  }

  void loadHeader(std::istream &iff) override {

    iff.read((char *)&header, sizeof(HEVMHeader));
    iff.read((char *)&config, sizeof(ConfigBody));

    arg_scale.resize(header.config_header.arg_length);
    arg_level.resize(header.config_header.arg_length);
    res_scale.resize(header.config_header.res_length);
    res_level.resize(header.config_header.res_length);
    res_dst.resize(header.config_header.res_length);
    iff.read((char *)arg_scale.data(), arg_scale.size() * sizeof(uint64_t));
    iff.read((char *)arg_level.data(), arg_level.size() * sizeof(uint64_t));
    iff.read((char *)res_scale.data(), res_scale.size() * sizeof(uint64_t));
    iff.read((char *)res_level.data(), res_level.size() * sizeof(uint64_t));
    iff.read((char *)res_dst.data(), res_dst.size() * sizeof(uint64_t));

    ciphers.resize(header.config_header.arg_length +
                       header.config_header.res_length,
                   HEaaN::Ciphertext(context));
    scalec.resize(config.num_ctxt_buffer);

    ciphers.resize(config.num_ctxt_buffer, HEaaN::Ciphertext(context));
    if (togpu) {
      for (auto &&cipher : ciphers)
        cipher.to(HEaaN::getCurrentCudaDevice());
    }

    HEaaN::u64 log_slot = std::log2(slot_size);
    HEaaN::Message datas(log_slot, 0.0);
    /* msgs.resize(config.num_ptxt_buffer, datas); */
    msgs.resize(config.num_ptxt_buffer);
    if (run_config.settings.usePreencode) {
      plains.resize(config.num_ptxt_buffer, HEaaN::Plaintext(context));
      if (togpu) {
        for (auto &&plain : plains)
          plain.to(HEaaN::getCurrentCudaDevice());
      }
    } else {
      plains.resize(1, HEaaN::Plaintext(context));
      if (togpu) {
        plains[0].to(HEaaN::getCurrentCudaDevice());
      }
    }
    scalec.resize(config.num_ctxt_buffer);
    scalep.resize(config.num_ptxt_buffer);
    levelp.resize(config.num_ptxt_buffer);
  }

  void resetResDst() {
    for (size_t i = 0; i < header.config_header.res_length; i++) {
      res_dst[i] = i + header.config_header.arg_length;
    }
  }

  void preprocess(std::vector<HEVMOperation> &heops) {
    std::vector<double> identity(slot_size, 1.0);
    for (HEVMOperation &op : heops) {
      if (op.opcode == uint64_t(opcode_t::ENCODE)) {
        if (run_config.settings.usePreencode) {
          encode_internal(plains[op.dst],
                          op.lhs == ((unsigned short)-1) ? identity
                                                         : constData[op.lhs],
                          op.rhs >> 10, op.rhs & 0x3FF);
        } else {
          // to_msg(op.dst,
          //        op.lhs == ((unsigned short)-1) ? identity :
          //        constData[op.lhs]);
          to_msg(op.dst, op.lhs);
        }
        levelp[op.dst] = op.rhs >> 10;
        scalep[op.dst] = op.rhs & 0x3FF;
      } else if (op.opcode == uint64_t(opcode_t::LOOP)) {
        std::vector<HEVMOperation> &loop_body = loop_insts[op.dst];
        preprocess(loop_body);
      }
    }
  }

  void to_msg(int16_t dst, uint16_t lhs) {
    HEaaN::u64 log_slot = std::log2(slot_size);
    std::vector<double> identity(slot_size, 1.0);

    if (!msgMap.count(lhs)) {
      msgMap[lhs] = HEaaN::Message(log_slot, 0.0);
      auto &msg = msgMap[lhs];
      auto &src = lhs == ((unsigned short)-1) ? identity : constData[lhs];
      for (size_t i = 0; i < msg.getSize(); i++) {
        msg[i].real(src[i % src.size()]);
        msg[i].imag(0);
      }

      if (togpu)
        msgMap[lhs].to(HEaaN::getCurrentCudaDevice());
    }
    msgs[dst] = &msgMap[lhs];

    return;
  }

  void encode_online(int16_t dst) override {
    /* if (togpu) */
    /*   msgs[dst].to(HEaaN::getCurrentCudaDevice()); */
    plains[0] =
        endecoder->encode(*msgs[dst], levelp[dst], std::pow(2.0, scalep[dst]));
  }

  void encode_internal(HEaaN::Plaintext &dst, std::vector<double> src,
                       int16_t level, uint64_t scale) {
    HEaaN::u64 log_slot = std::log2(slot_size);
    HEaaN::Message datas(log_slot, 0.0);

    for (size_t i = 0; i < datas.getSize(); i++) {
      datas[i].real(src[i % src.size()]);
      datas[i].imag(0);
    }
    if (togpu) {
      datas.to(HEaaN::getCurrentCudaDevice());
    }
    dst = endecoder->encode(datas, level, std::pow(2.0, scale));
    return;
  }

  void encode(int16_t dst, int16_t src, int8_t level, int8_t scale) override {
    return;
  }
  void rotate(int16_t dst, int16_t src, int16_t offset) override {
    evaluator->leftRotate(ciphers[src], offset, ciphers[dst]);
    scalec[dst] = scalec[src];
  }
  void negate(int16_t dst, int16_t src) override {
    evaluator->negate(ciphers[src], ciphers[dst]);
    scalec[dst] = scalec[src];
  }
  void rescale(int16_t dst, int16_t src) override {
    ciphers[dst] = ciphers[src];
    scalec[dst] =
        scalec[src] - std::round(ciphers[src].getCurrentScaleFactor());
    ciphers[dst].setRescaleCounter(1);
    evaluator->rescale(ciphers[dst]);
  }
  void modswitch(int16_t dst, int16_t src, int16_t downFactor) override {
    if (downFactor > 0) {
      scalec[dst] =
          scalec[src] - std::round(ciphers[src].getCurrentScaleFactor());
      evaluator->levelDownOne(ciphers[src], ciphers[dst]);
      scalec[dst] += std::round(ciphers[dst].getCurrentScaleFactor());
    }
    for (int i = 1; i < downFactor; i++) {
      scalec[dst] =
          scalec[dst] - std::round(ciphers[dst].getCurrentScaleFactor());
      evaluator->levelDownOne(ciphers[dst], ciphers[dst]);
      scalec[dst] += std::round(ciphers[dst].getCurrentScaleFactor());
    }
  }
  void upscale(int16_t dst, int16_t src, int16_t upFactor) override {
    assert(0 && "This VM does not support native upscale op");
  }
  void addcc(int16_t dst, int16_t lhs, int16_t rhs) override {
    scalec[dst] = scalec[lhs];
    evaluator->add(ciphers[lhs], ciphers[rhs], ciphers[dst]);
  }
  void addcp(int16_t dst, int16_t lhs, int16_t rhs) override {
    scalec[dst] = scalec[lhs];
    if (run_config.settings.usePreencode) {
      evaluator->add(ciphers[lhs], plains[rhs], ciphers[dst]);
    } else {
      encode_online(rhs);
      evaluator->add(ciphers[lhs], plains[0], ciphers[dst]);
    }
  }
  void mulcc(int16_t dst, int16_t lhs, int16_t rhs) override {
    evaluator->multWithoutRescale(ciphers[lhs], ciphers[rhs], ciphers[dst]);
    ciphers[dst].setRescaleCounter(0);
    scalec[dst] = scalec[lhs] + scalec[rhs];
  }
  void mulcp(int16_t dst, int16_t lhs, int16_t rhs) override {
    if (run_config.settings.usePreencode) {
      evaluator->multWithoutRescale(ciphers[lhs], plains[rhs], ciphers[dst]);
    } else {
      encode_online(rhs);
      evaluator->multWithoutRescale(ciphers[lhs], plains[0], ciphers[dst]);
    }
    ciphers[dst].setRescaleCounter(0);
    scalec[dst] = scalec[lhs] + scalep[rhs];
  }
  void bootstrap(int16_t dst, int64_t src, uint64_t targetLevel) override {
    bootstrapper->bootstrap(ciphers[src], ciphers[dst], targetLevel, false);
    // HEaaN::CudaTools::cudaDeviceSynchronize();
    scalec[dst] = ciphers[dst].getCurrentScaleFactor();
  }

  void copyCipher(int16_t dst, int16_t lhs) override {
    ciphers[dst] = ciphers[lhs];
    scalec[dst] = scalec[lhs];
  }

  // Debugging functions
  hecate::msg_t decrypt(int64_t dst) override {
    HEaaN::Plaintext ptxt(context);
    decryptor->decrypt(ciphers[dst], *seckey, ptxt);
    // encoder->decode(msg, ptxt);
    HEaaN::Message msg_heaan =
        endecoder->decode(ptxt, std::pow(2.0, std::round(scalec[dst])));
    if (togpu)
      msg_heaan.to(HEaaN::getDefaultDevice());
    hecate::msg_t msg(slot_size, 0.0);
    for (int i = 0; i < slot_size; i++) {
      msg[i] = msg_heaan[i].real();
    }
    return msg;
  }

  double getCipherScale(int16_t i) override { return scalec[i]; }
  double getPlainScale(int16_t i) override { return scalep[i]; }
  int getCipherLevel(int16_t i) override { return ciphers[i].getLevel(); }
  int getPlainLevel(int16_t i) override { return levelp[i]; }
};

extern "C" {
void *initFullVM(char *dir, int64_t N, int64_t L, bool device = false) {
  auto vm = new HEAAN_HEVM(N, L);
  vm->togpu = device;
  vm->loadHEAAN(dir);
  return (void *)vm;
}
void *initClientVM(char *dir, int64_t N, int64_t L) {
  auto vm = new HEAAN_HEVM(N, L);
  vm->loadClient(dir);
  return (void *)vm;
}
void *initServerVM(char *dir, int64_t N, int64_t L) {
  auto vm = new HEAAN_HEVM(N, L);
  vm->loadServer(dir);
  return (void *)vm;
}

void create_context(char *dir) { HEAAN_HEVM::create_context(dir); }

// Loader for server
void load(void *vm, char *constant, char *vmfile) {
  auto hevm = static_cast<HEAAN_HEVM *>(vm);
  hevm->loadConstants(constant);
  hevm->loadHEVM(vmfile, run_config);
}

// Loader for client
void loadClient(void *vm, void *is) {
  auto hevm = static_cast<HEAAN_HEVM *>(vm);
  std::istream &iss = *static_cast<std::istream *>(is);
  hevm->loadHeader(iss);
  hevm->resetResDst();
}

// set Epoch of loop
void setEpoch(void *vm, int64_t i, int epoch) {
  auto hevm = static_cast<HEAAN_HEVM *>(vm);
  hevm->integers[i] = epoch;
}

// encryption and decryption uses internal buffer id
void encrypt(void *vm, int64_t i, double *dat, int len) {
  auto hevm = static_cast<HEAAN_HEVM *>(vm);
  HEaaN::Plaintext ptxt(hevm->context);
  std::vector<double> dats(dat, dat + len);
  hevm->encode_internal(ptxt, dats, hevm->arg_level[i], hevm->arg_scale[i]);
  hevm->encryptor->encrypt(ptxt, *hevm->seckey, hevm->ciphers[i]);
  // TODO: Hide the visibleCiphers from the user
  if (hevm->runConfig.debug.printOpTypes || hevm->runConfig.debug.printRange) {
    for (int j = 0; j < hevm->slot_size; j++) {
      hevm->visibleCiphers[i][j] = dats[j % len];
    }
  }
  if (hevm->togpu) {
    hevm->ciphers[i].to(HEaaN::getCurrentCudaDevice());
  }
  hevm->scalec[i] = hevm->arg_scale[i];
}
void decrypt(void *vm, int64_t i, double *dat) {
  auto hevm = static_cast<HEAAN_HEVM *>(vm);
  HEaaN::Plaintext ptxt(hevm->context);
  hevm->decryptor->decrypt(hevm->ciphers[i], *hevm->seckey, ptxt);
  HEaaN::Message msg =
      hevm->endecoder->decode(ptxt, std::pow(2.0, std::round(hevm->scalec[i])));
  if (hevm->togpu)
    msg.to(HEaaN::getDefaultDevice());
  for (int i = 0; i < hevm->slot_size; i++) {
    /* for (size_t j = 0; j < msg.getSize(); j++) */
    dat[i] = msg[i].real();
  }
}
// simple wrapper to elide getResIdx call
// use res_idx for i
void decrypt_result(void *vm, int64_t i, double *dat) {
  auto hevm = static_cast<HEAAN_HEVM *>(vm);
  decrypt(vm, hevm->res_dst[i], dat);
}

// We need this for communication code to access the proper buffer id
int64_t getResIdx(void *vm, int64_t i) {
  auto hevm = static_cast<HEAAN_HEVM *>(vm);
  return hevm->res_dst[i];
}

// use this to implement communication
void *getCtxt(void *vm, int64_t id) {
  auto hevm = static_cast<HEAAN_HEVM *>(vm);
  return &(hevm->ciphers[id]);
}

void preprocess(void *vm) {
  auto hevm = static_cast<HEAAN_HEVM *>(vm);
  hevm->preprocess(hevm->ops);
}

void run(void *vm) {
  auto hevm = static_cast<HEAAN_HEVM *>(vm);
  hevm->run(hevm->ops);
  hevm->printFinalResults();
}
int64_t getArgLen(void *vm) {
  auto hevm = static_cast<HEAAN_HEVM *>(vm);
  return hevm->header.config_header.arg_length;
}
int64_t getResLen(void *vm) {
  auto hevm = static_cast<HEAAN_HEVM *>(vm);
  return hevm->header.config_header.res_length;
}
void setDebug(void *vm, bool enable) {
  auto hevm = static_cast<HEAAN_HEVM *>(vm);
  hevm->debug = enable;
}
void setToGPU(void *vm, bool ongpu) {
  auto hevm = static_cast<HEAAN_HEVM *>(vm);
  hevm->togpu = ongpu;
}
void getRunInfo(void *vm) {
  auto hevm = static_cast<HEAAN_HEVM *>(vm);
  /* info[0] = hevm->tttt; */
  /* info[1] = 0.0; */
  /* hevm->boot_cnt; */
  // hevm->printMemoryUsage();
  // hevm->printInfo();
}
};
