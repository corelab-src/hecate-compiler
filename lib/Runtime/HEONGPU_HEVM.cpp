#include <HEonGPU-1.1/heongpu.cuh>
#include <any>
#include <cassert>
#include <chrono>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

#include <cmath>
#include <map>

#include <memory>
#include <omp.h>
#include <type_traits>
#include <vector>

#include "hecate/Support/BackendInterface.h"
#include "hecate/Support/ConstData.h"
#include "hecate/Support/HEVMHeader.h"

#define PRINT_OPTYPES false
#define PRINT_OPSTATS true
#define PRINT_RANGE false

#define USE_PREENCODE false

constexpr auto Scheme = heongpu::Scheme::CKKS;
using Message = std::vector<double>;
struct HEONGPU_HEVM : virtual hecate::HEVMInterface {
  using HEVMInterface::HEVMInterface;
  std::vector<HEVMOperation> ops;
  std::vector<uint64_t> arg_scale;
  std::vector<uint64_t> arg_level;
  std::vector<uint64_t> res_scale;
  std::vector<uint64_t> res_level;
  std::vector<uint64_t> res_dst;

  std::vector<double> scalep;
  std::vector<uint64_t> levelp;
  std::vector<int> integers;
  std::vector<heongpu::Ciphertext<Scheme>> ciphers;
  std::vector<heongpu::Plaintext<Scheme>> plains;
  std::vector<Message *> msgs;
  std::map<int16_t, Message> msgMap;

  std::map<uint64_t, heongpu::Plaintext<Scheme>> upscale_const;

  std::unique_ptr<heongpu::HEContext<Scheme>> context;
  std::unique_ptr<heongpu::Secretkey<Scheme>> secret_key;
  std::unique_ptr<heongpu::Publickey<Scheme>> public_key;
  std::unique_ptr<heongpu::Galoiskey<Scheme>> galois_key;
  std::unique_ptr<heongpu::Relinkey<Scheme>> relin_key;
  std::unique_ptr<heongpu::HEEncoder<Scheme>> encoder;
  std::unique_ptr<heongpu::HEEncryptor<Scheme>> encryptor;
  std::unique_ptr<heongpu::HEArithmeticOperator<Scheme>> operators;
  std::unique_ptr<heongpu::HEDecryptor<Scheme>> decryptor;
  std::unique_ptr<heongpu::Bootstrap> bootstrapper;

  // runConfig

  static void create_context(char *dir) {}

  size_t getCurrentMemoryUsage() override {
    size_t free_mem, total_mem;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err != cudaSuccess) {
      assert(std::cerr << "CUDA Error: " << cudaGetErrorString(err)
                       << std::endl);
    }

    // std::cout << "Free Memory: " << free_mem / (1024.0 * 1024.0) << " MB"
    //           << std::endl;
    // std::cout << "Total Memory: " << total_mem / (1024.0 * 1024.0) << " MB"
    //           << std::endl;
    return total_mem - free_mem;
  }

  void loadHEONGPU(char *dir) {
    cudaSetDevice(0);
    auto strdir = std::string(dir);

    heongpu::HEContext<Scheme> context(
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_II,
        // heongpu::sec_level_type::sec128);
        heongpu::sec_level_type::none);

    context.set_poly_modulus_degree(N);

    std::vector<int> q_prime_list(26, 52);
    q_prime_list.insert(q_prime_list.begin(), 52);
    // std::vector<int> q_prime_list(27, 52);
    std::vector<int> p_prime_list(6, 52);
    p_prime_list.insert(p_prime_list.begin(), 52);

    context.set_coeff_modulus_bit_sizes(q_prime_list, p_prime_list);

    context.generate();
    context.print_parameters();

    getCurrentMemoryUsage();
    heongpu::HEKeyGenerator<Scheme> keygen(context);

    // Hamming weight of secret key is 32
    secret_key = std::make_unique<heongpu::Secretkey<Scheme>>(context, 32);
    keygen.generate_secret_key(*secret_key);

    // Hamming weight of sparse key is 32
    heongpu::Secretkey<Scheme> sparse_key(context, 32);
    keygen.generate_secret_key(sparse_key);

    heongpu::Switchkey<Scheme> switch_key_d2s(context);
    keygen.generate_switch_key(switch_key_d2s, sparse_key, *secret_key);
    heongpu::Switchkey<Scheme> switch_key_s2d(context);
    keygen.generate_switch_key(switch_key_s2d, *secret_key, sparse_key);

    // heongpu::Publickey<Scheme> public_key(context);
    public_key = std::make_unique<heongpu::Publickey<Scheme>>(context);
    keygen.generate_public_key(*public_key, *secret_key);

    // heongpu::Relinkey<Scheme> relin_key(context);
    relin_key = std::make_unique<heongpu::Relinkey<Scheme>>(context);
    keygen.generate_relin_key(*relin_key, *secret_key);

    key_memory_usage = getCurrentMemoryUsage();
    // heongpu::Galoiskey<Scheme> galois_key(context);

    std::vector<int> shifts;
    for (int i = 0; i < MAX_SHIFT; i++) {
      int power = pow(2, i);
      shifts.push_back(power);
      shifts.push_back(-power);
    }
    for (int i = 0; i < 16; i++) {
      for (int j : {1, 32, 1024}) {
        shifts.push_back(i * j);
      }
    }
    galois_key = std::make_unique<heongpu::Galoiskey<Scheme>>(context, shifts);

    keygen.generate_galois_key(
        *galois_key,
        *secret_key); // This way will create 16(2x8) different power
                      // of 2, if you need more change from define.h
    cudaDeviceSynchronize();
    this->context = std::make_unique<heongpu::HEContext<Scheme>>(context);
    encoder = std::make_unique<heongpu::HEEncoder<Scheme>>(context);
    encryptor =
        std::make_unique<heongpu::HEEncryptor<Scheme>>(context, *public_key);
    decryptor =
        std::make_unique<heongpu::HEDecryptor<Scheme>>(context, *secret_key);
    operators = std::make_unique<heongpu::HEArithmeticOperator<Scheme>>(
        context, *encoder);

    // bootstrapping setting
    auto scale = std::pow(2, 52);

    galois_key->store_in_device();

    bootstrapper = std::make_unique<heongpu::Bootstrap>(
        context, *secret_key, *relin_key, *galois_key, switch_key_d2s,
        switch_key_s2d, 3, 1.0, scale);

    cudaDeviceSynchronize();
  }

  void loadClient(char *dir) {}

  void loadServer(char *dir) {}

  void loadConstants(char *name) {
    std::string sname(name);
    constData.load(sname);
  }

  void loadHEVM(char *name) {
    std::string sname(name);

    std::ifstream iff(sname, std::ios::binary);

    loadHeader(iff);
    hecate::RuntimeConfig runOptions;
    runOptions.debug.printOpTypes = PRINT_OPTYPES;
    runOptions.debug.printOpStats = PRINT_OPSTATS;
    runOptions.debug.printRange = PRINT_RANGE;
    runOptions.settings.usePreencode = USE_PREENCODE;
    setRuntimeConfig(runOptions);

    ops.resize(config.num_operations);
    iff.read((char *)ops.data(), ops.size() * sizeof(HEVMOperation));

    std::vector<std::complex<double>> datas;
    for (int i = 0; i < slot_size; i++) {
      datas.push_back(std::complex<double>(0.0, 0.0));
    }
    msgs.resize(config.num_ptxt_buffer);
    ciphers.resize(config.num_ctxt_buffer,
                   heongpu::Ciphertext<Scheme>(*context));
    scalep.resize(config.num_ptxt_buffer);
    levelp.resize(config.num_ptxt_buffer);

    if (USE_PREENCODE) {
      plains.resize(config.num_ptxt_buffer,
                    heongpu::Plaintext<Scheme>(*context));
    } else {
      plains.resize(1, heongpu::Plaintext<Scheme>(*context));
    }
    getCurrentMemoryUsage();
  }

  void loadHeader(std::istream &iff) {
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
                   heongpu::Ciphertext<Scheme>(*context));
  }

  void resetResDst() {
    for (int i = 0; i < header.config_header.res_length; i++) {
      res_dst[i] = i + header.config_header.arg_length;
    }
  }

  void preprocess() {
    std::vector<double> datas(slot_size, 0.0);
    std::vector<double> identity(slot_size, 1.0);
    for (HEVMOperation &op : ops) {
      if (op.opcode == 0) {
        levelp[op.dst] = op.rhs >> 10;
        scalep[op.dst] = op.rhs & 0x3FF;
        if (USE_PREENCODE) {
          encode_internal(plains[op.dst],
                          op.lhs == ((unsigned short)-1) ? identity
                                                         : constData[op.lhs],
                          levelp[op.dst], scalep[op.dst]);
          cudaDeviceSynchronize();
          for (int i = L; i > levelp[op.dst]; i--) {
            operators->mod_drop_inplace(plains[op.dst]);
          }

        } else {
          to_msg(op.dst, op.lhs);
        }
      }
    }
  }

  void to_msg(int16_t dst, uint16_t lhs) {

    std::vector<double> identity(slot_size, 1.0);

    if (!msgMap.count(lhs)) {
      msgMap[lhs] = Message(slot_size, 0.0);
      auto &msg = msgMap[lhs];
      auto &src = lhs == ((unsigned short)-1) ? identity : constData[lhs];
      for (size_t i = 0; i < msg.size(); i++) {
        msg[i] = src[i % src.size()];
        // msg[i].imag(0);
      }
    }
    msgs[dst] = &msgMap[lhs];
    return;
  }

  void encode_online(int16_t dst) override {
    // if (debug)
    // std::cout << scalep[dst] << " " << levelp[dst] << std::endl;
    // encoder->encode(plains[0], constData[dst], std::pow(2.0, scalep[dst]));
    encoder->encode(plains[0], *msgs[dst], std::pow(2.0, scalep[dst]),
                    L - levelp[dst]);
    // for (int i = L; i > levelp[dst]; i--) {
    // operators->mod_drop_inplace(plains[0]);
    // }
  }

  void encode_internal(heongpu::Plaintext<Scheme> &dst, std::vector<double> src,
                       int16_t level, uint64_t scale) {

    heongpu::HostVector<double> datas(slot_size, 0.0);
    for (int i = 0; i < datas.size(); i++) {
      datas[i] = src[i % src.size()];
    }

    encoder->encode(dst, datas, std::pow(2.0, scale));
    return;
  }

  void encode(int16_t dst, int16_t src, int8_t level, int8_t scale) override {
    return;
  }
  void rotate(int16_t dst, int16_t src, int16_t offset) override {

    ciphers[dst] = ciphers[src];

    // Adjust offset to be within the range of -slot_size to slot_size
    if (-slot_size <= offset && offset < -(slot_size / 2))
      offset += slot_size;
    else if ((slot_size / 2) <= offset && offset < slot_size)
      offset -= slot_size;

    operators->rotate_rows_inplace(ciphers[dst], *galois_key, offset);
  }
  void negate(int16_t dst, int16_t src) override {
    operators->negate(ciphers[src], ciphers[dst]);
  }
  void rescale(int16_t dst, int16_t src) override {
    // rescale_inplace can occur error
    ciphers[dst] = ciphers[src];
    operators->set_rescale_required(ciphers[dst], true);
    operators->rescale_inplace(ciphers[dst]);
  }
  void modswitch(int16_t dst, int16_t src, int16_t downFactor) override {
    ciphers[dst] = ciphers[src];
    for (int i = 0; i < downFactor; i++) {
      operators->mod_drop_inplace(ciphers[dst]);
    }
  }
  void upscale(int16_t dst, int16_t src, int16_t upFactor) override {
    assert(0 && "This VM does not support native upscale op");
  }
  void addcc(int16_t dst, int16_t lhs, int16_t rhs) override {
    operators->add(ciphers[lhs], ciphers[rhs], ciphers[dst]);
  }
  void addcp(int16_t dst, int16_t lhs, int16_t rhs) override {
    if (!USE_PREENCODE) {
      encode_online(rhs);
      operators->add_plain(ciphers[lhs], plains[0], ciphers[dst]);
    } else
      operators->add_plain(ciphers[lhs], plains[rhs], ciphers[dst]);
  }

  void mulcc(int16_t dst, int16_t lhs, int16_t rhs) override {
    operators->multiply(ciphers[lhs], ciphers[rhs], ciphers[dst]);
    operators->relinearize_inplace(ciphers[dst], *relin_key);
    operators->set_rescale_required(ciphers[dst], false);
  }
  void mulcp(int16_t dst, int16_t lhs, int16_t rhs) override {
    if (!USE_PREENCODE) {
      encode_online(rhs);
      operators->multiply_plain(ciphers[lhs], plains[0], ciphers[dst]);
    } else {
      operators->multiply_plain(ciphers[lhs], plains[rhs], ciphers[dst]);
    }
    operators->set_rescale_required(ciphers[dst], false);
  }
  void bootstrap(int16_t dst, int64_t src, uint64_t targetLevel) override {
    ciphers[dst] = ciphers[src];
    heongpu::Plaintext<Scheme> ptxt(*context);

    // TODO: fix the target level of bootstrapping
    bootstrapper->execute_hoisted(ciphers[dst], 27);
  }

  // Debugging functions
  hecate::msg_t decrypt(int64_t dst) override {
    heongpu::Plaintext<Scheme> ptxt(*context);
    decryptor->decrypt(ptxt, ciphers[dst]);
    hecate::msg_t msg(slot_size, 0.0);
    encoder->decode(msg, ptxt);
    return msg;
  }

  double getCipherScale(int16_t i) override {
    return std::log2(ciphers[i].scale());
  }
  double getPlainScale(int16_t i) override {
    return std::log2(plains[i].scale());
  }
  int getCipherLevel(int16_t i) override {
    return context->get_ciphertext_modulus_count() - ciphers[i].depth();
  }
  int getPlainLevel(int16_t i) override {
    return context->get_ciphertext_modulus_count() - plains[i].depth();
  }
};

extern "C" {
void *initFullVM(char *dir, int64_t N, int64_t L, bool device = false) {
  auto vm = new HEONGPU_HEVM(N, L);
  vm->loadHEONGPU(dir);
  return (void *)vm;
}
void *initClientVM(char *dir, int64_t N, int64_t L) {
  auto vm = new HEONGPU_HEVM(N, L);
  vm->loadClient(dir);
  return (void *)vm;
}
void *initServerVM(char *dir, int64_t N, int64_t L) {
  auto vm = new HEONGPU_HEVM(N, L);
  vm->loadServer(dir);
  return (void *)vm;
}

void create_context(char *dir) { HEONGPU_HEVM::create_context(dir); }

// Loader for server
void load(void *vm, char *constant, char *vmfile) {
  auto hevm = static_cast<HEONGPU_HEVM *>(vm);
  hevm->loadConstants(constant);
  hevm->loadHEVM(vmfile);
}

// Loader for client
void loadClient(void *vm, void *is) {
  auto hevm = static_cast<HEONGPU_HEVM *>(vm);
  std::istream &iss = *static_cast<std::istream *>(is);
  hevm->loadHeader(iss);
  hevm->resetResDst();
}

// set Epoch of loop
void setEpoch(void *vm, int64_t i, int epoch) {
  auto hevm = static_cast<HEONGPU_HEVM *>(vm);
  hevm->integers[i] = epoch;
}

// encryption and decryption uses internal buffer id
void encrypt(void *vm, int64_t i, double *dat, int len) {
  auto hevm = static_cast<HEONGPU_HEVM *>(vm);
  heongpu::Plaintext<Scheme> ptxt(*hevm->context);
  std::vector<double> dats(dat, dat + len);
  hevm->encode_internal(ptxt, dats, hevm->arg_level[i], hevm->arg_scale[i]);

  // TODO: Hide the visibleCiphers from the user
  if (hevm->runConfig.debug.printOpTypes) {
    for (int j = 0; j < hevm->slot_size; j++) {
      hevm->visibleCiphers[i][j] = dats[j % len];
    }
  }
  // ptxt should be zero depth before encryption in HEONGPU
  hevm->encryptor->encrypt(hevm->ciphers[i], ptxt);
  for (int l = hevm->L; l > hevm->arg_level[i]; l--) {
    hevm->operators->mod_drop_inplace(hevm->ciphers[i]);
  }
}
void decrypt(void *vm, int64_t i, double *dat) {
  auto hevm = static_cast<HEONGPU_HEVM *>(vm);
  heongpu::Plaintext<Scheme> ptxt(*hevm->context);
  hevm->decryptor->decrypt(ptxt, hevm->ciphers[i]);
  std::vector<double> msg(hevm->slot_size, 0.0);
  // hevm->encoder->decode(msg, ptxt, std::pow(2.0, hevm->scalep[i]));
  hevm->encoder->decode(msg, ptxt);
  for (int j = 0; j < hevm->slot_size; j++) {
    /* for (size_t j = 0; j < msg.getSize(); j++) */
    dat[j] = msg[j];
  }
}
// simple wrapper to elide getResIdx call
// use res_idx for i
void decrypt_result(void *vm, int64_t i, double *dat) {
  auto hevm = static_cast<HEONGPU_HEVM *>(vm);
  decrypt(vm, hevm->res_dst[i], dat);
}

// We need this for communication code to access the proper buffer id
int64_t getResIdx(void *vm, int64_t i) {
  auto hevm = static_cast<HEONGPU_HEVM *>(vm);
  return hevm->res_dst[i];
}

// use this to implement communication
void *getCtxt(void *vm, int64_t id) {
  auto hevm = static_cast<HEONGPU_HEVM *>(vm);
  return &(hevm->ciphers[id]);
}

void preprocess(void *vm) {
  auto hevm = static_cast<HEONGPU_HEVM *>(vm);
  hevm->preprocess();
}
void run(void *vm) {
  auto hevm = static_cast<HEONGPU_HEVM *>(vm);
  hevm->run(hevm->ops);
  if (hevm->runConfig.debug.printOpStats)
    hevm->printPerformanceStats();
}
int64_t getArgLen(void *vm) {
  auto hevm = static_cast<HEONGPU_HEVM *>(vm);
  return hevm->header.config_header.arg_length;
}
int64_t getResLen(void *vm) {
  auto hevm = static_cast<HEONGPU_HEVM *>(vm);
  return hevm->header.config_header.res_length;
}
void setDebug(void *vm, bool enable) {
  auto hevm = static_cast<HEONGPU_HEVM *>(vm);
  // hevm->debug = enable;
}
void setToGPU(void *vm, bool ongpu) {
  auto hevm = static_cast<HEONGPU_HEVM *>(vm);
}
void printMem(void *vm) {
  auto hevm = static_cast<HEONGPU_HEVM *>(vm);
  // hevm->printMemoryUsage();
}
};
