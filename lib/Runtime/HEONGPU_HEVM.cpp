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

#include "hecate/Support/ConstData.h"
#include "hecate/Support/HEVMHeader.h"

constexpr auto Scheme = heongpu::Scheme::CKKS;
using Message = std::vector<double>;
struct HEONGPU_HEVM {
  /* std::vector<std::vector<double>> buffer; */
  hecate::ConstData constData;
  HEVMHeader header;
  ConfigBody config;
  /* std::vector<uint64_t> config_dats; */
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

  /* heongpu::Parameters context; */
  std::unique_ptr<heongpu::HEContext<Scheme>> context;
  std::unique_ptr<heongpu::Secretkey<Scheme>> secret_key;
  std::unique_ptr<heongpu::Publickey<Scheme>> public_key;
  std::unique_ptr<heongpu::Galoiskey<Scheme>> galois_key;
  std::unique_ptr<heongpu::Relinkey<Scheme>> relin_key;
  std::unique_ptr<heongpu::HEEncoder<Scheme>> encoder;
  std::unique_ptr<heongpu::HEEncryptor<Scheme>> encryptor;
  std::unique_ptr<heongpu::HEArithmeticOperator<Scheme>> operators;
  std::unique_ptr<heongpu::HEDecryptor<Scheme>> decryptor;
  std::unique_ptr<heongpu::Bootstrap> booter;

  static const int N = 16;
  // static const int L = 18;
  static const int L = 28;
  // static const int N = 15;
  // static const int L = 14;

  bool debug = false;
  bool togpu = true;
  bool preencode = false;

  static void create_context(char *dir) {}

  void printCudaMemInfo() {
    size_t free_mem, total_mem;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);

    if (err != cudaSuccess) {
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
      return;
    }
    std::cout << "Free Memory: " << free_mem / (1024.0 * 1024.0) << " MB"
              << std::endl;
    std::cout << "Total Memory: " << total_mem / (1024.0 * 1024.0) << " MB"
              << std::endl;
  }
  void printInfo() {}

  void loadHEONGPU(char *dir) {
    cudaSetDevice(0);
    auto strdir = std::string(dir);

    heongpu::HEContext<Scheme> context(
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_II,
        // heongpu::sec_level_type::sec128);
        heongpu::sec_level_type::none);

    context.set_poly_modulus_degree(1LL << N);

    std::vector<int> q_prime_list(26, 52);
    q_prime_list.insert(q_prime_list.begin(), 52);
    // std::vector<int> q_prime_list(27, 52);
    std::vector<int> p_prime_list(6, 52);
    p_prime_list.insert(p_prime_list.begin(), 52);

    context.set_coeff_modulus_bit_sizes(q_prime_list, p_prime_list);

    context.generate();
    context.print_parameters();

    heongpu::HEKeyGenerator<Scheme> keygen(context);

    // heongpu::Secretkey<Scheme> secret_key(context, 32);
    secret_key = std::make_unique<heongpu::Secretkey<Scheme>>(context, 32);
    keygen.generate_secret_key(*secret_key);

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

    // heongpu::Galoiskey<Scheme> galois_key(context);

    galois_key = std::make_unique<heongpu::Galoiskey<Scheme>>(context);
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

    booter = std::make_unique<heongpu::Bootstrap>(
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

    ops.resize(config.num_operations);
    iff.read((char *)ops.data(), ops.size() * sizeof(HEVMOperation));

    auto log_slot = N - 1;
    std::vector<std::complex<double>> datas;
    for (int i = 0; i < (1 << log_slot); i++) {
      datas.push_back(std::complex<double>(0.0, 0.0));
    }
    msgs.resize(config.num_ptxt_buffer);
    ciphers.resize(config.num_ctxt_buffer,
                   heongpu::Ciphertext<Scheme>(*context));
    scalep.resize(config.num_ptxt_buffer);
    levelp.resize(config.num_ptxt_buffer);

    if (preencode) {
      plains.resize(config.num_ptxt_buffer,
                    heongpu::Plaintext<Scheme>(*context));
    } else {
      plains.resize(1, heongpu::Plaintext<Scheme>(*context));
    }
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
    std::vector<double> datas(1LL << (N - 1), 0.0);
    std::vector<double> identity(1LL << (N - 1), 1.0);
    for (HEVMOperation &op : ops) {
      if (op.opcode == 0) {
        if (preencode) {
          if (debug) {
            std::cout << std::endl;
            std::cout << "encode \n";
            std::cout << "opcode [" << op.opcode << "], dst [" << op.dst
                      << "], lhs [" << op.lhs << "], rhs [" << op.rhs << "]"
                      << std::endl;
          }
          encode_internal(plains[op.dst],
                          op.lhs == ((unsigned short)-1) ? identity
                                                         : constData[op.lhs],
                          op.rhs >> 10, op.rhs & 0x3FF);
        } else {
          to_msg(op.dst, op.lhs);
        }
        levelp[op.dst] = op.rhs >> 10;
        scalep[op.dst] = op.rhs & 0x3FF;
      }
    }
  }

  void to_msg(int16_t dst, uint16_t lhs) {

    std::vector<double> identity(1LL << (N - 1), 1.0);

    if (!msgMap.count(lhs)) {
      msgMap[lhs] = Message(std::pow(2, N - 1), 0.0);
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

  void encode_online(int16_t dst) {
    if (debug)
      std::cout << scalep[dst] << " " << levelp[dst] << std::endl;
    // encoder->encode(plains[0], constData[dst], std::pow(2.0, scalep[dst]));
    encoder->encode(plains[0], *msgs[dst], std::pow(2.0, scalep[dst]));
    for (int i = L - 1; i > levelp[dst]; i--) {
      operators->mod_drop_inplace(plains[0]);
    }
  }

  void encode_internal(heongpu::Plaintext<Scheme> &dst, std::vector<double> src,
                       int8_t level, uint64_t scale) {
    if (debug) {
      std::cout << scale << " " << level << std::endl;
    }
    heongpu::HostVector<double> datas(1LL << (N - 1), 0.0);
    for (int i = 0; i < datas.size(); i++) {
      datas[i] = src[i % src.size()];
    }
    encoder->encode(dst, datas, std::pow(2.0, scale));
    return;
  }

  void encode(int16_t dst, int16_t src, int8_t level, int8_t scale) { return; }
  void rotate(int16_t dst, int16_t src, int16_t offset) {
    if (debug)
      std::cout << std::log2(ciphers[src].scale()) << std::endl;
    ciphers[dst] = ciphers[src];
    operators->rotate_rows_inplace(ciphers[dst], *galois_key, offset);
    if (debug)
      std::cout << std::log2(ciphers[dst].scale()) << std::endl;
  }
  void negate(int16_t dst, int16_t src) {
    if (debug)
      std::cout << std::log2(ciphers[src].scale()) << std::endl;
    operators->negate(ciphers[src], ciphers[dst]);
  }
  void rescale(int16_t dst, int16_t src) {
    if (debug)
      std::cout << std::log2(ciphers[src].scale()) << std::endl;
    // rescale_inplace can occur error
    ciphers[dst] = ciphers[src];
    operators->set_rescale_required(ciphers[dst], true);
    operators->rescale_inplace(ciphers[dst]);
  }
  void modswitch(int16_t dst, int16_t src, int16_t downFactor) {
    if (debug) {
      std::cout << std::log2(ciphers[src].scale()) << std::endl;
      std::cout << ciphers[src].depth() << std::endl;
    }
    ciphers[dst] = ciphers[src];
    for (int i = 0; i < downFactor; i++) {
      operators->mod_drop_inplace(ciphers[dst]);
    }
    if (debug) {
      std::cout << ciphers[dst].depth() << std::endl;
    }
  }
  void upscale(int16_t dst, int16_t src, int16_t upFactor) {
    assert(0 && "This VM does not support native upscale op");
  }
  void addcc(int16_t dst, int16_t lhs, int16_t rhs) {
    if (debug) {
      std::cout << std::log2(ciphers[lhs].scale())
                << std::log2(ciphers[rhs].scale()) << std::endl;
      std::cout << ciphers[lhs].depth() << " " << ciphers[rhs].depth() << '\n';
    }
    operators->add(ciphers[lhs], ciphers[rhs], ciphers[dst]);
    if (debug)
      std::cout << std::log2(ciphers[dst].scale()) << std::endl;
  }
  void addcp(int16_t dst, int16_t lhs, int16_t rhs) {
    if (preencode) {
      if (debug) {
        std::cout << std::log2(ciphers[lhs].scale())
                  << std::log2(plains[rhs].scale()) << std::endl;
        std::cout << ciphers[lhs].depth() << " " << plains[rhs].depth() << '\n';
      }
      operators->add_plain(ciphers[lhs], plains[rhs], ciphers[dst]);
    } else {
      encode_online(rhs);
      if (debug) {
        std::cout << std::log2(ciphers[lhs].scale())
                  << std::log2(plains[0].scale()) << std::endl;
        std::cout << ciphers[lhs].depth() << " " << plains[0].depth() << '\n';
      }
      operators->add_plain(ciphers[lhs], plains[0], ciphers[dst]);
    }
  }

  void mulcc(int16_t dst, int16_t lhs, int16_t rhs) {
    if (debug)
      std::cout << std::log2(ciphers[lhs].scale())
                << std::log2(ciphers[rhs].scale()) << std::endl;
    operators->multiply(ciphers[lhs], ciphers[rhs], ciphers[dst]);
    operators->relinearize_inplace(ciphers[dst], *relin_key);
    operators->set_rescale_required(ciphers[dst], false);
  }
  void mulcp(int16_t dst, int16_t lhs, int16_t rhs) {
    if (preencode) {
      if (debug) {
        std::cout << std::log2(ciphers[lhs].scale())
                  << std::log2(plains[rhs].scale()) << std::endl;
        std::cout << ciphers[lhs].depth() << " " << plains[rhs].depth()
                  << std::endl;
      }
      operators->multiply_plain(ciphers[lhs], plains[rhs], ciphers[dst]);

    } else {
      encode_online(rhs);
      if (debug) {
        std::cout << std::log2(ciphers[lhs].scale())
                  << std::log2(plains[0].scale()) << std::endl;
        std::cout << ciphers[lhs].depth() << " " << plains[0].depth()
                  << std::endl;
      }
      operators->multiply_plain(ciphers[lhs], plains[0], ciphers[dst]);
    }
    operators->set_rescale_required(ciphers[dst], false);
  }
  void bootstrap(int16_t dst, int64_t src, uint64_t targetLevel) {
    if (debug) {
      std::cout << std::log2(ciphers[src].scale()) << std::endl;
      std::cout << ciphers[src].depth() << std::endl;
    }
    ciphers[dst] = ciphers[src];
    // auto gap = context.get()->get_ciphertext_modulus_count() -
    //            ciphers[dst].depth() - 1;
    // for (int i = 0; i < gap; i++) {
    //   operators->mod_drop_inplace(ciphers[dst]);
    // }
    ciphers[dst] = booter->execute(ciphers[dst]);

    if (debug) {
      std::cout << ciphers[dst].depth() << std::endl;
    }
  }

  void run() {
    int i = (header.hevm_header_size + config.config_body_length) / 8;
    int j = 0;
    /* cudaDeviceSynchronize(); */
    for (HEVMOperation &op : ops) {
      if (debug) {
        /* std::cout << std::oct << i++ << " " << std::dec << j++ << std::endl;
         */
        /* std::cout << op.opcode << " " << op.dst << " " << op.lhs << " " */
        /* << op.rhs << std::endl; */
        std::cout << std::endl;
        std::cout << std::oct << i++ << " " << std::dec << j++ << std::endl;
        std::cout << "opcode [" << op.opcode << "], dst [" << op.dst
                  << "], lhs [" << op.lhs << "], rhs [" << op.rhs << "]"
                  << std::endl;
      }
      switch (op.opcode) {
      case 0: { // Encode
        encode(op.dst, op.lhs, op.rhs >> 10, op.rhs & 0x3FF);
        break;
      }
      case 1: { // RotateC
        rotate(op.dst, op.lhs, op.rhs);
        break;
      }
      case 2: { // NegateC
        negate(op.dst, op.lhs);
        break;
      }
      case 3: { // RescaleC
        rescale(op.dst, op.lhs);
        break;
      }
      case 4: { // ModswtichC
        modswitch(op.dst, op.lhs, op.rhs);
        break;
      }
      case 5: { // UpscaleC
        upscale(op.dst, op.lhs, op.rhs);
        break;
      }
      case 6: { // AddCC
        addcc(op.dst, op.lhs, op.rhs);
        break;
      }
      case 7: { // AddCP
        addcp(op.dst, op.lhs, op.rhs);
        break;
      }
      case 8: { // MulCC
        mulcc(op.dst, op.lhs, op.rhs);
        break;
      }
      case 9: { // MulCP
        mulcp(op.dst, op.lhs, op.rhs);
        break;
      }
      case 10: { // Bootstrap
        bootstrap(op.dst, op.lhs, op.rhs);
        break;
      }
      default: {
        break;
      }
      }
    }
  }
  /* std::cout << "boot_time : " << boot_time << '\n'; */
  /* std::cout << "boot_cnt : " << boot_cnt << '\n'; */
};

extern "C" {
void *initFullVM(char *dir, bool device = false) {
  auto vm = new HEONGPU_HEVM();
  /* vm->togpu = device; */
  vm->loadHEONGPU(dir);
  return (void *)vm;
}
void *initClientVM(char *dir) {
  auto vm = new HEONGPU_HEVM();
  vm->loadClient(dir);
  return (void *)vm;
}
void *initServerVM(char *dir) {
  auto vm = new HEONGPU_HEVM();
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
  hevm->encryptor->encrypt(hevm->ciphers[i], ptxt);
  for (int l = HEONGPU_HEVM::L - 1; l > hevm->arg_level[i]; l--) {
    hevm->operators->mod_drop_inplace(hevm->ciphers[i]);
  }
}
void decrypt(void *vm, int64_t i, double *dat) {
  std::cout << "Decrypting " << i << std::endl;
  auto hevm = static_cast<HEONGPU_HEVM *>(vm);
  cudaDeviceSynchronize();
  heongpu::Plaintext<Scheme> ptxt(*hevm->context);
  cudaDeviceSynchronize();
  hevm->decryptor->decrypt(ptxt, hevm->ciphers[i]);
  cudaDeviceSynchronize();
  std::vector<double> msg(1LL << (HEONGPU_HEVM::N - 1), 0.0);
  std::cout << "Before Decode " << i << std::endl;
  hevm->encoder->decode(msg, ptxt);
  std::cout << "After Decode " << i << std::endl;
  for (int j = 0; j < (1LL << (HEONGPU_HEVM::N - 1)); j++) {
    /* for (size_t j = 0; j < msg.getSize(); j++) */
    dat[j] = msg[j];
  }
  std::cout << "End of Decrypt " << i << std::endl;
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
  hevm->run();
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
  hevm->debug = enable;
}
void setToGPU(void *vm, bool ongpu) {
  auto hevm = static_cast<HEONGPU_HEVM *>(vm);
  hevm->togpu = ongpu;
}
void printMem(void *vm) {
  auto hevm = static_cast<HEONGPU_HEVM *>(vm);
  hevm->printCudaMemInfo();
  hevm->printInfo();
}
};
