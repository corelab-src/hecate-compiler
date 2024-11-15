#include <HEonGPU-1.0/heongpu.cuh>
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
#include <publickey.cuh>
#include <secretkey.cuh>
#include <type_traits>
#include <vector>

#include "hecate/Support/ConstData.h"
#include "hecate/Support/HEVMHeader.h"

struct HEONGPU_HEVM {
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

  std::vector<heongpu::Ciphertext> ciphers;
  std::vector<heongpu::Plaintext> plains;
  std::map<uint64_t, heongpu::Plaintext> upscale_const;

  /* heongpu::Parameters context; */
  std::unique_ptr<heongpu::Parameters> context;
  std::unique_ptr<heongpu::Secretkey> secret_key;
  std::unique_ptr<heongpu::Publickey> public_key;
  std::unique_ptr<heongpu::Galoiskey> galois_key;
  std::unique_ptr<heongpu::Relinkey> relin_key;
  std::unique_ptr<heongpu::HEEncoder> encoder;
  std::unique_ptr<heongpu::HEEncryptor> encryptor;
  std::unique_ptr<heongpu::HEOperator> operators;
  std::unique_ptr<heongpu::HEDecryptor> decryptor;

  static const int N = 15;
  static const int L = 14;

  bool debug = false;
  bool togpu = true;

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

    heongpu::Parameters context(
        heongpu::scheme_type::ckks,
        heongpu::keyswitching_type::KEYSWITHING_METHOD_II,
        heongpu::sec_level_type::sec128);

    context.set_poly_modulus_degree(1LL << N);
    std::vector<int> coeffs;
    for (int i = 0; i < L; i++) {
      coeffs.push_back(50);
    }
    context.set_coeff_modulus(coeffs, {50, 50});

    context.generate();
    heongpu::HEKeyGenerator keygen(context);

    secret_key = std::make_unique<heongpu::Secretkey>(context);
    keygen.generate_secret_key(*secret_key);

    public_key = std::make_unique<heongpu::Publickey>(context);
    keygen.generate_public_key(*public_key, *secret_key);

    relin_key = std::make_unique<heongpu::Relinkey>(context);
    keygen.generate_relin_key(*relin_key, *secret_key);
    relin_key->store_key_in_device();

    galois_key = std::make_unique<heongpu::Galoiskey>(context);
    keygen.generate_galois_key(*galois_key, *secret_key);
    galois_key->store_key_in_device();

    this->context = std::make_unique<heongpu::Parameters>(context);
    encoder = std::make_unique<heongpu::HEEncoder>(context);
    encryptor = std::make_unique<heongpu::HEEncryptor>(context, *public_key);
    decryptor = std::make_unique<heongpu::HEDecryptor>(context, *secret_key);
    operators = std::make_unique<heongpu::HEOperator>(context);
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

    ciphers.resize(config.num_ctxt_buffer, heongpu::Ciphertext(*context));

    plains.resize(config.num_ptxt_buffer, heongpu::Plaintext(*context));
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
                   heongpu::Ciphertext(*context));
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
      }
    }
  }

  void to_msg(int16_t dst, std::vector<double> src) {}
  void encode_online(int16_t dst) {}

  void encode_internal(heongpu::Plaintext &dst, std::vector<double> src,
                       int8_t level, uint64_t scale) {
    heongpu::HostVector<double> datas(1LL << (N - 1), 0.0);
    for (int i = 0; i < datas.size(); i++) {
      datas[i] = src[i % src.size()];
    }
    encoder->encode(dst, datas, std::pow(2.0, scale));
    for (int i = L - 1; i > level; i--) {
      operators->mod_drop_inplace(dst);
    }

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

    if (debug) {
      std::cout << std::log2(ciphers[lhs].scale())
                << std::log2(plains[rhs].scale()) << std::endl;
      std::cout << ciphers[lhs].depth() << " " << plains[rhs].depth() << '\n';
    }
    operators->add_plain(ciphers[lhs], plains[rhs], ciphers[dst]);
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
    if (debug) {
      std::cout << std::log2(ciphers[lhs].scale())
                << std::log2(plains[rhs].scale()) << std::endl;
      std::cout << ciphers[lhs].depth() << " " << plains[rhs].depth()
                << std::endl;
    }
    operators->multiply_plain(ciphers[lhs], plains[rhs], ciphers[dst]);
    operators->set_rescale_required(ciphers[dst], false);
  }
  void bootstrap(int16_t dst, int64_t src, uint64_t targetLevel) {}

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

// encryption and decryption uses internal buffer id
void encrypt(void *vm, int64_t i, double *dat, int len) {
  auto hevm = static_cast<HEONGPU_HEVM *>(vm);
  heongpu::Plaintext ptxt(*hevm->context);
  std::vector<double> dats(dat, dat + len);
  hevm->encode_internal(ptxt, dats, hevm->arg_level[i], hevm->arg_scale[i]);
  std::vector<Data> tmp_ptxt(ptxt.size());
  hevm->encryptor->encrypt(hevm->ciphers[i], ptxt);
}
void decrypt(void *vm, int64_t i, double *dat) {
  auto hevm = static_cast<HEONGPU_HEVM *>(vm);
  cudaDeviceSynchronize();
  heongpu::Plaintext ptxt(*hevm->context);
  cudaDeviceSynchronize();
  hevm->decryptor->decrypt(ptxt, hevm->ciphers[i]);
  cudaDeviceSynchronize();
  std::vector<double> msg(1LL << (HEONGPU_HEVM::N - 1), 0.0);
  hevm->encoder->decode(msg, ptxt);
  for (int i = 0; i < (1LL << (HEONGPU_HEVM::N - 1)); i++) {
    dat[i] = msg[i];
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
