# hecate-compiler
Hecate (Homomorphic Encryption Compiler for Approximate TEnsor computation) is an optimizing compiler for the CKKS FHE scheme, built by Compiler Optimization Research Laboratory (Corelab) @ Yonsei University. 
Hecate is built on the top of Multi-Level Intermediate Representation (MLIR) compiler framework. 
We aim to support privacy-preserving machine learning and deep learning applications. 

## Quick Start Guides

   * [Installation](docs/Installation.md)
    <!-- + [Requirements](docs/Installation.md#requirements) -->
    <!-- + [Install MLIR](docs/Installation.md#install-mlir) -->
    <!-- + [Install SEAL](docs/Installation.md#install-seal) -->
    <!-- + [Install HEonGPU](docs/Installation.md#install-heongpu) -->
    <!-- + [Build Hecate](docs/Installation.md#build-hecate) -->
    <!-- + [Configure Hecate](docs/Installation.md#configure-hecate) -->
    <!-- + [Install Hecate Python Binding](docs/Installation.md#install-hecate-python-binding) -->
   * [Tutorial](docs/Tutorial.md)
    <!-- + [Trace the example python file to Encrypted ARiTHmetic IR](docs/Tutorial.md#trace-the-example-python-file-to-encrypted-arithmetic-ir) -->
    <!-- + [Compile the traced Earth Hecate IR](docs/Tutorial.md#compile-the-traced-earth-ir) -->
    <!-- + [Test the optimized code](docs/Tutorial.md#test-code) -->
    <!-- + [One-liner for compilation and testing](#one-liner-for-compilation-and-testing) -->
   * [Docs](docs/)
      + [Docker Environment Setting](docs/Docker.md)
      + [Support Operations](docs/SupportedOps.md)
      + [How to MLIR Pass Test](docs/MLIRtest.md)
  
## Papers 
**HALO: Loop-aware Bootstrapping Management for Fully Homomorphic Encryption**\
Seonyoung Cheon, Yongwoo Lee, Hoyun Youm, Dongkwan Kim, Sungwoo Yun, Kunmo Jeong, Dongyoon Lee, and Hanjun Kim
*Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating System (ASPLOS)*, April 2025. 
[[Publication](https://dl.acm.org/doi/10.1145/3669940.3707275)]
```bibtex
@INPROCEEDINGS{cheon:halo:asplos,
  title = {HALO: Loop-aware Bootstrapping Management for Fully Homomorphic Encryption},
  author = {Cheon, Seonyoung and Lee, Yongwoo and Youm, Hoyun and Kim, Dongkwan and Yun, Sungwoo and Jeong, Kunmo and Lee, Dongyoon and Kim, Hanjun},
  booktitle = {Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1},
  year = {2025},
  publisher = {Association for Computing Machinery},
  doi = {10.1145/3669940.3707275},
  pages = {572–585},
  numpages = {14},
  location = {Rotterdam, Netherlands},
  series = {ASPLOS '25}
}
```

**DaCapo: Automatic Bootstrapping Management for Efficient Fully Homomorphic Encryption**\
Seonyoung Cheon, Yongwoo Lee, Ju Min Lee, Dongkwan Kim, Sunchul Jung, Taekyung Kim, Dongyoon Lee, and Hanjun Kim  
*33rd USENIX Security Symposium (USENIX Security)*, August 2024. 
[[Publication](https://www.usenix.org/system/files/usenixsecurity24-cheon.pdf)]
```bibtex
@INPROCEEDINGS{cheon:dacapo:sec,
  title={{DaCapo}: Automatic Bootstrapping Management for Efficient Fully Homomorphic Encryption},
  author={Cheon, Seonyoung and Lee, Yongwoo and Kim, Dongkwan and Lee, Ju Min and Jung, Sunchul and Kim, Taekyung and Lee, Dongyoon and Kim, Hanjun},
  booktitle={{33rd} USENIX Security Symposium (USENIX Security 24)},
 year={2024},
 address = {Philadelphia, CA},
 publisher = {USENIX Association},
 month = aug
}
```

**ELASM: Error-Latency-Aware Scale Management for Fully Homomorphic Encryption** [[abstract](https://www.usenix.org/conference/usenixsecurity23/presentation/lee-yongwoo)]   
Yongwoo Lee, Seonyoung Cheon, Dongkwan Kim, Dongyoon Lee, and Hanjun Kim  
*32nd USENIX Security Symposium (USENIX Security)*, August 2023. 
[[Publication](https://www.usenix.org/system/files/usenixsecurity23-lee-yongwoo.pdf)]
```bibtex
@INPROCEEDINGS{lee:elasm:sec,
  title={{ELASM}: Error-Latency-Aware Scale Management for Fully Homomorphic Encryption},
  author={Lee, Yongwoo and Cheon, Seonyoung and Kim, Dongkwan and Lee, Dongyoon and Kim, Hanjun},
  booktitle={{32nd} USENIX Security Symposium (USENIX Security 23)},
 year={2023},
 address = {Anaheim, CA},
 publisher = {USENIX Association},
 month = aug
}
```


**HECATE: Performance-Aware Scale Optimization for Homomorphic Encryption Compiler**\[[IEEE Xplore](http://doi.org/10.1109/CGO53902.2022.9741265)]   
Yongwoo Lee, Seonyeong Heo, Seonyoung Cheon, Shinnung Jeong, Changsu Kim, Eunkyung Kim, Dongyoon Lee, and Hanjun Kim  
*Proceedings of the 2022 International Symposium on Code Generation and Optimization (CGO)*, April 2022. 
[[Publication](http://corelab.or.kr/Pubs/cgo22_hecate.pdf)]
```bibtex
@INPROCEEDINGS{lee:hecate:cgo,
  author={Lee, Yongwoo and Heo, Seonyeong and Cheon, Seonyoung and Jeong, Shinnung and Kim, Changsu and Kim, Eunkyung and Lee, Dongyoon and Kim, Hanjun},
  booktitle={2022 IEEE/ACM International Symposium on Code Generation and Optimization (CGO)}, 
  title={HECATE: Performance-Aware Scale Optimization for Homomorphic Encryption Compiler}, 
  year={2022},
  volume={},
  number={},
  pages={193-204},
  doi={10.1109/CGO53902.2022.9741265}}
```



