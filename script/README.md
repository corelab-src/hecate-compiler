# Default Directory Tree
After install projects through scripts, the default directory tree is below.
```bash
RootDir
├── HEaaN
├── SEAL
├── hecate-compiler
│   ├── build
│   ├── cmake
│   ├── docs
│   ├── examples
│   ├── include
│   ├── lib
│   ├── python
│   ├── script
│   └── tools
├── install
│   ├── MLIR
│   ├── SEAL
│   └── HEON
└── llvm-project
```

# initialize
Clone other projects (llvm-project, SEAL, HEonGPU library).
```bash
./Initialize.sh
```

Model Downloads (ResNet20, AlexNet, ...).
```bash
./Model_Download.sh
```

Build docker image.
```bash
IMAGE_NAME=hecate IMAGE_TAG=v1.0 TARGET=cuda13 ./script/Docker_Build.sh
```
> **Note**: The TARGET option specifies the CUDA toolkit version used in the Docker image.
Currently supported values are: cuda12, cuda13

Run docker container. Because the execution of existing container is able to reset, the script asks you to Reset (y/N). If you just want to execute the existing container, just press 'S'.
```bash
./Docker_Run.sh
```

# in docker,
Build all projects. You can also build a certain project.
```
./Build.sh all 120
```

Activate python environments and aliases.
```
source activate.sh
```

