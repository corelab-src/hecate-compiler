# **Testing Guide for Hecate Compiler**  

This guide explains how to **add**, **configure**, and **run tests** for the Hecate Compiler using **llvm-lit** and **FileCheck**.

---

## **1. Adding a New Test File**  

### **File Location**  
- Add your `.mlir` test files to the appropriate subdirectory in `tests/`.  


### **Directory Structure**  

```
hecate-compiler/
  ├── ...
  └── tests/
      └── mlir/
          ├── CMakeLists.txt
          ├── README.md
          ├── lit.cfg.py.in
          ├── Earth/
          │   ├── Analysis/
          │   │   └── ...
          │   ├── Transform/
          │   │   ├── LatencyEstimator.mlir
          │   │   └── ...
          │   └── Conversion/
          │       ├── convert_earth_to_ckks.mlir
          │       └── ...
          └── CKKS/
              └── Transform/
                  ├── RemoveLevel.mlir
                  └── ...
```

### **Test File Format (`.mlir`)**  

#### **1. Add a Test Directive**  

```mlir
// RUN: hecate-opt %s -convert-earth-to-ckks --ckks-config=%hecate_root/%ckks_config | FileCheck %s

module {
  func.func @test_func(%arg0: tensor<1x!earth.ci<47 * 11>>, %arg1: tensor<1x!earth.ci<47 * 11>>) -> tensor<1x!earth.ci<47 * 11>> {
    %0 = "earth.add"(%arg0, %arg1) : (tensor<1x!earth.ci<47 * 11>>, tensor<1x!earth.ci<47 * 11>>) -> tensor<1x!earth.ci<47 * 11>>
    return %0 : tensor<1x!earth.ci<47 * 11>>
  }
}
```

#### **2. Add FileCheck Directives**  

```mlir
// CHECK-LABEL: func.func @test_func(
// CHECK-SAME:   %[[ARG0:.*]]: tensor<1x!ckks.poly<2 * 2>>,
// CHECK-SAME:   %[[ARG1:.*]]: tensor<1x!ckks.poly<2 * 2>>
// CHECK-SAME: ) -> tensor<1x!ckks.poly<2 * 2>> {
// CHECK:   %[[DST:.*]] = tensor.empty() : tensor<1x!ckks.poly<2 * 2>>
// CHECK:   %[[RESULT:.*]] = "ckks.addcc"(%[[DST]], %[[ARG0]], %[[ARG1]]) : (tensor<1x!ckks.poly<2 * 2>>, tensor<1x!ckks.poly<2 * 2>>, tensor<1x!ckks.poly<2 * 2>>) -> tensor<1x!ckks.poly<2 * 2>>
// CHECK:   return %[[RESULT]] : tensor<1x!ckks.poly<2 * 2>>
// CHECK: }

```

#### **Important Note:**  
> We currently use a **profile-based configuration** for tests by passing `--ckks-config=%hecate_var/%ckks_config` to `hecate-opt`.  
> If you are using **different profiles**, **adjust the file path accordingly** or **update the configuration**  in `lit.cfg.py.in` to fit your environment.
- **`%hecate_root`**: The **project root directory**.
- **`%ckks_config`**: The **profiled file path** used for testing.


## **2. Running Tests**  


### **2.1 Run All Tests**  
Run **all tests**:  
```bash
cmake --build build --target check-hecate
```

#### **Enable Verbose Output (Optional)**  

To **enable detailed output**, update `CMakeLists.txt` by setting `CHECK_VERBOSE` to `--verbose`:

```cmake
# Enable verbose test output
set(CHECK_VERBOSE "--verbose")
```

To **disable verbose output**, set it to an empty string:  

```cmake
# Disable verbose test output (default)
set(CHECK_VERBOSE "")
```

### **2.2 Run Tests for a Specific Directory**  
Run **tests for a specific directory**:  
```bash
cmake --build build --target check-<Directory-Name>
```

#### **Examples:**  
- Run all tests in `Earth/Conversion`:  
  ```bash
  cmake --build build --target check-Earth-Conversion
  ```

- Run all tests in `CKKS`:  
  ```bash
  cmake --build build --target check-CKKS
  ```


### **2.3 Run a Specific Test File**  
Run **tests for a specific file**:  
```bash
cmake --build build --target check-<File-Name>
```

#### **Example:**  
Run the `convert_earth_to_ckks.mlir` test:  
```bash
cmake --build build --target check-convert_earth_to_ckks
```

## **3. Adding a New Test Directory**  

1. **Create a new directory** in `tests/mlir/`.  
2. **Add `.mlir` files** with `// RUN` and `// CHECK` directives.  
3. **Update `CMakeLists.txt`** to register the new test directory:  

```cmake
add_mlir_test_targets(NewTestDirectory)
```

4. **Rebuild and run tests:**  

```bash
cmake -S . -B build
cmake --build build --target check-hecate
```

## **4. Reference Pages**  

### **Official LLVM and FileCheck Documentation**  

- **llvm-lit Testing Framework:**  
  [llvm-lit Documentation](https://llvm.org/docs/CommandGuide/lit.html)

- **FileCheck Testing Tool:**  
  [FileCheck Reference Guide](https://llvm.org/docs/CommandGuide/FileCheck.html)



