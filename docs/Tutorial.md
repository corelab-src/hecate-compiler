## Tutorial 

### Trace the example python file to Encrypted ARiTHmetic IR 

```bash
hc-trace <example-name>
```
e.g., 
```bash
hc-trace ResNet_mpcb
```

### Compile the traced Earth IR 

```bash
hc-opt <example-name> --opt <pars|dacapo> --waterline <waterline:integer> --library <library-name> --device <hardware-device>
```
e.g., 
```bash
hc-opt ResNet_mpcb --opt dacapo --waterline 40 --library HEONGPU --device GPU
```
This command will print like this:
```
===-------------------------------------------------------------------------===
                         ... Execution time report ...
===-------------------------------------------------------------------------===
  Total Execution Time: 18.1769 seconds
```

You can see the optimized code in "$hecate-compiler/examples/optimized/dacapo/ResNet_mpcb.40.earth.mlir"\

If you see an error message like "error: 'earth.bootstrap' op failed to infer returned types",\
just wait as it is in the normal compilation process.


### Test code
We are currently conducting experiments using a modified version of the HEonGPU library.

> **Note**: In addition, we have included the SEAL library, which does not natively support bootstrapping.
We've provided a wrapper (SEAL\_HEVM.cpp) that modifies the bootstrapping algorithm into the decryption+encryption form.
Please note that while this method allows you to test the results, it is not recommended from a privacy perspective.
It need to generate trace code that matches SEAL parameters: In /examples/benchmarks/ResNet.py, "nt" : 2 ** 16 ----> "nt" : 2 ** 14

If you want to test with the HEonGPU library, refer to the example below:
```bash
hc-test <example-name> --opt <pars|dacapo> --waterline <waterline:integer> --library <library-name> --device <hardware-device>
```
e.g., 
```bash
hc-test ResNet_mpcb --opt dacapo --waterline 40 --library HEONGPU --device GPU
```
This command will print like this:
```
==================================================
--------------- Runtime Option ------------------
compiler: dacapo
benchname: ResNet_mpcb
waterline: 40
library: HEONGPU
device: GPU
----------------- Test Results -------------------
code size (bytes): 166216
const size (bytes): 9258474
latency (s): 10.156035864
rms: 0.0009392255808502616
==================================================

```

