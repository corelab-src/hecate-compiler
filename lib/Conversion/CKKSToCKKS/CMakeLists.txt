
add_mlir_conversion_library(HecateCKKSToCKKS
 UpscaleToMulcp.cpp
 SCFToDpsSCF.cpp
 ArithToDpsArith.cpp

 DEPENDS
 HecateConversionPassIncGen

 LINK_COMPONENTS
 Core

 LINK_LIBS PUBLIC
 EarthDialect
 CKKSDialect
 MLIRLLVMCommonConversion
 MLIRTransforms
 )
