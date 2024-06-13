add_mlir_conversion_library(HecateEarthToEarth
 EarthToEarth.cpp

 DEPENDS
 HecateConversionPassIncGen

 LINK_COMPONENTS
 Core

 LINK_LIBS PUBLIC
 EarthDialect
 MLIRLLVMCommonConversion
 MLIRTransforms
 )
