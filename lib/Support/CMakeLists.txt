
add_mlir_dialect_library (HecateSupport
  ConstData.cpp
  BootstrapGen.cpp
 DEPENDS
  HecateEarthOpsIncGen
 HecateEarthTransformsIncGen
 LINK_LIBS
 PUBLIC
 EarthDialect
 EarthAnalysis
 MLIRIR
 # MLIRFuncDialect
 # MLIRSupport
 # MLIRIR
 # EarthDialect
 # EarthAnalysis
 #  ${dialect_libs}
 #  ${conversion_libs}
  )
