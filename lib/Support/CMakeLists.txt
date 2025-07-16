add_library (ConstData
  STATIC
  ConstData.cpp
  BackendInterface.cpp
  )

find_package(ZLIB)
target_link_libraries(ConstData PUBLIC ZLIB::ZLIB)

add_mlir_dialect_library (HecateSupport
  ConstData.cpp
  BootstrapGen.cpp
  BackendInterface.cpp
 DEPENDS
  HecateEarthOpsIncGen
 HecateEarthTransformsIncGen
 LINK_LIBS
 PUBLIC
 EarthDialect
 EarthAnalysis
 MLIRIR
 ConstData
 # MLIRFuncDialect
 # MLIRSupport
 # MLIRIR
 # EarthDialect
 # EarthAnalysis
 #  ${dialect_libs}
 #  ${conversion_libs}
  )


