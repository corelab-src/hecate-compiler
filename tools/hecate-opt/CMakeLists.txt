add_executable (hecate-opt
  HecatePipeline.cpp
  PrototypePipeline.cpp
  HecateOptMain.cpp
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)


target_link_libraries(hecate-opt
  HecateSupport
  MLIREmitCDialect
  MLIROptLib
  EarthDialect
  CKKSDialect
  EarthTransforms
  CKKSTransforms
  HecateCKKSCommonConversion
  HecateEarthToCKKS
  HecateCKKSToCKKS
  ${dialect_libs}
  ${conversion_libs}
  )

