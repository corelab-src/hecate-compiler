
add_library (SEAL_HEVM
 SHARED
 SEAL_HEVM.cpp)
target_link_libraries(SEAL_HEVM
 PUBLIC
 SEAL::seal
  HecateSupport
 )


add_library (HEAAN_HEVM
  SHARED
  HEAAN_HEVM.cpp)
target_link_libraries(HEAAN_HEVM
  PUBLIC
  HEaaN::HEaaN
  HecateSupport
  )

add_library(HEONGPU_HEVM
  SHARED
  HEONGPU_HEVM.cpp
  )
target_link_libraries(HEONGPU_HEVM
	PUBLIC
  HEonGPU::heongpu
  CUDA::cudart
  HecateSupport
  OpenMP::OpenMP_CXX
  )

target_compile_definitions(HEONGPU_HEVM PRIVATE BARRETT_64)
target_compile_definitions(HEONGPU_HEVM PRIVATE FLOAT_64)
target_compile_options(HEONGPU_HEVM 
	PUBLIC -frtti
	# PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>
	)
# target_compile_options(HEONGPU_HEVM PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp)
# target_compile_options(${exe} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)
target_compile_definitions(HEONGPU_HEVM PRIVATE BARRETT_64)
target_compile_definitions(HEONGPU_HEVM PRIVATE FLOAT_64)
set_target_properties(HEONGPU_HEVM PROPERTIES 
	CUDA_SEPARABLE_COMPILATION ON
	# POSITION_INDEPENDENT_CODE ON
	# CUDA_RUNTIME_LIBRARY Static
	)
