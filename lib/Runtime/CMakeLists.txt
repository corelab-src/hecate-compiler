
add_library (SEAL_HEVM
  SHARED
  SEAL_HEVM.cpp)
target_link_libraries(SEAL_HEVM
  PUBLIC
  SEAL::seal
  )

add_library (HEAAN_HEVM
  SHARED
  HEAAN_HEVM.cpp)
target_link_libraries(HEAAN_HEVM
  PUBLIC
  HEaaN::HEaaN
  )

add_library(HEONGPU_HEVM
  SHARED
  HEONGPU_HEVM.cpp
  )
target_link_libraries(HEONGPU_HEVM
  PUBLIC
  HEonGPU::heongpu
  CUDA::cudart
  )

add_compile_definitions(BARRETT_64)
add_compile_definitions(FLOAT_64)
target_compile_options(HEONGPU_HEVM PUBLIC -frtti)
target_compile_definitions(HEONGPU_HEVM PRIVATE BARRETT_64)
target_compile_definitions(HEONGPU_HEVM PRIVATE FLOAT_64)
set_target_properties(HEONGPU_HEVM PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
