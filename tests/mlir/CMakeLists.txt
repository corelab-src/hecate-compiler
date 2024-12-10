set(HECATE_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(HECATE_BINARY_DIR ${CMAKE_BINARY_DIR})
set(HECATE_TOOLS_DIR ${HECATE_BINARY_DIR}/bin)
set(LLVM_TOOLS_DIR ${MLIR_ROOT}/bin)

set(CHECK_VERBOSE "")
#set(CHECK_VERBOSE "--verbose")

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.cfg.py
  @ONLY
)

function(add_mlir_test_targets test_dir)
  set(base_dir ${CMAKE_CURRENT_SOURCE_DIR}/${test_dir})
  file(GLOB_RECURSE mlir_files ${base_dir}/*.mlir)

  set(dir_targets "")
  set(file_targets "")

  foreach(mlir_file ${mlir_files})
    file(RELATIVE_PATH rel_path ${CMAKE_CURRENT_SOURCE_DIR} ${mlir_file})
    get_filename_component(test_dir ${rel_path} DIRECTORY)
    get_filename_component(file_name ${mlir_file} NAME_WE)

    string(REPLACE "/" "-" unique_file_target ${test_dir}-${file_name})
    add_custom_target(check-${unique_file_target}
      COMMAND FILECHECK_OPTS="--color" ${LLVM_EXTERNAL_LIT} -s ${CMAKE_CURRENT_BINARY_DIR}/${test_dir} --filter "${rel_path}" ${CHECK_VERBOSE}
      DEPENDS hecate-opt
    )

    list(APPEND file_targets ${file_name})
    list(APPEND dir_targets ${test_dir})
    while(test_dir)
      get_filename_component(parent_dir ${test_dir} DIRECTORY)
      if("${parent_dir}" STREQUAL "${test_dir}")
        break()
      endif()
      list(APPEND dir_targets ${parent_dir})
      set(test_dir ${parent_dir})
    endwhile()
  endforeach()

  list(REMOVE_DUPLICATES dir_targets)
  foreach(dir_target ${dir_targets})
    if(NOT dir_target STREQUAL "")
      string(REPLACE "/" "-" dir_target_name ${dir_target})
      add_custom_target(check-${dir_target_name}
        COMMAND FILECHECK_OPTS="--color" ${LLVM_EXTERNAL_LIT} -s ${CMAKE_CURRENT_BINARY_DIR}/${dir_target} ${CHECK_VERBOSE}
        DEPENDS hecate-opt
      )
    endif()
  endforeach()

  list(REMOVE_DUPLICATES file_targets)
  foreach(file_target ${file_targets})
    add_custom_target(check-${file_target}
      COMMAND FILECHECK_OPTS="--color" ${LLVM_EXTERNAL_LIT} -s ${CMAKE_CURRENT_BINARY_DIR} --filter "${file_target}.mlir" ${CHECK_VERBOSE}
      DEPENDS hecate-opt
    )
  endforeach()
endfunction()

add_custom_target(check-hecate
  COMMAND FILECHECK_OPTS="--color" ${LLVM_EXTERNAL_LIT} -s ${CMAKE_CURRENT_BINARY_DIR} ${CHECK_VERBOSE}
  DEPENDS hecate-opt
)

add_dependencies(check-hecate hecate-opt)

add_mlir_test_targets(Earth)
add_mlir_test_targets(CKKS)
#add_subdirectory(Earth)
#add_subdirectory(CKKS)
