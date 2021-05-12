if ("${MPIEXEC_EXECUTABLE}" EQUAL "")
    tests(GROUPS correctness_distributed
          SOURCES distributed.cpp)
else()
    tests(GROUPS correctness_distributed
          COMMAND_PREFIX ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
          COMMAND_SUFFIX ${MPIEXEC_POSTFLAGS}
          SOURCES distributed.cpp)
endif()

target_link_libraries(correctness_distributed_distributed PRIVATE MPI::MPI_CXX)

function(halide_define_aot_test NAME)
    set(options OMIT_DEFAULT_GENERATOR)
    set(oneValueArgs FUNCTION_NAME)
    set(multiValueArgs GEN_DEPS EXTRA_LIBS ENABLE_IF FEATURES PARAMS GEN_TARGET)
    cmake_parse_arguments(args "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (args_ENABLE_IF AND NOT (${args_ENABLE_IF}))
        return()
    endif ()

    add_executable("${NAME}.generator" "${NAME}_generator.cpp")

    target_link_libraries("${NAME}.generator" PRIVATE Halide::Generator ${args_GEN_DEPS})

    set(TARGET "generator_aot_${NAME}")
    set(DEPS ${args_EXTRA_LIBS})
    if (NOT args_OMIT_DEFAULT_GENERATOR)
        add_halide_library(${NAME}
                           FROM "${NAME}.generator"
                           PARAMS "${args_PARAMS}"
                           TARGETS "${args_GEN_TARGET}"
                           FUNCTION_NAME "${args_FUNCTION_NAME}"
                           FEATURES "${args_FEATURES}")
        list(APPEND DEPS ${NAME} ${NAME}.runtime)
    endif ()

    add_executable("${TARGET}" "${NAME}_aottest.cpp")
    target_include_directories("${TARGET}" PRIVATE
        "${Halide_SOURCE_DIR}/test/common"
        "${Halide_SOURCE_DIR}/tools")
    if (NOT args_OMIT_DEFAULT_GENERATOR)
        target_link_libraries(${TARGET} PRIVATE ${NAME})
    endif ()

    # TODO(#4938): remove need for these definitions
    if ("${Halide_TARGET}" MATCHES "opencl")
        target_compile_definitions("${TARGET}" PRIVATE TEST_OPENCL)
    endif ()
    if ("${Halide_TARGET}" MATCHES "metal")
        target_compile_definitions("${TARGET}" PRIVATE TEST_METAL)
    endif ()
    if ("${Halide_TARGET}" MATCHES "cuda")
        target_compile_definitions("${TARGET}" PRIVATE TEST_CUDA)
    endif ()

    target_link_libraries("${TARGET}" PRIVATE MPI::MPI_CXX)

    if ("${MPIEXEC_EXECUTABLE}" EQUAL "")
        add_halide_test("${TARGET}" GROUPS distributed_generator)
    else()
        add_halide_test("${TARGET}"
                        COMMAND "${MPIEXEC_EXECUTABLE}"
                        ARGS ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} "${TARGET}" ${MPIEXEC_POSTFLAGS}
                        GROUPS distributed_generator)
    endif()
endfunction()

halide_define_aot_test(distributed)
