cmake_minimum_required(VERSION 3.14)
project(hw2 LANGUAGES C CXX)

set(CMAKE_BUILD_TYPE "Release")

find_package(BLAS REQUIRED)
find_package(OpenMP REQUIRED)

# We require the GNU compiler for this assignment.
if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(Prg-Intel "PrgEnv-intel")
    set(Prg-Clang "PrgEnv-cray")
    message(WARNING
            "Must use GNU Compiler for submission. Make sure you ran:\n"
            "module swap ${Prg-${CMAKE_C_COMPILER_ID}} PrgEnv-gnu")
endif ()

# LIKWID support
option(USE_LIKWID "Use LIKWID for performance monitoring" ON)
if(USE_LIKWID)
    find_path(LIKWID_INCLUDE_DIR NAMES likwid.h PATHS ENV LIKWID_INCLUDE REQUIRED)
    find_library(LIKWID_LIBRARY NAMES likwid PATHS ENV LIKWID_LIB REQUIRED)
    add_compile_definitions(LIKWID_PERFMON)
    include_directories(${LIKWID_INCLUDE_DIR})
endif()

# Common library target for benchmarking.
add_library(benchmark OBJECT benchmark.cpp)
target_compile_features(benchmark PRIVATE cxx_std_11)
if(USE_LIKWID)
    target_include_directories(benchmark PRIVATE ${LIKWID_INCLUDE_DIR})
endif()

# Basic implementation with OpenMP
add_executable(benchmark-basic-omp dgemm-basic.cpp)
target_link_libraries(benchmark-basic-omp PRIVATE benchmark ${BLAS_LIBRARIES} OpenMP::OpenMP_CXX)
if(USE_LIKWID)
    target_link_libraries(benchmark-basic-omp PRIVATE ${LIKWID_LIBRARY})
endif()
target_compile_features(benchmark-basic-omp PRIVATE c_std_11 c_restrict)
target_compile_options(benchmark-basic-omp PRIVATE -Wall -pedantic -march=native)
set(BENCHMARK "basic-omp")
configure_file(job.in job-${BENCHMARK})

# BLAS reference (serial only)
add_executable(benchmark-blas dgemm-blas.cpp)
target_link_libraries(benchmark-blas PRIVATE benchmark ${BLAS_LIBRARIES})
if(USE_LIKWID)
    target_link_libraries(benchmark-blas PRIVATE ${LIKWID_LIBRARY})
endif()
target_compile_features(benchmark-blas PRIVATE c_std_11 c_restrict)
target_compile_options(benchmark-blas PRIVATE -Wall -pedantic -march=native)
set(BENCHMARK "cblas")
configure_file(job.in job-${BENCHMARK})

# Blocked implementation with OpenMP
add_executable(benchmark-blocked-omp dgemm-blocked.cpp)
add_library(bench-blocked OBJECT benchmark.cpp)
target_compile_definitions(bench-blocked PRIVATE -DBLOCKED)
target_compile_features(bench-blocked PRIVATE cxx_std_11)
if(USE_LIKWID)
    target_include_directories(bench-blocked PRIVATE ${LIKWID_INCLUDE_DIR})
endif()
target_link_libraries(benchmark-blocked-omp PRIVATE bench-blocked ${BLAS_LIBRARIES} OpenMP::OpenMP_CXX)
if(USE_LIKWID)
    target_link_libraries(benchmark-blocked-omp PRIVATE ${LIKWID_LIBRARY})
endif()
target_compile_features(benchmark-blocked-omp PRIVATE c_std_11 c_restrict)
target_compile_options(benchmark-blocked-omp PRIVATE -Wall -pedantic -march=native)
set(BENCHMARK "blocked-omp")
configure_file(job.in job-${BENCHMARK})