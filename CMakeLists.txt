cmake_minimum_required(VERSION 3.14)
project(vdb2cvdb CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# CUDA
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES "75;86;89")

# OpenVDB
set(OPENVDB_STATIC_LIB /usr/local/lib64/libopenvdb.a)
set(OPENVDB_INCLUDE_DIR /usr/local/include)

# Dependencies
find_package(TBB REQUIRED COMPONENTS tbb)
find_library(BLOSC_LIB NAMES libblosc.a blosc PATHS /usr/lib64 /usr/local/lib64)
find_library(BOOST_IOSTREAMS_LIB NAMES libboost_iostreams.a boost_iostreams
    PATHS /opt/boost_1_87_0_static/lib)
find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)

add_executable(vdb2cvdb vdb2cvdb.cpp cuda_kmeans.cu cuda_kmeans.h)
target_include_directories(vdb2cvdb PRIVATE ${OPENVDB_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vdb2cvdb PRIVATE
    ${OPENVDB_STATIC_LIB}
    TBB::tbb
    ${BLOSC_LIB}
    ${BOOST_IOSTREAMS_LIB}
    ZLIB::ZLIB
    Threads::Threads
    CUDA::cudart
    CUDA::cublas
)
set_target_properties(vdb2cvdb PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
