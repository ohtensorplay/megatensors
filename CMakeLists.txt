cmake_minimum_required(VERSION 3.21)

project(megatensors LANGUAGES CXX)

option(MEGATENSORS_NATIVE_OPT "Build the MEGA C++ extension for the local CPU" ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)
find_library(ZSTD_LIBRARY REQUIRED NAMES zstd)

set(MEGATENSORS_CPP_SOURCES
    megatensors/cpp/ext.cpp
)

if(WIN32)
    list(APPEND MEGATENSORS_CPP_SOURCES megatensors/cpp/dstorage_reader.cpp)
endif()

pybind11_add_module(cpp MODULE ${MEGATENSORS_CPP_SOURCES})
target_compile_features(cpp PRIVATE cxx_std_17)
target_compile_definitions(cpp PRIVATE __MOD_NAME__=cpp)
target_include_directories(cpp PRIVATE megatensors/cpp)
target_link_libraries(cpp PRIVATE OpenSSL::Crypto ${ZSTD_LIBRARY})

set_property(TARGET cpp PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cpp PRIVATE -O3 -DNDEBUG -fno-semantic-interposition)
    if(MEGATENSORS_NATIVE_OPT)
        target_compile_options(cpp PRIVATE -march=native)
    endif()
elseif(MSVC)
    target_compile_options(cpp PRIVATE /O2 /DNDEBUG)
endif()

if(WIN32)
    target_compile_features(cpp PRIVATE cxx_std_20)
    target_compile_definitions(cpp PRIVATE _CRT_SECURE_NO_WARNINGS)

    if(DEFINED ENV{CUDA_HOME})
        set(_MEGATENSORS_CUDA_HOME "$ENV{CUDA_HOME}")
    elseif(DEFINED ENV{CUDA_PATH})
        set(_MEGATENSORS_CUDA_HOME "$ENV{CUDA_PATH}")
    endif()

    if(_MEGATENSORS_CUDA_HOME AND EXISTS "${_MEGATENSORS_CUDA_HOME}/include")
        target_include_directories(cpp PRIVATE "${_MEGATENSORS_CUDA_HOME}/include")
    endif()
endif()

install(TARGETS cpp LIBRARY DESTINATION megatensors)
