cmake_minimum_required(VERSION 3.10)
project(ML_Compiler)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable ARM NEON
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)")
    add_compile_options(-march=armv8-a+simd)
endif()

# Add compiler optimization options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Add test program
add_executable(operators_test
    backEnd/arm_simd/operators_test.cpp
)

# Add include directories
target_include_directories(operators_test PRIVATE
    ${CMAKE_SOURCE_DIR}/backEnd/arm_simd
)
