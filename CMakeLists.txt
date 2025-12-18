cmake_minimum_required(VERSION 3.10)
project(EDA_competition)

option(USE_EIGEN_SOLVER "Use Eigen BiCGSTAB solver" OFF)
option(USE_GMRES "Use GMRES with ILU preconditioner" OFF)
option(USE_SPARSE_LU "Use SparseLU solver" OFF)
option(USE_DDGMRES "Use Domain Decomposition GMRES" OFF)

if(USE_EIGEN_SOLVER)
    add_definitions(-DEIGEN_SOLVER)
endif()
if(USE_GMRES)
    add_definitions(-DGMRES)
endif()
if(USE_SPARSE_LU)
    add_definitions(-DSpLU)
endif()
if(USE_DDGMRES)
    add_definitions(-DDDGMRES)
endif()

# 设置静态链接
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_EXE_LINKER_FLAGS "-static")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 强制静态链接标准库
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++")
endif()

# 寻找静态库
find_package(CGAL REQUIRED)
find_package(Eigen3 REQUIRED)

# Find and configure OpenMP for multi-threading support
find_package(OpenMP REQUIRED)

add_executable(levelset src/levelset_main.cpp)

# 设置目标属性为静态链接
set_target_properties(levelset PROPERTIES LINK_SEARCH_START_STATIC 1)
set_target_properties(levelset PROPERTIES LINK_SEARCH_END_STATIC 1)

# 链接库
target_link_libraries(levelset Eigen3::Eigen CGAL::CGAL)
if(OpenMP_CXX_FOUND)
    target_link_libraries(levelset OpenMP::OpenMP_CXX)
endif()

# 额外的静态链接设置
if(UNIX AND NOT APPLE)
    target_link_libraries(levelset -static-libgcc -static-libstdc++)
endif()

target_compile_options(levelset PRIVATE -O3 -march=native -Wall -Wextra -fopenmp -mavx512f -funroll-loops -ffast-math)