cmake_minimum_required(VERSION 3.10)
project(EDA_competition)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # For Clang (macOS default compiler)
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif()

# Ensure debug symbols are included
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g")

find_package(CGAL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CERES REQUIRED)
find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(levelset src/levelset_main.cpp)
target_link_libraries(levelset ${EIGEN3_LIBRARIES} ${CGAL_LIBRARIES} ${CERES_LIBRARIES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(levelset OpenMP::OpenMP_CXX)
endif()


add_executable(test test/test.cpp)
target_link_libraries(test ${CGAL_LIBRARIES} ${EIGEN3_LIBRARIES}) 

# add_executable(main src/main.cpp src/DFISEParser.hpp)

# add_executable(alphawrap src/alphawrap.cpp src/output_helper.h)
# target_link_libraries(alphawrap ${CGAL_LIBRARIES})