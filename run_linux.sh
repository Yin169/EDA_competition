#!/bin/bash

# Clean previous build
rm -rf build

# Create build directory
mkdir -p build

# Install dependencies with Conan
conan install . --build=missing 

# Configure and build with CMake
cd build
cmake -DUSE_SPARSE_LU=ON .. -DCMAKE_TOOLCHAIN_FILE=Release/generators/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release

# Build using all available cores
cmake --build . 

echo "Build completed successfully!"
cd .. 

export OMP_PROC_BIND=true
export OMP_PLACES=cores

for numthread in 32 16 8 4 2 1
do 
  for timescheme in CRANK_NICOLSON BACKWARD_EULER
  do
  ./build/levelset "data/initial_struct.bnd" "./out/" ${numthread} ${timescheme} > log_${numthread}_${timescheme}.txt
  break
  done
  break
done
