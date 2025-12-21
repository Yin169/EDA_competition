#!/bin/bash

# Clean previous build
rm -rf build

# Create build directory
mkdir -p build

# Install dependencies with Conan
conan install . --build=missing 


for solver in USE_GMRES
do
# Configure and build with CMake
cd build
cmake -D${solver}=ON .. -DCMAKE_TOOLCHAIN_FILE=Release/generators/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release

# Build using all available cores
cmake --build . 

echo "Build completed successfully!"
cd .. 

export OMP_PROC_BIND=true
export OMP_PLACES=cores

for numthread in 104 82 64 32 16 8 4 2 1
do 
  for timescheme in CRANK_NICOLSON
  do
  ./build/levelset "data/initial_struct.bnd" "./out/" ${numthread} ${timescheme} > ${solver}_${numthread}_${timescheme}.log
  done
done
done

for numthread in 104 82 64 32 16 8 4 2 1
do 
  for timescheme in RUNGE_KUTTA_3
  do
  ./build/levelset "data/initial_struct.bnd" "./out/" ${numthread} ${timescheme} > ${numthread}_${timescheme}.log
  done
done
