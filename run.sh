#!/bin/bash

# Detect operating system
OS="$(uname -s)"
echo "Detected OS: $OS"

# Set compiler variables
export CC="clang"
export CXX="clang++"

# OS-specific configurations
if [ "$OS" = "Darwin" ]; then
  # macOS specific configuration
  echo "Configuring for macOS..."
  
  # Check if libomp is installed
  if [ ! -d "/opt/homebrew/opt/libomp" ]; then
    echo "Error: libomp not found. Installing libomp..."
    brew install libomp
  fi
  
  # Add proper flags for OpenMP support on macOS
  export CFLAGS="-isysroot $(xcrun --show-sdk-path) -I/opt/homebrew/opt/libomp/include"
  export CXXFLAGS="-isysroot $(xcrun --show-sdk-path) -I/opt/homebrew/opt/libomp/include"
  export LDFLAGS="-L/opt/homebrew/opt/libomp/lib -lomp"

  
  # Set OpenMP library paths for macOS
  OPENMP_C_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include"
  OPENMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include"
  OPENMP_LIB_NAMES="omp"
  OPENMP_LIB_PATH="/opt/homebrew/opt/libomp/lib/libomp.dylib"
  CMAKE_PREFIX_PATH="/opt/homebrew/lib/cmake"
  CMAKE_OSX_FLAGS="-DCMAKE_OSX_SYSROOT=\"$(xcrun --show-sdk-path)\" -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0"
else
  echo "Unsupported operating system: $OS"
  exit 1
fi

rm -rf build

# Add specific CMake flags based on OS
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
      $CMAKE_OSX_FLAGS \
      -DCMAKE_C_COMPILER="$(which $CC)" \
      -DCMAKE_CXX_COMPILER="$(which $CXX)" \
      -DOpenMP_C_FLAGS="$OPENMP_C_FLAGS" \
      -DOpenMP_CXX_FLAGS="$OPENMP_CXX_FLAGS" \
      -DOpenMP_C_LIB_NAMES="$OPENMP_LIB_NAMES" \
      -DOpenMP_CXX_LIB_NAMES="$OPENMP_LIB_NAMES" \
      ${OPENMP_LIB_PATH:+-DOpenMP_omp_LIBRARY="$OPENMP_LIB_PATH"}

cmake --build build

for timescheme in BACKWARD_EULER CRANK_NICOLSON RUNGE_KUTTA_3
do 
  for numthread in 0 2 4 8 16 32 64
  do
  ./build/levelset "data/initial_struct.bnd" "./out/" ${numthread} ${timescheme}
  done
done

# ./build/test


           