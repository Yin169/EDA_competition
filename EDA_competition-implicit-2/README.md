# EDA_competition

## Project Overview

EDA_competition is a scientific computing project focused on simulating and analyzing geometric evolution using advanced numerical methods. The core of the project is the implementation of the Level Set Method (LSM), which is widely used for tracking interfaces and shapes in computational geometry, computer graphics, and physical simulations. The codebase leverages C++ and the CGAL library for robust mesh and geometry processing.

## Main Features

- **Level Set Method Implementation**: Evolving surfaces and interfaces using implicit representations.
- **Mesh Handling**: Loading, processing, and analyzing 3D polygon meshes.
- **Signed Distance Field (SDF) Initialization**: Efficient computation of the distance from grid points to the mesh surface.
- **Surface Extraction**: Generating and exporting surface meshes from the level set function.
- **Material Properties**: Support for heterogeneous materials with customizable etching and lateral ratios.
- **Parallelization**: Utilizes OpenMP for efficient computation on multi-core systems.

## Workflow and Structure

The main workflow is encapsulated in the `LevelSetMethod` class, which orchestrates the simulation pipeline:

1. **Mesh Loading**: 
   - The mesh is loaded from a file and validated for correctness (closed, non-empty, triangle mesh).
   - An Axis-Aligned Bounding Box (AABB) tree is constructed for fast spatial queries.

2. **Grid Generation**:
   - A 3D grid is generated to discretize the computational domain.
   - Each grid point represents a location in space where the level set function is evaluated.

3. **Signed Distance Field Initialization**:
   - For each grid point, the signed distance to the mesh surface is computed using the AABB tree.
   - The sign indicates whether the point is inside, outside, or on the boundary of the mesh.

4. **Level Set Evolution**:
   - The level set function (`phi`) is evolved over time using numerical schemes (e.g., Upwind, WENO, Forward Euler, Runge-Kutta).
   - Material properties and etching rates can be incorporated for more realistic simulations.
   - Reinitialization and narrow band techniques are used to maintain numerical stability and efficiency.

5. **Surface Extraction**:
   - The zero level set (interface) is extracted as a surface mesh using CGAL's implicit surface meshing.
   - Trilinear interpolation ensures smooth surface reconstruction from the grid data.

6. **Result Saving**:
   - The final surface mesh and simulation results are saved to files for visualization and analysis.

## Numerical Methods in Detail

### Level Set Method

The Level Set Method represents interfaces implicitly as the zero contour of a higher-dimensional function (`phi`). The evolution of the interface is governed by partial differential equations (PDEs), which are solved numerically on a fixed grid.

- **Initialization**: The signed distance field is initialized so that `phi(x) = 0` on the interface, negative inside, and positive outside.
- **Evolution Equation**: The interface evolves according to a velocity field, often derived from physical models or geometric properties (e.g., curvature).
- **Numerical Schemes**: The project supports multiple spatial and temporal discretization schemes, including Upwind, WENO, Forward Euler, Backward Euler, Crank-Nicolson, and Runge-Kutta.

### Grid and SDF

- **Grid Generation**: The computational domain is discretized into a regular 3D grid, with spacing and size configurable via the constructor.
- **Signed Distance Field**: For each grid point, the closest point on the mesh is found using the AABB tree, and the signed distance is computed. CGAL's `Side_of_triangle_mesh` is used to determine inside/outside status efficiently.

### Mesh Handling

- **AABB Tree**: Accelerates distance queries and inside/outside tests, crucial for initializing the SDF and for efficient simulation.
- **Surface Extraction**: The zero level set is extracted using CGAL's implicit surface meshing, which reconstructs a triangle mesh from the implicit function defined by `phi` and the grid.

### Material Properties

- The simulation supports multiple materials, each with its own etching and lateral ratios, allowing for heterogeneous simulations.

### Parallelization

- OpenMP is used to parallelize computationally intensive loops, such as SDF initialization and level set evolution, making the code scalable on modern hardware.

## Example Class Structure

```
LevelSetMethod
|
+--> generateGrid()
|
+--> loadMesh()
|    |
|    +--> Create AABB tree
|
+--> evolve()
|    |
|    +--> initializeSignedDistanceField()
|    |    |
|    |    +--> Use tree to check if points are inside mesh
|    |
|    +--> isOnBoundary()
|    +--> getIndex()
|    +--> reinitialize()
|         |
|         +--> isOnBoundary()
|         +--> getIndex()
|
+--> saveResult()
|    |
|    +--> Write results to file
|
+--> extractSurfaceMeshCGAL()
|
+--> Create LevelSetImplicitFunction
|    |
|    +--> Use phi and grid data
|    +--> getIndex() (indirectly)
```