#ifndef LEVEL_SET_METHOD_HPP
#define LEVEL_SET_METHOD_HPP

#define EIGEN_USE_OPENMP
#define EIGEN_USE_MKL_ALL

#include "TimeScheme.hpp"

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Implicit_surface_3.h>
#include <CGAL/Surface_mesh_default_criteria_3.h>
#include <CGAL/make_surface_mesh.h>
#include <CGAL/Surface_mesh_complex_2_in_triangulation_3.h>
#include <CGAL/IO/facets_in_complex_2_to_triangle_mesh.h>
#include <CGAL/IO/polygon_mesh_io.h>

#include <CGAL/Surface_mesh_default_triangulation_3.h>
#include <CGAL/Complex_2_in_triangulation_3.h>
#include <CGAL/make_surface_mesh.h>
#include <CGAL/Surface_mesh.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <memory>
#include <vector>
#include <string>
#include <random>  // For std::random_device, std::mt19937, std::uniform_real_distribution
#include <functional>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <iostream>

namespace PMP = CGAL::Polygon_mesh_processing;

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point_3;
typedef CGAL::Surface_mesh<Point_3> Mesh;
typedef CGAL::AABB_face_graph_triangle_primitive<Mesh> Primitive;
typedef CGAL::AABB_traits<Kernel, Primitive> AABB_traits;
typedef CGAL::AABB_tree<AABB_traits> AABB_tree;

enum class TimeSchemeType {
    RUNGE_KUTTA_3,
    BACKWARD_EULER,
    CRANK_NICOLSON
};

class LevelSetMethod {
public:
    // Constructor that accepts a CSV file for material information
    LevelSetMethod(
                const std::string& meshFile,
                int gridSize = 400, 
                double timeStep = 0.01, 
                int maxSteps = 80, 
                int reinitInterval = 5,
                TimeSchemeType timeScheme = TimeSchemeType::BACKWARD_EULER,
                int numThreads = -1
            )
        : GRID_SIZE(gridSize),
        dt(timeStep),
        STEPS(maxSteps),
        REINIT_INTERVAL(reinitInterval),
        timeScheme(timeScheme){

        if (numThreads > 0) {
            Eigen::initParallel();
            omp_set_num_threads(numThreads);
            Eigen::setNbThreads(numThreads);
            std::cout << "Eigen N Threads: " << Eigen::nbThreads() << std::endl; 
        }
        
        // Load mesh and material information
        loadMesh(meshFile);
        generateGrid();
        phi = initializeSignedDistanceField();
        
        // Always use Backward Euler scheme for time integration
        solver = std::make_shared<implicitCN>(dt, GRID_SPACING);

        switch (timeScheme) {
            case TimeSchemeType::RUNGE_KUTTA_3:
                std::cout << "Using Runge-Kutta 3 scheme" << std::endl;
                solver = std::make_shared<TVDRK3RoeQUICKScheme>(dt, GRID_SPACING);
                break;
            case TimeSchemeType::BACKWARD_EULER:
                std::cout << "Using Backward Euler scheme" << std::endl;
                solver = std::make_shared<BackwardEulerScheme>(dt, GRID_SPACING);
                break;
            case TimeSchemeType::CRANK_NICOLSON:
                std::cout << "Using Crank-Nicolson scheme" << std::endl;
                solver = std::make_shared<implicitCN>(dt, GRID_SPACING);
                break;
            default:
                std::cout << "Using Backward Euler scheme as default" << std::endl;
                solver = std::make_shared<BackwardEulerScheme>(dt, GRID_SPACING);
                break;
        }
        
    }
    
    ~LevelSetMethod() = default;
    
    CGAL::Bbox_3 calculateBoundingBox() const;
    bool smoothShape(double smoothingFactor, int iterations);
    bool extractSurfaceMeshCGAL(const std::string& filename, 
        bool smoothSurface, 
        bool refineMesh, 
        bool remeshSurface,
        int smoothingIterations, 
        double targetEdgeLength,
        bool smoothShape,             // New parameter
        double shapeSmoothing,          // New parameter
        int shapeSmoothingIterations);
    void loadMesh(const std::string& filename);
    bool evolve();
    void reinitialize();
    bool exportGridMaterialsToCSV(const std::string& filename);
    void setGridMaterial(const std::string& material, const double zmax, const double zmin) {
        // Input validation
        if (zmax < zmin) {
            throw std::invalid_argument("zmax must be greater than or equal to zmin");
        }
        if (material.empty()) {
            throw std::invalid_argument("material name cannot be empty");
        }
        if (gridMaterials.empty() || grid.empty()) {
            throw std::runtime_error("grid materials or grid points not initialized");
        }

        // Use OpenMP for parallel processing
        #pragma omp parallel for
        for (size_t i = 0; i < gridMaterials.size(); ++i) {
            const double z = grid[i].z();
            // Use inclusive range check and avoid string comparison in tight loop
            if (z <= zmax && z > zmin && phi[i] <= 0.0) {
                gridMaterials[i] = material;
            }
        }
    }
    void setMaterialProperties(const std::string& material, double etchRatio, double lateralRatio){
        materialProperties[material].etchRatio = etchRatio;
        materialProperties[material].lateralRatio = lateralRatio;
        materialProperties[material].name = material;
    }
    void clearMaterialProperties() {
        materialProperties.clear();
    }
    void setSTEPS(int steps){
        STEPS = steps;
    }
    void updateU(){
        Ux = Eigen::VectorXd::Zero(phi.size());
        Uy = Eigen::VectorXd::Zero(phi.size());
        Uz = Eigen::VectorXd::Zero(phi.size());
        
        #pragma omp parallel for schedule(static, 128)
        for (int idx = 0; idx < static_cast<int>(phi.size()); ++idx) {
            std::string material = getMaterialAtPoint(idx);
            Eigen::Vector3d modifiedU_components;
            
            const auto it_mat = materialProperties.find(material);
            if (it_mat != materialProperties.end()) {
                const auto& props = it_mat->second;
                const double lateral_etch = props.lateralRatio * props.etchRatio;
                modifiedU_components << lateral_etch, lateral_etch, props.etchRatio;
            } else {
                modifiedU_components.setZero();
            }
                
            Ux(idx) = -modifiedU_components.x();
            Uy(idx) = -modifiedU_components.y();
            Uz(idx) = -modifiedU_components.z();
        }
    }

private:
    const int GRID_SIZE;
    double GRID_SPACING;
    const double dt;
    int STEPS;
    const int REINIT_INTERVAL;
    double BOX_SIZE = -1.0;
    TimeSchemeType timeScheme;
    
    double gridOriginX = 0.0;
    double gridOriginY = 0.0;
    double gridOriginZ = 0.0;
   
    std::shared_ptr<TimeScheme> solver;

    Mesh mesh;
    std::unique_ptr<AABB_tree> tree;
    std::vector<Point_3> grid;
    Eigen::VectorXd phi;
    Eigen::VectorXd Ux, Uy, Uz;
    
    // Add material related members
    struct MaterialProperties {
        double etchRatio;
        double lateralRatio;
        std::string name;
    };
    
    std::unordered_map<std::string, MaterialProperties> materialProperties;
    std::vector<std::string> gridMaterials; // Store material for each grid point
    
    // Add new methods
    std::string getMaterialAtPoint(int idx) const;
    void generateGrid();
    Eigen::VectorXd initializeSignedDistanceField();
    bool isOnBoundary(int idx) const;
    int getIndex(int x, int y, int z) const;

};

#endif // LEVEL_SET_METHOD_HPP
