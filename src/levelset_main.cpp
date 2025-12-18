#include <omp.h>
#include <iostream>
#include <cmath>
#include <filesystem>
#include <Eigen/Core>
#include <chrono>

namespace fs = std::filesystem;

#include "convert.hpp"
#include "alphawrap.hpp"
#include "OBJToBNDConverter.hpp"

#include "LevelSetMethod.hpp"
#include "LevelSetMethod.cpp"

inline std::chrono::steady_clock::time_point tic() { return std::chrono::steady_clock::now(); }

inline double toc(std::chrono::steady_clock::time_point tic_) {
    return std::chrono::duration_cast<std::chrono::duration<double>>(tic() - tic_).count();
}

int main(int argc, char* argv[]) {
   
    const std::string inFile = argv[1];
    const std::string outputFile = argv[2];
    const std::string numThread = argv[3];    
    const std::string timeScheme = argv[4];
    TimeSchemeType TimeScheme; 

    if (timeScheme == "RUNGE_KUTTA_3") {
        TimeScheme = TimeSchemeType::RUNGE_KUTTA_3;
    }
    if (timeScheme == "BACKWARD_EULER") {
        TimeScheme = TimeSchemeType::BACKWARD_EULER;
    }
    if (timeScheme == "CRANK_NICOLSON") {
        TimeScheme = TimeSchemeType::CRANK_NICOLSON;
    }

    
    
    std::string folderPath = "./data";

    if (!fs::exists(folderPath)) {
        if (fs::create_directory(folderPath)) {
            std::cout << "Directory created successfully.\n";
        } else {
            std::cerr << "Failed to create directory.\n";
        }
    } else {
        std::cout << "Directory already exists.\n";
    }

    std::cout << "Input file: " << inFile << std::endl;
    std::cout << "Output file: " << outputFile << std::endl;

    Convert(inFile);
    Wrapper("data/initial_struct.obj", 600, 600);

    std::string inputFile = "initial_struct_600_600.obj";
    
    LevelSetMethod levelSet(
        inputFile,
        100,    // gridSize
        1.0,   // timeStep
        60,    // maxSteps
        3,      // reinitInterval
        TimeScheme,
        std::stoi(numThread)     // numThreads (auto)
    );

    levelSet.setGridMaterial("SiO2_PECVD", 500, 400); 
    levelSet.setGridMaterial("Si_Amorph", 400, 364); 
    levelSet.setGridMaterial("Polymer", 364, 304);
    levelSet.setGridMaterial("Si3N4_LPCVD", 304, 204); 
    levelSet.setGridMaterial("SiO2_Thermal", 204, 203); 
    levelSet.setGridMaterial("Si_Xtal", 203, -200); 
    // levelSet.exportGridMaterialsToCSV("checking.csv");
    levelSet.updateU();


    // case 1
    levelSet.clearMaterialProperties();
    levelSet.setMaterialProperties("Polymer", 0.1, 0.01);
    levelSet.setMaterialProperties( "SiO2_PECVD", 0.6, 0.01);
    levelSet.setMaterialProperties("Si_Amorph", 1.0, 0.01);
    std::string surfaceFile = timeScheme+ "_"+ numThread +"_Silicon_etch.off";
    std::string outputBNDfile = "Silicon_etch.bnd";
    surfaceFile = outputFile + surfaceFile;
    outputBNDfile = outputFile + outputBNDfile;

    auto tic_ = tic();
    std::cout << "Running level set evolution..." << std::endl;
    if (!levelSet.evolve()) {
        std::cerr << "Evolution failed. Exiting." << std::endl; return 1;
    }
    auto toc_ = toc(tic_);
    std::cout << "TimeScheme : " << timeScheme << " numThread : " << numThread << " SpendingTime : " << toc_ << std::endl;

    std::cout << "Saving surface mesh to " << surfaceFile << "..." << std::endl;
    if (!levelSet.extractSurfaceMeshCGAL(surfaceFile)) {
        std::cerr << "Failed to save surface mesh. Exiting." << std::endl;
        return 1;
    }

    int saveBND = ConvertOBJToDFISE(surfaceFile, outputBNDfile); 
    if (saveBND == 0) {
        std::cout << "Conversion completed successfully." << std::endl;
    } else {
        std::cerr << "Conversion failed." << std::endl;
    }

    // // case 2
    // levelSet.clearMaterialProperties();
    // levelSet.setMaterialProperties("Si3N4_LPCVD", 0.3, 0.01);
    // levelSet.setMaterialProperties("Polymer", 1, 0.01);
    // levelSet.setMaterialProperties("SiO2_PECVD", 0.6, 0.01);
    // levelSet.setSTEPS(120);  // maxSteps
    // surfaceFile = "Polymer_etch.obj";
    // outputBNDfile = "Polymer_etch.bnd";
    // surfaceFile = outputFile + surfaceFile;
    // outputBNDfile = outputFile + outputBNDfile;

    // std::cout << "Running level set evolution..." << std::endl;
    // if (!levelSet.evolve()) {
    //     std::cerr << "Evolution failed. Exiting." << std::endl; return 1;
    // }
    // std::cout << "Saving surface mesh to " << surfaceFile << "..." << std::endl;
    // if (!levelSet.extractSurfaceMeshCGAL(surfaceFile)) {
    //     std::cerr << "Failed to save surface mesh. Exiting." << std::endl;
    //     return 1;
    // }

    // saveBND = ConvertOBJToDFISE(surfaceFile, outputBNDfile); 
    // if (saveBND == 0) {
    //     std::cout << "Conversion completed successfully." << std::endl;
    // } else {
    //     std::cerr << "Conversion failed." << std::endl;
    // }

    // // case 3
    // levelSet.clearMaterialProperties();
    // levelSet.setMaterialProperties("Si3N4_LPCVD", 1, 0.01);
    // levelSet.setMaterialProperties("Polymer", 0.1, 0.01);  
    // levelSet.setMaterialProperties("SiO2_PECVD", 0.4, 0.01); 
    // levelSet.setMaterialProperties("SiO2_Thermal", 0.7, 0.01);  
    // levelSet.setMaterialProperties("Si_Amorph", 0.35, 0.01); 
    // levelSet.setMaterialProperties("Si_Xtal", 0.35, 0.01);
    // levelSet.setSTEPS(160);  // maxSteps
    // surfaceFile = "Nitride_etch.obj";
    // outputBNDfile = "Nitride_etch.bnd"; 
    // surfaceFile = outputFile + surfaceFile;
    // outputBNDfile = outputFile + outputBNDfile;

    // std::cout << "Running level set evolution..." << std::endl;
    // if (!levelSet.evolve()) {
    //     std::cerr << "Evolution failed. Exiting." << std::endl; return 1;
    // }
    // std::cout << "Saving surface mesh to " << surfaceFile << "..." << std::endl;
    // if (!levelSet.extractSurfaceMeshCGAL(surfaceFile)) {
    //     std::cerr << "Failed to save surface mesh. Exiting." << std::endl;
    //     return 1;
    // }

    // saveBND = ConvertOBJToDFISE(surfaceFile, outputBNDfile); 
    // if (saveBND == 0) {
    //     std::cout << "Conversion completed successfully." << std::endl;
    // } else {
    //     std::cerr << "Conversion failed." << std::endl;
    // }

    std::cout << "Level set method completed successfully." << std::endl;
    
    return 0;
}
