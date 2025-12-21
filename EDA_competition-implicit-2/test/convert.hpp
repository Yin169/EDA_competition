#ifndef CONVERT_HPP
#define CONVERT_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <regex>
#include <sstream>
#include <cmath>
#include "DFISEParser.hpp"

int Convert(const std::string& inputFile = "data/initial_struct.bnd") {
    DFISEParser parser(inputFile);
    parser.parse();
    
    // Print some basic information
    std::cout << "File version: " << parser.getInfoDouble()["version"] << std::endl;
    std::cout << "Type: " << parser.getInfoString()["type"] << std::endl;
    std::cout << "Dimension: " << parser.getInfoInt()["dimension"] << std::endl;
    std::cout << "Number of vertices: " << parser.getInfoInt()["nb_vertices"] << std::endl;
    std::cout << "Number of edges: " << parser.getInfoInt()["nb_edges"] << std::endl;
    std::cout << "Number of faces: " << parser.getInfoInt()["nb_faces"] << std::endl;
    std::cout << "Number of elements: " << parser.getInfoInt()["nb_elements"] << std::endl;
    std::cout << "Number of regions: " << parser.getInfoInt()["nb_regions"] << std::endl;
    
    // Print the first few vertices
    auto vertices = parser.getVertices();
    if (!vertices.empty()) {
        std::cout << vertices.size() <<"\nFirst 5 vertices:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), vertices.size()); ++i) {
            std::cout << "Vertex " << i+1 << ": [";
            for (size_t j = 0; j < vertices[i].size(); ++j) {
                std::cout << vertices[i][j];
                if (j < vertices[i].size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
    }
    
    // Print the first few edges
    auto edges = parser.getEdges();
    if (!edges.empty()) {
        std::cout << "\nFirst 5 edges:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), edges.size()); ++i) {
            std::cout << "Edge " << i+1 << ": [";
            for (size_t j = 0; j < edges[i].size(); ++j) {
                std::cout << edges[i][j];
                if (j < edges[i].size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
    }
    
    // Print the first few faces
    auto faces = parser.getFaces();
    if (!faces.empty()) {
        std::cout << "\nFirst 5 faces:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), faces.size()); ++i) {
            std::cout << "Face " << i+1 << ": [";
            for (size_t j = 0; j < faces[i].size(); ++j) {
                std::cout << faces[i][j];
                if (j < faces[i].size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
    }
    
    auto getElementToMaterialMap = parser.getElementToMaterial();
    if (!getElementToMaterialMap.empty()) {
        std::cout << "\nFirst 5 elements:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), getElementToMaterialMap.size()); ++i) {
            std::cout << " Element " << i+1 << ": ";
            std::cout << getElementToMaterialMap[i];   
            std::cout << std::endl;
        }
    }

    auto getFaceToMaterialMap = parser.getFaceToMaterial();
    if (!getFaceToMaterialMap.empty()) {
        std::cout << "\nFirst 5 faces:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), getFaceToMaterialMap.size()); ++i) {
            std::cout << " Face " << i+1 << ": ";   
            std::cout << getFaceToMaterialMap[i];
            std::cout << std::endl;
        }
    }


    std::string outputFile = "./initial_struct.obj";
    if (!parser.exportToObj(outputFile)) {
        std::cerr << "Failed to export to OBJ format" << std::endl;
        return 1;
    }
 
    parser.exportFaceMaterials("./initial_struct_test.csv");
    parser.exportVertexMaterials("./initial_struct_vertex_material.csv");
    return 0;
}

#endif