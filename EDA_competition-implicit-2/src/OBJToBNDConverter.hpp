#ifndef OBJ_TO_BND_CONVERTER_H
#define OBJ_TO_BND_CONVERTER_H
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>

struct Vertex {
    double x, y, z;
    Vertex(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}
};

struct Edge {
    int v1, v2;  // vertex indices
    //Edge(int v1, int v2) : v1(std::min(v1, v2)), v2(std::max(v1, v2)) {}
    Edge(int v1, int v2) : v1(v1), v2(v2) {}
    bool operator<(const Edge& other) const {
        return (v1 < other.v1) || (v1 == other.v1 && v2 < other.v2);
    }
};

struct Face {
    std::vector<int> edges;  // edge indices
    std::vector<int> vertices; // original vertex indices from OBJ
};

struct Element {
    int shapeCode;
    std::vector<int> faces;  // face indices
    std::string material;
};

struct Region {
    std::string name;
    std::string material;
    std::vector<int> elements;  // element indices
};

class ObjToBndConverter {
private:
    std::vector<Vertex> vertices;
    std::vector<Edge> edges;
    std::vector<Face> faces;
    std::vector<Element> elements;
    std::vector<Region> regions;
    std::map<Edge, int> edgeMap;  // edge to index mapping
    std::vector<char> locations;  // location codes for faces
    
    // Current material/region being processed
    std::string currentMaterial = "Polymer";
    
public:
    bool loadObjFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open OBJ file: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        std::vector<std::vector<int>> objFaces;  // temporary storage for OBJ faces
        
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string type;
            iss >> type;
            
            if (type == "v") {
                // Vertex
                double x, y, z;
                iss >> x >> y >> z;
                vertices.push_back(Vertex(x, y, z));
            }
            else if (type == "f") {
                // Face
                std::vector<int> faceVertices;
                std::string vertex;
                while (iss >> vertex) {
                    // Handle vertex/texture/normal format (v/vt/vn)
                    size_t slashPos = vertex.find('/');
                    int vertexIndex;
                    if (slashPos != std::string::npos) {
                        vertexIndex = std::stoi(vertex.substr(0, slashPos));
                    } else {
                        vertexIndex = std::stoi(vertex);
                    }
                    // OBJ indices are 1-based, convert to 0-based
                    faceVertices.push_back(vertexIndex - 1);
                }
                if (faceVertices.size() >= 3) {
                    objFaces.push_back(faceVertices);
                }
            }
            else if (type == "usemtl") {
                // Material
                iss >> currentMaterial;
            }
            else if (type == "g" || type == "o") {
                // Group or object name - could be used for region names
                std::string name;
                iss >> name;
                if (!name.empty()) {
                    currentMaterial = name;
                }
            }
        }
        
        file.close();
        
        // Process faces and create edges
        processFaces(objFaces);
        
        // Create elements and regions
        createElementsAndRegions();
        
        // Determine location codes
        determineLocationCodes();
        
        std::cout << "Loaded OBJ file successfully:" << std::endl;
        std::cout << "  Vertices: " << vertices.size() << std::endl;
        std::cout << "  Edges: " << edges.size() << std::endl;
        std::cout << "  Faces: " << faces.size() << std::endl;
        std::cout << "  Elements: " << elements.size() << std::endl;
        std::cout << "  Regions: " << regions.size() << std::endl;
        
        return true;
    }
    
    void processFaces(const std::vector<std::vector<int>>& objFaces) {
        for (const auto& objFace : objFaces) {
            Face face;
            face.vertices = objFace;
            
            std::vector<int> faceEdges;
            int size_obj = objFace.size();
            if (size_obj > 3) {
                size_obj--;
            }
            for (size_t i = 0; i < size_obj; ++i) {
                int v1 = objFace[i];
                int v2 = objFace[(i + 1) % objFace.size()];
                
                Edge edge(v1, v2);
                
                int edgeIndex;
                std::map<Edge, int>::iterator it;
                if ( (it = edgeMap.find(edge) )!= edgeMap.end()) {
                    // edge exists
                    edgeIndex = it->second;
                } else if ((it = edgeMap.find(Edge(edge.v2, edge.v1))) != edgeMap.end()) {
                    // edge exists but with reversed vertices
                    edgeIndex = - it -> second - 1; // negative index for reversed edges
                } else {
                    //new edge
                    edgeIndex = edges.size();
                    edges.push_back(edge);
                    edgeMap[edge] = edgeIndex;
                }
                
                face.edges.push_back(edgeIndex);
            }
            
            faces.push_back(face);
        }
    }
    
    void createElementsAndRegions() {
        // For simplicity, create one region with all faces as polygon elements
        Region region;
        region.name = "Polymer_1";
        region.material = currentMaterial;
        Element element;
        element.shapeCode = 10;
        element.material = currentMaterial; 
        for (size_t i = 0; i < faces.size(); ++i) {
            element.faces.push_back(i);  
        }
        elements.push_back(element);
        int elementIndex = elements.size();
        if (elementIndex > 0){
            region.elements.push_back(elementIndex-1); 
        }
        regions.push_back(region);
    }
    
   void determineLocationCodes() {
        locations.resize(faces.size(), 'e');  // external interface
    }
    
    bool writeBndFile(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot create BND file: " << filename << std::endl;
            return false;
        }
        
        // Write DF-ISE header
        file << "DF-ISE text" << std::endl;
        file << std::endl;
        
        // Write Info block
        file << "Info {" << std::endl;
        file << "    version = 1.0" << std::endl;
        file << "    type = boundary" << std::endl;
        file << "    dimension = 3" << std::endl;
        file << "    nb_vertices = " << vertices.size() << std::endl;
        file << "    nb_edges = " << edges.size() << std::endl;
        file << "    nb_faces = " << faces.size() << std::endl;
        file << "    nb_elements = " << elements.size() << std::endl;
        file << "    nb_regions = " << regions.size() << std::endl;
        
        // Write regions list
        file << "    regions = [";
        for (size_t i = 0; i < regions.size(); ++i) {
            file << " \"" << regions[i].name << "\"";
            if (i < regions.size() - 1) file << ",";
        }
        file << " ]" << std::endl;
        
        // Write materials list
        file << "    materials = [";
        for (size_t i = 0; i < regions.size(); ++i) {
            file << " " << regions[i].material;
            if (i < regions.size() - 1) file << ",";
        }
        file << " ]" << std::endl;
        
        file << "}" << std::endl;
        file << std::endl;
        
        // Write Data block
        file << "Data {" << std::endl;
        
        // Write coordinate system (identity transformation)
        file << "    CoordSystem {" << std::endl;
        file << "        translate = [ 0 0 0 ]" << std::endl;
        file << "        transform = [ 1 0 0" << std::endl;
        file << "                      0 1 0" << std::endl;
        file << "                      0 0 1 ]" << std::endl;
        file << "    }" << std::endl;
        file << std::endl;
        
        // Write Vertices
        file << "    Vertices (" << vertices.size() << ") {" << std::endl;
        for (const auto& vertex : vertices) {
            file << "        " << vertex.x << " " << vertex.y << " " << vertex.z << std::endl;
        }
        file << "    }" << std::endl;
        file << std::endl;
        
        // Write Edges
        file << "    Edges (" << edges.size() << ") {" << std::endl;
        for (const auto& edge : edges) {
            file << "        " << edge.v1 << " " << edge.v2 << std::endl;
        }
        file << "    }" << std::endl;
        file << std::endl;
        
        // Write Faces
        file << "    Faces (" << faces.size() << ") {" << std::endl;
        for (const auto& face : faces) {
            file << "        " << face.edges.size();
            for (int edgeIdx : face.edges) {
                file << " " << edgeIdx;
            }
            file << std::endl;
        }
        file << "    }" << std::endl;
        file << std::endl;
        
        // Write Locations
        file << "    Locations (" << faces.size() << ") {";
        for (size_t i = 0; i < locations.size(); ++i) {
            if (i % 10 == 0) file << std::endl << "            ";
            file << locations[i] ;
        }
        file << std::endl << "    }" << std::endl;
        file << std::endl;
        
        // Write Elements
        file << "    Elements (" << elements.size() << ") {" << std::endl;
        for (const auto& element : elements) {
            file << "        " << element.shapeCode;
            file << " " << element.faces.size();
            for (size_t i = 0; i < element.faces.size(); ++i) {
                if (i>0 && i % 10 == 0) file << std::endl << "            ";
                file << " " << element.faces[i];
            }
            file << std::endl;
        }
        file << "    }" << std::endl;
        file << std::endl;
        
        // Write Regions
        for (const auto& region : regions) {
            file << "    Region (\"" << region.name << "\") {" << std::endl;
            file << "        material = " << region.material << std::endl;
            file << "        Elements (" << region.elements.size() << ") {";
            for (size_t i = 0; i < region.elements.size(); ++i) {
                file << " " << region.elements[i];
            }
            file << " }" << std::endl;
            file << "    }" << std::endl;
        }
        
        file << "}" << std::endl;
        file.close();
        
        std::cout << "Successfully wrote BND file: " << filename << std::endl;
        return true;
    }
};


int ConvertOBJToDFISE(const std::string& inputFile, const std::string& outputFile) {
	ObjToBndConverter converter;
	if (!converter.loadObjFile(inputFile)) {
        std::cerr << "Failed to load OBJ file." << std::endl;
        return 1;
    }
    
    if (!converter.writeBndFile(outputFile)) {
        std::cerr << "Failed to write BND file." << std::endl;
        return 1;
    }
    return 0; 
}

#endif