#ifndef DFISE_PARSER_H
#define DFISE_PARSER_H
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <regex>
#include <sstream>
#include <cmath>
#include <cstring>
#include <set>

class DFISEParser {
public:
    // Constructor and public methods
    DFISEParser(const std::string& path) : file_path(path) {}
    
    // Method to parse the DF-ISE file
    bool parse() {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << file_path << std::endl;
            return false;
        }
        
        // Process the file in chunks to reduce memory usage
        bool found_info = false;
        bool found_data = false;
        std::string section_name;
        std::string section_content;
        int brace_count = 0;
        bool in_section = false;
        
        std::string line;
        while (std::getline(file, line)) {
            // Look for section starts
            if (!in_section) {
                if (line.find("Info") != std::string::npos && line.find("{") != std::string::npos) {
                    section_name = "Info";
                    in_section = true;
                    brace_count = 1; // We've found the opening brace
                    section_content.clear();
                    
                    // Extract content after the opening brace
                    size_t pos = line.find("{");
                    if (pos != std::string::npos && pos + 1 < line.length()) {
                        section_content += line.substr(pos + 1) + "\n";
                        
                        // Check if there are more braces in this line
                        for (size_t i = pos + 1; i < line.length(); ++i) {
                            if (line[i] == '{') brace_count++;
                            else if (line[i] == '}') brace_count--;
                        }
                    }
                }
                else if (line.find("Data") != std::string::npos && line.find("{") != std::string::npos) {
                    section_name = "Data";
                    in_section = true;
                    brace_count = 1; // We've found the opening brace
                    section_content.clear();
                    
                    // Extract content after the opening brace
                    size_t pos = line.find("{");
                    if (pos != std::string::npos && pos + 1 < line.length()) {
                        section_content += line.substr(pos + 1) + "\n";
                        
                        // Check if there are more braces in this line
                        for (size_t i = pos + 1; i < line.length(); ++i) {
                            if (line[i] == '{') brace_count++;
                            else if (line[i] == '}') brace_count--;
                        }
                    }
                }
            }
            else {
                // We're inside a section, keep track of braces
                for (char c : line) {
                    if (c == '{') brace_count++;
                    else if (c == '}') brace_count--;
                }
                
                section_content += line + "\n";
                
                // If we've found the matching closing brace, process the section
                if (brace_count == 0) {
                    // Remove the last closing brace
                    size_t pos = section_content.rfind('}');
                    if (pos != std::string::npos) {
                        section_content = section_content.substr(0, pos);
                    }
                    
                    if (section_name == "Info") {
                        parseInfo(section_content);
                        found_info = true;
                    }
                    else if (section_name == "Data") {
                        parseData(section_content);
                        found_data = true;
                    }
                    
                    in_section = false;
                    section_content.clear();
                }
            }
        }
        
        file.close();
        return found_info && found_data;
    }
    
    // Method to output material information for all faces
    void printFaceMaterials() const {
        std::cout << "\n===== Face Materials =====" << std::endl;
        std::cout << "Total faces: " << faces.size() << std::endl;
        
        for (size_t i = 0; i < faces.size(); ++i) {
            std::string material = getMaterialForFace(i);
            std::cout << "Face " << i << ": Material = " << material << std::endl;
        }
        std::cout << "========================" << std::endl;
    }
    
    // Method to export face materials to a file
    bool exportFaceMaterials(const std::string& output_path) const {
        std::ofstream file(output_path);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << output_path << " for writing" << std::endl;
            return false;
        }
        
        file << "# Face Materials Export" << std::endl;
        file << "# Format: FaceIndex,Material" << std::endl;
        file << "# Total faces: " << faces.size() << std::endl;
        
        for (size_t i = 0; i < faces.size(); ++i) {
            std::string material = getMaterialForFace(i);
            file << i << "," << material << std::endl;
        }
        
        file.close();
        std::cout << "Face materials exported to " << output_path << std::endl;
        return true;
    }
    
    // Method to export detailed face materials to a CSV file
    bool exportDetailedFaceMaterials(const std::string& output_path) const {
        std::ofstream file(output_path);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << output_path << " for writing" << std::endl;
            return false;
        }
        
        // Write CSV header
        file << "FaceIndex,Material,VertexCount,VertexIndices" << std::endl;
        
        // Write each face with its material and vertices
        for (size_t i = 0; i < faces.size(); ++i) {
            std::string material = getMaterialForFace(i);
            const auto& face = faces[i];
            
            file << i << "," << material << "," << face.size();
            
            // Add vertex indices
            for (const auto& vertex_idx : face) {
                file << "," << vertex_idx;
            }
            file << std::endl;
        }
        
        file.close();
        std::cout << "Detailed face materials exported to " << output_path << std::endl;
        return true;
    }
    
    // Get all face materials as a map
    std::map<int, std::string> getAllFaceMaterials() const {
        std::map<int, std::string> result;
        for (size_t i = 0; i < faces.size(); ++i) {
            result[i] = getMaterialForFace(i);
        }
        return result;
    }
    
    // Get the faces vector
    const std::vector<std::vector<int>>& getFaces() const {
        return faces;
    }
    
    // Get material for a specific face
    std::string getMaterialForFace(int face_idx) const {
        auto it = face_to_material.find(face_idx);
        if (it != face_to_material.end()) {
            return it->second;
        }
        return "unknown";
    }
    
    // Get material for a specific vertex
    std::string getMaterialForVertex(int vertex_idx) const {
        auto it = vertex_to_material.find(vertex_idx);
        if (it != vertex_to_material.end()) {
            return it->second;
        }
        return "unknown";
    }
    
     // Map materials to vertices and return the mapping
    void mapMaterialsToVertices() {
        // Make sure we have the face-to-material mapping first
        if (face_to_material.empty()) {
            mapMaterialsToFaces();
        }
        
        // Clear any existing vertex-to-material mapping
        vertex_to_material.clear();
        
        // For each face, assign its material to all vertices in the face
        for (size_t face_idx = 0; face_idx < faces.size(); ++face_idx) {
            const auto& face = faces[face_idx];
            std::string material = getMaterialForFace(face_idx);
            
            // Skip faces with unknown material
            if (material == "unknown") {
                continue;
            }
            
            // Assign this material to all vertices in this face
            for (auto edge_idx : face) {
                if (edge_idx < 0){
                    edge_idx = -edge_idx-1;
                    const auto& edge = edges[edge_idx];
                    if (edge.size() == 2) {
                        const auto& vertex_idx0 = edge[0];
                        vertex_to_material[vertex_idx0] = material;
                        const auto& vertex_idx1 = edge[1];
                        vertex_to_material[vertex_idx1] = material;
                    }
                }
            }
        }
        
        // Debug output
        std::cout << "Mapped materials to " << vertex_to_material.size() << " vertices" << std::endl;
    }
    
    // Export vertex materials to a file
    bool exportVertexMaterials(const std::string& output_path) const {
        std::ofstream file(output_path);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << output_path << " for writing" << std::endl;
            return false;
        }
        
        file << "# Vertex Materials Export" << std::endl;
        file << "# Format: VertexIndex,X,Y,Z,Material" << std::endl;
        file << "# Total vertices with material: " << vertex_to_material.size() << std::endl;
        
        for (const auto& pair : vertex_to_material) {
            int vertex_idx = pair.first;
            // Check if vertex index is valid
            if (vertex_idx >= 0 && vertex_idx < vertices.size()) {
                const auto& vertex = vertices[vertex_idx];
                // Write vertex index, coordinates, and material
                file << vertex_idx << ","
                     << vertex[0] << ","
                     << vertex[1] << ","
                     << vertex[2] << ","
                     << pair.second << std::endl;
            }
        }
        
        file.close();
        std::cout << "Vertex materials and coordinates exported to " << output_path << std::endl;
        return true;
    }
    
private:
    std::string file_path;
    std::map<std::string, double> info_double;
    std::map<std::string, int> info_int;
    std::map<std::string, std::string> info_string;
    std::vector<std::vector<double>> vertices;
    std::vector<std::vector<int>> edges;
    std::vector<std::vector<int>> faces;
    std::vector<std::vector<int>> elements;
    std::vector<std::string> regions;
    std::vector<std::string> materials;
    std::map<std::string, std::vector<double>> coord_system_vector;
    std::map<std::string, std::vector<std::vector<double>>> coord_system_matrix;
    
    // New data structures for region-element-material mapping
    struct RegionInfo {
        std::string name;
        std::string material;
        std::vector<int> elements;
    };
    std::vector<RegionInfo> region_info;
    std::map<int, std::string> element_to_material;
    std::map<int, std::string> face_to_material;
    std::map<int, std::string> vertex_to_material; // New mapping for vertices to materials
    
    // Helper function to extract content between braces - optimized version
    // This is kept for backward compatibility but no longer used in the optimized parse() method
    std::string extractBetweenBraces(const std::string& content, const std::string& section) {
        // Find the starting position of the section using string search instead of regex
        size_t section_pos = content.find(section);
        if (section_pos == std::string::npos) {
            return "";
        }
        
        // Find the opening brace
        size_t open_brace = content.find('{', section_pos);
        if (open_brace == std::string::npos) {
            return "";
        }
        
        size_t start_pos = open_brace + 1;
        size_t pos = start_pos;
        int brace_count = 1; // We've already found the opening brace
        
        // Find the matching closing brace by counting opening and closing braces
        while (pos < content.length() && brace_count > 0) {
            if (content[pos] == '{') {
                brace_count++;
            } else if (content[pos] == '}') {
                brace_count--;
            }
            pos++;
        }
        
        // If we found the matching closing brace, return the content between braces
        if (brace_count == 0) {
            return content.substr(start_pos, pos - start_pos - 1);
        }
        
        return "";
    }

    // Helper function to extract content between parentheses - optimized version
    // This is kept for backward compatibility but no longer used in the optimized parseData method
    std::string extractBetweenParentheses(const std::string& content, const std::string& section) {
        // Find the section using string search instead of regex
        size_t section_pos = content.find(section);
        if (section_pos == std::string::npos) {
            return "";
        }
        
        // Find the opening parenthesis
        size_t open_paren = content.find('(', section_pos);
        if (open_paren == std::string::npos) {
            return "";
        }
        
        // Find the closing parenthesis
        size_t close_paren = content.find(')', open_paren);
        if (close_paren == std::string::npos) {
            return "";
        }
        
        // Find the opening brace
        size_t open_brace = content.find('{', close_paren);
        if (open_brace == std::string::npos) {
            return "";
        }
        
        // Find the closing brace
        size_t close_brace = content.find('}', open_brace);
        if (close_brace == std::string::npos) {
            return "";
        }
        
        // Return the content between braces
        return content.substr(open_brace + 1, close_brace - open_brace - 1);
    }

    // Helper function to extract vector values
    std::vector<double> extractVector(const std::string& text) {
        std::vector<double> result;
        std::regex pattern("\\[(.*?)\\]");
        std::smatch matches;
        if (std::regex_search(text, matches, pattern)) {
            std::string values = matches[1].str();
            std::istringstream iss(values);
            double value;
            while (iss >> value) {
                result.push_back(value);
            }
        }
        return result;
    }

    // Helper function to extract matrix values
    std::vector<std::vector<double>> extractMatrix(const std::string& text) {
        std::vector<std::vector<double>> result;
        std::regex pattern("\\[(.*?)\\]");
        std::smatch matches;
        if (std::regex_search(text, matches, pattern)) {
            std::string values = matches[1].str();
            std::istringstream iss(values);
            std::string line;
            while (std::getline(iss, line)) {
                std::vector<double> row;
                std::istringstream line_iss(line);
                double value;
                while (line_iss >> value) {
                    row.push_back(value);
                }
                if (!row.empty()) {
                    result.push_back(row);
                }
            }
        }
        return result;
    }

    // Helper function to extract string list - optimized version
    std::vector<std::string> extractStringList(const std::string& text) {
        std::vector<std::string> result;
        
        // Find the content between brackets using string operations
        size_t start_bracket = text.find('[');
        size_t end_bracket = text.find(']', start_bracket);
        
        if (start_bracket == std::string::npos || end_bracket == std::string::npos || end_bracket <= start_bracket) {
            return result; // Return empty result if no valid brackets found
        }
        
        // Extract the content between brackets
        std::string values = text.substr(start_bracket + 1, end_bracket - start_bracket - 1);
        
        // Process the string to extract quoted values
        size_t pos = 0;
        while (pos < values.length()) {
            // Find opening quote
            size_t quote_start = values.find('"', pos);
            if (quote_start == std::string::npos) break;
            
            // Find closing quote
            size_t quote_end = values.find('"', quote_start + 1);
            if (quote_end == std::string::npos) break;
            
            // Extract the string between quotes
            std::string value = values.substr(quote_start + 1, quote_end - quote_start - 1);
            result.push_back(value);
            
            // Move to the next position after the closing quote
            pos = quote_end + 1;
        }
        
        return result;
    }

    void parseInfo(const std::string& info_text) {
        // Process the info text line by line to extract key-value pairs
        std::istringstream info_stream(info_text);
        std::string line;
        
        // Temporary storage for regions and materials text
        std::string regions_text;
        std::string materials_text;
        bool in_regions = false;
        bool in_materials = false;
        int bracket_count = 0;
        
        while (std::getline(info_stream, line)) {
            // Check for regions and materials lists which may span multiple lines
            if (line.find("regions") != std::string::npos && line.find("=") != std::string::npos) {
                in_regions = true;
                regions_text = line.substr(line.find("=") + 1);
                
                // Count brackets to track multi-line lists
                for (char c : regions_text) {
                    if (c == '[') bracket_count++;
                    else if (c == ']') bracket_count--;
                }
                
                // If the list ends on this line, process it
                if (bracket_count == 0) {
                    in_regions = false;
                    size_t start = regions_text.find('[');
                    size_t end = regions_text.find(']');
                    if (start != std::string::npos && end != std::string::npos && end > start) {
                        regions = extractStringList(regions_text.substr(start, end - start + 1));
                    }
                }
                continue;
            }
            else if (line.find("materials") != std::string::npos && line.find("=") != std::string::npos) {
                in_materials = true;
                materials_text = line.substr(line.find("=") + 1);
                
                // Count brackets to track multi-line lists
                for (char c : materials_text) {
                    if (c == '[') bracket_count++;
                    else if (c == ']') bracket_count--;
                }
                
                // If the list ends on this line, process it
                if (bracket_count == 0) {
                    in_materials = false;
                    size_t start = materials_text.find('[');
                    size_t end = materials_text.find(']');
                    if (start != std::string::npos && end != std::string::npos && end > start) {
                        materials = extractStringList(materials_text.substr(start, end - start + 1));
                    }
                }
                continue;
            }
            
            // Continue collecting multi-line lists
            if (in_regions) {
                regions_text += line;
                for (char c : line) {
                    if (c == '[') bracket_count++;
                    else if (c == ']') bracket_count--;
                }
                
                if (bracket_count == 0) {
                    in_regions = false;
                    size_t start = regions_text.find('[');
                    size_t end = regions_text.find(']');
                    if (start != std::string::npos && end != std::string::npos && end > start) {
                        regions = extractStringList(regions_text.substr(start, end - start + 1));
                    }
                }
                continue;
            }
            else if (in_materials) {
                materials_text += line;
                for (char c : line) {
                    if (c == '[') bracket_count++;
                    else if (c == ']') bracket_count--;
                }
                
                if (bracket_count == 0) {
                    in_materials = false;
                    size_t start = materials_text.find('[');
                    size_t end = materials_text.find(']');
                    if (start != std::string::npos && end != std::string::npos && end > start) {
                        materials = extractStringList(materials_text.substr(start, end - start + 1));
                    }
                }
                continue;
            }
            
            // Process simple key-value pairs
            // Extract key and value using string operations instead of regex
            size_t equals_pos = line.find('=');
            if (equals_pos != std::string::npos) {
                std::string key = line.substr(0, equals_pos);
                std::string value = line.substr(equals_pos + 1);
                
                // Trim whitespace
                key.erase(0, key.find_first_not_of(" \t\n\r\f\v"));
                key.erase(key.find_last_not_of(" \t\n\r\f\v") + 1);
                value.erase(0, value.find_first_not_of(" \t\n\r\f\v"));
                value.erase(value.find_last_not_of(" \t\n\r\f\v") + 1);
                
                // Process different types of values
                if (key == "version") {
                    try {
                        info_double[key] = std::stod(value);
                    } catch (...) {
                        // Handle conversion error
                    }
                }
                else if (key == "type") {
                    info_string[key] = value;
                }
                else if (key == "dimension" || key == "nb_vertices" || key == "nb_edges" || 
                         key == "nb_faces" || key == "nb_elements" || key == "nb_regions") {
                    try {
                        info_int[key] = std::stoi(value);
                    } catch (...) {
                        // Handle conversion error
                    }
                }
            }
        }
    }

    void parseData(const std::string& data_text) {
        // Process the data section line by line to find subsections
        std::istringstream data_stream(data_text);
        std::string line;
        
        bool in_coord_system = false;
        bool in_vertices = false;
        bool in_edges = false;
        bool in_faces = false;
        bool in_elements = false;
        
        std::string current_section;
        std::stringstream section_content;
        int brace_count = 0;
        int paren_count = 0; // Variable for tracking parenthesis count
        
        while (std::getline(data_stream, line)) {
            // Check for section starts
            if (line.find("CoordSystem") != std::string::npos && line.find("{") != std::string::npos) {
                in_coord_system = true;
                current_section = "CoordSystem";
                section_content.str("");
                brace_count = 1; // We've found the opening brace
                
                // Extract content after the opening brace
                size_t pos = line.find("{");
                if (pos != std::string::npos && pos + 1 < line.length()) {
                    section_content << line.substr(pos + 1) << "\n";
                }
                continue;
            }
            else if (line.find("Vertices") != std::string::npos && line.find("(") != std::string::npos) {
                in_vertices = true;
                current_section = "Vertices";
                section_content.str("");
                paren_count = 1; // We've found the opening parenthesis
                
                // Extract content after the opening parenthesis and before closing brace
                size_t open_paren = line.find("(");
                size_t open_brace = line.find("{", open_paren);
                if (open_brace != std::string::npos && open_brace + 1 < line.length()) {
                    section_content << line.substr(open_brace + 1) << "\n";
                }
                continue;
            }
            else if (line.find("Edges") != std::string::npos && line.find("(") != std::string::npos) {
                in_edges = true;
                current_section = "Edges";
                section_content.str("");
                paren_count = 1; // We've found the opening parenthesis
                
                // Extract content after the opening parenthesis and before closing brace
                size_t open_paren = line.find("(");
                size_t open_brace = line.find("{", open_paren);
                if (open_brace != std::string::npos && open_brace + 1 < line.length()) {
                    section_content << line.substr(open_brace + 1) << "\n";
                }
                continue;
            }
            else if (line.find("Faces") != std::string::npos && line.find("(") != std::string::npos) {
                in_faces = true;
                current_section = "Faces";
                section_content.str("");
                paren_count = 1; // We've found the opening parenthesis
                
                // Extract content after the opening parenthesis and before closing brace
                size_t open_paren = line.find("(");
                size_t open_brace = line.find("{", open_paren);
                if (open_brace != std::string::npos && open_brace + 1 < line.length()) {
                    section_content << line.substr(open_brace + 1) << "\n";
                }
                continue;
            }
            else if (line.find("Elements") != std::string::npos && line.find("(") != std::string::npos) {
                in_elements = true;
                current_section = "Elements";
                section_content.str("");
                paren_count = 1; // We've found the opening parenthesis
                
                // Extract content after the opening parenthesis and before closing brace
                size_t open_paren = line.find("(");
                size_t open_brace = line.find("{", open_paren);
                if (open_brace != std::string::npos && open_brace + 1 < line.length()) {
                    section_content << line.substr(open_brace + 1) << "\n";
                }
                continue;
            }
            
            // We're inside a section, check for section end
            if (in_coord_system) {
                // Update brace count
                for (char c : line) {
                    if (c == '{') brace_count++;
                    else if (c == '}') brace_count--;
                }
                
                // Add line to section content
                section_content << line << "\n";
                
                // Check if section has ended
                if (brace_count == 0) {
                    parseCoordSystem(section_content.str());
                    in_coord_system = false;
                }
            }
            else if (in_vertices || in_edges || in_faces || in_elements) {
                // Check for closing brace which ends the section
                if (line.find("}") != std::string::npos) {
                    // Process the section content without the closing brace
                    size_t close_pos = line.find("}");
                    if (close_pos > 0) {
                        section_content << line.substr(0, close_pos) << "\n";
                    }
                    
                    // Parse the appropriate section
                    if (in_vertices) {
                        parseVertices(section_content.str());
                        in_vertices = false;
                    }
                    else if (in_edges) {
                        parseEdges(section_content.str());
                        in_edges = false;
                    }
                    else if (in_faces) {
                        parseFaces(section_content.str());
                        in_faces = false;
                    }
                    else if (in_elements) {
                        parseElements(section_content.str());
                        in_elements = false;
                    }
                }
                else {
                    // Add line to section content
                    section_content << line << "\n";
                }
            }
        }
        
        // Parse Regions - this is handled separately as regions can be more complex
        parseRegions(data_text);
        
        // Map materials to faces
        mapMaterialsToFaces();
        mapMaterialsToVertices();
    }
     // New method to parse regions
     void parseRegions(const std::string& data_text) {
        // This pattern matches the entire region block including nested braces
        std::regex region_pattern("Region\\s*\\(\\s*\"([^\"]*)\"\\s*\\)\\s*\\{([^{}]*)\\}");
        std::smatch region_matches;
        std::string::const_iterator search_start(data_text.cbegin());
        
        while (std::regex_search(search_start, data_text.cend(), region_matches, region_pattern)) {
            std::string region_name = region_matches[1].str();
            std::string region_content = region_matches[2].str();
            
            // Extract material
            std::regex material_pattern("material\\s*=\\s*(\\w+)");
            std::smatch material_match;
            std::string material_name;
            if (std::regex_search(region_content, material_match, material_pattern)) {
                material_name = material_match[1].str();
            }
            
            // Extract elements - modified pattern to match your data format
            std::regex elements_pattern("Elements\\s*\\(\\s*\\d+\\s*\\)\\s*\\{\\s*([^{}]*)\\s*\\}");
            std::smatch elements_match;
            std::vector<int> region_elements;
            if (std::regex_search(region_content, elements_match, elements_pattern)) {
                std::string elements_list = elements_match[1].str();
                std::cout << "Region: " << region_name << ", Material: " << material_name << ", Elements: " << elements_list << std::endl;
                std::istringstream iss(elements_list);
                int element_idx;
                while (iss >> element_idx) {
                    region_elements.push_back(element_idx);
                    // Map element to material
                    element_to_material[element_idx] = material_name;
                    std::cout << "  Mapped element " << element_idx << " to material " << material_name << std::endl;
                }
            }
            
            // Add region info
            RegionInfo info;
            info.name = region_name;
            info.material = material_name;
            info.elements = region_elements;
            region_info.push_back(info);
            
            // Move to next region
            search_start = region_matches.suffix().first;
        }
        
        // Debug output - check how many regions were parsed
        std::cout << "Parsed " << region_info.size() << " regions" << std::endl;
        
        // If no regions were found, try an alternative approach with a more flexible pattern
        if (region_info.empty()) {
            std::cout << "No regions found with standard pattern, trying alternative approach..." << std::endl;
            
            // Alternative pattern that can handle more complex nested structures
            std::regex alt_region_pattern("Region\\s*\\(\\s*\"([^\"]*)\"\\s*\\)\\s*\\{");
            std::smatch alt_region_match;
            search_start = data_text.cbegin();
            
            while (std::regex_search(search_start, data_text.cend(), alt_region_match, alt_region_pattern)) {
                std::string region_name = alt_region_match[1].str();
                size_t region_start = search_start - data_text.cbegin() + alt_region_match.position() + alt_region_match.length();
                
                // Find the closing brace for this region
                size_t pos = region_start;
                int brace_count = 1;
                while (pos < data_text.length() && brace_count > 0) {
                    if (data_text[pos] == '{') {
                        brace_count++;
                    } else if (data_text[pos] == '}') {
                        brace_count--;
                    }
                    pos++;
                }
                
                if (brace_count == 0) {
                    std::string region_content = data_text.substr(region_start, pos - region_start - 1);
                    
                    // Extract material
                    std::regex material_pattern("material\\s*=\\s*(\\w+)");
                    std::smatch material_match;
                    std::string material_name;
                    if (std::regex_search(region_content, material_match, material_pattern)) {
                        material_name = material_match[1].str();
                    }
                    
                    // Extract elements with a more flexible pattern
                    std::regex elements_pattern("Elements\\s*\\(\\s*\\d+\\s*\\)\\s*\\{\\s*([^{}]*)\\s*\\}");
                    std::smatch elements_match;
                    std::vector<int> region_elements;
                    if (std::regex_search(region_content, elements_match, elements_pattern)) {
                        std::string elements_list = elements_match[1].str();
                        std::cout << "Region: " << region_name << ", Material: " << material_name << ", Elements: " << elements_list << std::endl;
                        std::istringstream iss(elements_list);
                        int element_idx;
                        while (iss >> element_idx) {
                            region_elements.push_back(element_idx);
                            // Map element to material
                            element_to_material[element_idx] = material_name;
                            std::cout << "  Mapped element " << element_idx << " to material " << material_name << std::endl;
                        }
                    }
                    
                    // Add region info
                    RegionInfo info;
                    info.name = region_name;
                    info.material = material_name;
                    info.elements = region_elements;
                    region_info.push_back(info);
                }
                
                // Move to next region
                search_start = data_text.cbegin() + pos;
            }
            
            std::cout << "Parsed " << region_info.size() << " regions with alternative approach" << std::endl;
        }
    }
    
    // New method to map materials to faces
    void mapMaterialsToFaces() {
        // For each element, find the faces that belong to it and assign the material
        for (size_t elem_idx = 0; elem_idx < elements.size(); ++elem_idx) {
            const auto& element = elements[elem_idx];
            
            // Check if this element has a material assigned
            auto material_it = element_to_material.find(elem_idx);
            if (material_it == element_to_material.end()) {
                continue;  // Skip elements without material
            }
            
            std::string material = material_it->second;
            
            // In DF-ISE format, elements typically have a specific structure
            // The first value is often the element type, followed by face indices
            // We need to determine which values in the element are face indices
            
            for (size_t i = 0; i < element.size(); ++i) {
                int face_idx = element[i];
                
                // Check if this is a valid face index
                if (face_idx >= 0 && face_idx < static_cast<int>(faces.size())) {
                    face_to_material[face_idx] = material;
                }
                // Handle negative indices (which might indicate orientation)
                else if (face_idx < 0 && -face_idx - 1 < static_cast<int>(faces.size())) {
                    face_to_material[-face_idx - 1] = material;
                }
            }
        }
        
        // Debug output - check how many faces have materials assigned
        std::cout << "Assigned materials to " << face_to_material.size() << " faces out of " << faces.size() << std::endl;
    }
    
    
    void parseCoordSystem(const std::string& coord_text) {
        // Process the coord system text line by line
        std::istringstream coord_stream(coord_text);
        std::string line;
        
        // Temporary storage for vector and matrix data
        std::string translate_text;
        std::string transform_text;
        bool in_translate = false;
        bool in_transform = false;
        int bracket_count = 0;
        
        while (std::getline(coord_stream, line)) {
            // Check for translate vector
            if (!in_translate && !in_transform && line.find("translate") != std::string::npos && line.find("=") != std::string::npos) {
                in_translate = true;
                translate_text = line.substr(line.find("=") + 1);
                
                // Count brackets to track multi-line vectors
                for (char c : translate_text) {
                    if (c == '[') bracket_count++;
                    else if (c == ']') bracket_count--;
                }
                
                // If the vector ends on this line, process it
                if (bracket_count == 0) {
                    in_translate = false;
                    coord_system_vector["translate"] = extractVector(translate_text);
                }
                continue;
            }
            // Check for transform matrix
            else if (!in_translate && !in_transform && line.find("transform") != std::string::npos && line.find("=") != std::string::npos) {
                in_transform = true;
                transform_text = line.substr(line.find("=") + 1);
                
                // Count brackets to track multi-line matrices
                for (char c : transform_text) {
                    if (c == '[') bracket_count++;
                    else if (c == ']') bracket_count--;
                }
                
                // If the matrix ends on this line, process it
                if (bracket_count == 0) {
                    in_transform = false;
                    coord_system_matrix["transform"] = extractMatrix(transform_text);
                }
                continue;
            }
            
            // Continue collecting multi-line vectors or matrices
            if (in_translate) {
                translate_text += line;
                for (char c : line) {
                    if (c == '[') bracket_count++;
                    else if (c == ']') bracket_count--;
                }
                
                if (bracket_count == 0) {
                    in_translate = false;
                    coord_system_vector["translate"] = extractVector(translate_text);
                }
            }
            else if (in_transform) {
                transform_text += line;
                for (char c : line) {
                    if (c == '[') bracket_count++;
                    else if (c == ']') bracket_count--;
                }
                
                if (bracket_count == 0) {
                    in_transform = false;
                    coord_system_matrix["transform"] = extractMatrix(transform_text);
                }
            }
        }
    }

    void parseVertices(const std::string& vertices_text) {
        // Reserve space based on estimated number of vertices
        size_t line_count = std::count(vertices_text.begin(), vertices_text.end(), '\n');
        vertices.reserve(vertices.size() + line_count);
        
        std::istringstream iss(vertices_text);
        std::string line;
        while (std::getline(iss, line)) {
            // Skip empty lines
            if (line.empty() || line.find_first_not_of(" \t\n\r") == std::string::npos) {
                continue;
            }
            
            // Parse vertex coordinates
            std::vector<double> coords;
            coords.reserve(3); // Most vertices are 3D
            
            std::istringstream line_iss(line);
            double coord;
            while (line_iss >> coord) {
                coords.push_back(coord);
            }
            
            if (coords.size() == 3) {
                vertices.push_back(std::move(coords));
            }
        }
    }

    void parseEdges(const std::string& edges_text) {
        // Reserve space based on estimated number of edges
        size_t line_count = std::count(edges_text.begin(), edges_text.end(), '\n');
        edges.reserve(edges.size() + line_count);
        
        std::istringstream iss(edges_text);
        std::string line;
        while (std::getline(iss, line)) {
            // Skip empty lines
            if (line.empty() || line.find_first_not_of(" \t\n\r") == std::string::npos) {
                continue;
            }
            
            // Parse edge vertices
            std::vector<int> indices;
            indices.reserve(2); // Most edges have 2 vertices
            
            std::istringstream line_iss(line);
            int index;
            while (line_iss >> index) {
                indices.push_back(index);
            }
            
            if (indices.size() == 2) {
                edges.push_back(std::move(indices));
            }
        }
    }

    void parseFaces(const std::string& faces_text) {
        // Reserve space based on estimated number of faces
        size_t line_count = std::count(faces_text.begin(), faces_text.end(), '\n');
        faces.reserve(faces.size() + line_count);
        
        std::istringstream iss(faces_text);
        std::string line;
        while (std::getline(iss, line)) {
            // Skip empty lines
            if (line.empty() || line.find_first_not_of(" \t\n\r") == std::string::npos) {
                continue;
            }
            
            // Parse face vertices
            std::vector<int> indices;
            indices.reserve(4); // Most faces are quads or triangles
            
            std::istringstream line_iss(line);
            int index;
            while (line_iss >> index) {
                indices.push_back(index);
            }
            
            if (!indices.empty()) {
                faces.push_back(std::move(indices));
            }
        }
    }

    void parseElements(const std::string& elements_text) {
        std::istringstream iss(elements_text);
        std::string line;
        std::vector<int> current_element;
        bool in_element = false;

        while (std::getline(iss, line)) {
            std::istringstream line_iss(line);
            std::vector<int> line_numbers;
            int number;
            
            // Read all numbers from the current line
            while (line_iss >> number) {
                line_numbers.push_back(number);
            }
            
            // Skip empty lines
            if (line_numbers.empty()) continue;
            
            // Check if this is the start of a new element
            if (line_numbers[0] == 10 && line_numbers.size() == 12) {
                // If we were already processing an element, add it to elements
                if (!current_element.empty()) {
                    elements.push_back(current_element);
                }
                
                // Start new element, skip the first two numbers (10 and descriptor)
                current_element.clear();
                for (size_t i = 2; i < line_numbers.size(); i++) {
                    current_element.push_back(line_numbers[i]);
                }
                in_element = true;
            }
            // If we're in an element, add all numbers from this line
            else if (in_element) {
                current_element.insert(current_element.end(), line_numbers.begin(), line_numbers.end());
            }
        }
        
        // Add the last element if there is one
        if (!current_element.empty()) {
            elements.push_back(current_element);
        }
    }
    
    public:

    std::vector<std::vector<double>> getVertices() const {
        return vertices;
    }

    std::vector<std::vector<int>> getEdges() const {
        return edges;
    }

    std::vector<std::vector<int>> getElements() const {
        return elements;
    }

    std::map<std::string, double> getInfoDouble() const {
        return info_double;
    }

    std::map<std::string, int> getInfoInt() const {
        return info_int;
    }

    std::map<std::string, std::string> getInfoString() const {
        return info_string;
    }

    std::vector<std::string> getRegions() const {
        return regions;
    }

    std::vector<std::string> getMaterials() const {
        return materials;
    }
     // Export the geometry to Wavefront OBJ format
     bool exportToObj(const std::string& output_file) {
        std::ofstream file(output_file);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << output_file << " for writing" << std::endl;
            return false;
        }
        
        // Write OBJ header with information
        file << "# Wavefront OBJ file exported from DF-ISE format" << std::endl;
        file << "# Original file: " << file_path << std::endl;
        file << "# Exported by DFISEParser" << std::endl;
        file << "# Vertices: " << vertices.size() << std::endl;
        file << "# Faces: " << faces.size() << std::endl;
        file << std::endl;
        
        // Write vertices (v x y z)
        for (const auto& vertex : vertices) {
            if (vertex.size() == 3) {
                file << "v " << vertex[0] << " " << vertex[1] << " " << vertex[2] << std::endl;
            }
        }
        file << std::endl;
        
        // Write faces (f v1 v2 v3 ...)
        // Note: OBJ format uses 1-based indexing, while our internal representation is 0-based
        for (const auto& face : faces) {
            if (!face.empty()) {
                file << "f";
                std::vector<int> face_vertices;
                
                // Process each edge in face (skip first element if it's count)
                for (size_t i = 1; i < face.size(); ++i) {
                    int edge_idx = face[i];
                    bool reverse = edge_idx < 0;
                    int abs_edge_idx;
                    if (reverse) {
                        abs_edge_idx = -edge_idx - 1;                    
                    } else {
                        abs_edge_idx = edge_idx;
                    }
                    
                    if (abs_edge_idx >= 0 && abs_edge_idx < edges.size()) {
                        const auto& edge = edges[abs_edge_idx];
                        if (edge.size() == 2) {
                            // Add vertices in correct order based on edge direction
                            if (reverse) {
                                face_vertices.push_back(edge[1]);
                                face_vertices.push_back(edge[0]);
                            } else {
                                face_vertices.push_back(edge[0]);
                                face_vertices.push_back(edge[1]);
                            }
                        }
                    }
                }
                
                // Remove consecutive duplicates while maintaining order
                std::vector<int> unique_vertices;
                for (auto& v : face_vertices) {
                    if (unique_vertices.empty() || unique_vertices.back() != v) {
                        unique_vertices.push_back(v);
                    }
                }
                
                // Write final vertex indices
                for (int v : unique_vertices) {
                    file << " " << (v + 1);
                }
                file << std::endl;
            }
        }
        
        // If there are no faces but we have edges, write edges as lines
        if (faces.empty() && !edges.empty()) {
            for (const auto& edge : edges) {
                if (edge.size() == 2) {
                    // Convert to 1-based indexing for OBJ format
                    file << "l " << (edge[0] + 1) << " " << (edge[1] + 1) << std::endl;
                }
            }
        }
        
        file.close();
        std::cout << "Successfully exported to OBJ format: " << output_file << std::endl;
        return true;
    }


    std::vector<RegionInfo> getRegionInfo() const {
        return region_info;
    }

    std::map<int, std::string> getElementToMaterial() const {
        return element_to_material;
    }

    std::map<int, std::string> getFaceToMaterial() const {
        return face_to_material;
    }



};
#endif
