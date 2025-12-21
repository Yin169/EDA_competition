import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class DFISEParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.info = {}
        self.vertices = []
        self.edges = []
        self.faces = []
        self.elements = []
        self.regions = []
        self.materials = []
        self.coord_system = {}
        
    def parse(self):
        """Parse the DF-ISE format file."""
        with open(self.file_path, 'r') as f:
            content = f.read()
            
        # Extract the Info section
        info_match = re.search(r'Info\s*{(.*?)}', content, re.DOTALL)
        if info_match:
            info_text = info_match.group(1)
            self._parse_info(info_text)
            
        # Extract the Data section
        data_match = re.search(r'Data\s*{(.*)}', content, re.DOTALL)
        if data_match:
            data_text = data_match.group(1)
            self._parse_data(data_text)
            
        return self
    
    def _parse_info(self, info_text):
        """Parse the Info section of the file."""
        # Extract version
        version_match = re.search(r'version\s*=\s*([\d\.]+)', info_text)
        if version_match:
            self.info['version'] = float(version_match.group(1))
            
        # Extract type
        type_match = re.search(r'type\s*=\s*(\w+)', info_text)
        if type_match:
            self.info['type'] = type_match.group(1)
            
        # Extract dimension
        dim_match = re.search(r'dimension\s*=\s*(\d+)', info_text)
        if dim_match:
            self.info['dimension'] = int(dim_match.group(1))
            
        # Extract number of vertices
        vertices_match = re.search(r'nb_vertices\s*=\s*(\d+)', info_text)
        if vertices_match:
            self.info['nb_vertices'] = int(vertices_match.group(1))
            
        # Extract number of edges
        edges_match = re.search(r'nb_edges\s*=\s*(\d+)', info_text)
        if edges_match:
            self.info['nb_edges'] = int(edges_match.group(1))
            
        # Extract number of faces
        faces_match = re.search(r'nb_faces\s*=\s*(\d+)', info_text)
        if faces_match:
            self.info['nb_faces'] = int(faces_match.group(1))
            
        # Extract number of elements
        elements_match = re.search(r'nb_elements\s*=\s*(\d+)', info_text)
        if elements_match:
            self.info['nb_elements'] = int(elements_match.group(1))
            
        # Extract number of regions
        regions_match = re.search(r'nb_regions\s*=\s*(\d+)', info_text)
        if regions_match:
            self.info['nb_regions'] = int(regions_match.group(1))
            
        # Extract regions
        regions_match = re.search(r'regions\s*=\s*\[(.*?)\]', info_text, re.DOTALL)
        if regions_match:
            regions_text = regions_match.group(1)
            # Remove quotes and split by spaces or newlines
            self.regions = [r.strip('" \t\n') for r in regions_text.split('"') if r.strip('" \t\n')]
            
        # Extract materials
        materials_match = re.search(r'materials\s*=\s*\[(.*?)\]', info_text, re.DOTALL)
        if materials_match:
            materials_text = materials_match.group(1)
            # Split by spaces or newlines
            self.materials = [m.strip() for m in materials_text.split() if m.strip()]
    
    def _parse_data(self, data_text):
        """Parse the Data section of the file."""
        # Extract CoordSystem
        coord_match = re.search(r'CoordSystem\s*{(.*?)}', data_text, re.DOTALL)
        if coord_match:
            coord_text = coord_match.group(1)
            self._parse_coord_system(coord_text)
            
        # Extract Vertices
        vertices_match = re.search(r'Vertices\s*\(\d+\)\s*{(.*?)}', data_text, re.DOTALL)
        if vertices_match:
            vertices_text = vertices_match.group(1)
            self._parse_vertices(vertices_text)
            
        # Extract Edges if present
        edges_match = re.search(r'Edges\s*\(\d+\)\s*{(.*?)}', data_text, re.DOTALL)
        if edges_match:
            edges_text = edges_match.group(1)
            self._parse_edges(edges_text)
            
        # Extract Faces if present
        faces_match = re.search(r'Faces\s*\(\d+\)\s*{(.*?)}', data_text, re.DOTALL)
        if faces_match:
            faces_text = faces_match.group(1)
            self._parse_faces(faces_text)
            
        # Extract Elements if present
        elements_match = re.search(r'Elements\s*\(\d+\)\s*{(.*?)}', data_text, re.DOTALL)
        if elements_match:
            elements_text = elements_match.group(1)
            self._parse_elements(elements_text)
    
    def _parse_coord_system(self, coord_text):
        """Parse the CoordSystem section."""
        # Extract translate
        translate_match = re.search(r'translate\s*=\s*\[(.*?)\]', coord_text)
        if translate_match:
            translate_text = translate_match.group(1)
            self.coord_system['translate'] = [float(x) for x in translate_text.split()]
            
        # Extract transform
        transform_match = re.search(r'transform\s*=\s*\[(.*?)\]', coord_text, re.DOTALL)
        if transform_match:
            transform_text = transform_match.group(1)
            # Split by newlines and remove any empty lines
            transform_lines = [line.strip() for line in transform_text.split('\n') if line.strip()]
            transform_matrix = []
            for line in transform_lines:
                # Extract numbers from each line
                row = [float(x) for x in line.split()]
                if row:
                    transform_matrix.append(row)
            self.coord_system['transform'] = transform_matrix
    
    def _parse_vertices(self, vertices_text):
        """Parse the Vertices section."""
        # Split by newlines and remove any empty lines
        lines = [line.strip() for line in vertices_text.split('\n') if line.strip()]
        for line in lines:
            # Extract the three coordinates
            coords = [float(x) for x in line.split()]
            if len(coords) == 3:
                self.vertices.append(coords)
    
    def _parse_edges(self, edges_text):
        """Parse the Edges section."""
        # Split by newlines and remove any empty lines
        lines = [line.strip() for line in edges_text.split('\n') if line.strip()]
        for line in lines:
            # Extract the two vertex indices
            indices = [int(x) for x in line.split()]
            if len(indices) == 2:
                self.edges.append(indices)
    
    def _parse_faces(self, faces_text):
        """Parse the Faces section."""
        # This would depend on the specific format of faces in your file
        # For now, just split by newlines and parse each line
        lines = [line.strip() for line in faces_text.split('\n') if line.strip()]
        for line in lines:
            # Extract the vertex indices that form the face
            indices = [int(x) for x in line.split()]
            if indices:
                self.faces.append(indices)
    
    def _parse_elements(self, elements_text):
        """Parse the Elements section."""
        # This would depend on the specific format of elements in your file
        # For now, just split by newlines and parse each line
        lines = [line.strip() for line in elements_text.split('\n') if line.strip()]
        for line in lines:
            # Extract the face indices that form the element
            indices = [int(x) for x in line.split()]
            if indices:
                self.elements.append(indices)
    
    def get_vertices(self):
        """Return the vertices as a numpy array."""
        return np.array(self.vertices)
    
    def get_edges(self):
        """Return the edges as a numpy array."""
        return np.array(self.edges)
    
    def get_faces(self):
        """Return the faces as a list of lists."""
        return self.faces
    
    def get_elements(self):
        """Return the elements as a list of lists."""
        return self.elements
    
    def get_info(self):
        """Return the info dictionary."""
        return self.info
    
    def get_regions(self):
        """Return the regions list."""
        return self.regions
    
    def get_materials(self):
        """Return the materials list."""
        return self.materials

    def plot_geometry(self, show_vertices=True, show_edges=True, vertex_size=10, edge_color='b', vertex_color='r'):
        """
        Plot the 3D geometry of the structure.
        
        Parameters:
        -----------
        show_vertices : bool
            Whether to show vertices in the plot
        show_edges : bool
            Whether to show edges in the plot
        vertex_size : int
            Size of the vertex points
        edge_color : str
            Color of the edges
        vertex_color : str
            Color of the vertices
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        vertices = self.get_vertices()
        edges = self.get_edges()
        
        # Plot vertices
        if show_vertices and len(vertices) > 0:
            ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                       color=vertex_color, s=vertex_size, label='Vertices')
        
        # Plot edges
        if show_edges and len(edges) > 0:
            for edge in edges:
                if len(edge) == 2:
                    v1, v2 = edge
                    # Check if indices are valid
                    if 0 <= v1 < len(vertices) and 0 <= v2 < len(vertices):
                        ax.plot([vertices[v1, 0], vertices[v2, 0]],
                                [vertices[v1, 1], vertices[v2, 1]],
                                [vertices[v1, 2], vertices[v2, 2]],
                                color=edge_color, linewidth=1)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Geometry from DF-ISE File')
        
        # Add legend
        if show_vertices:
            ax.scatter([], [], [], color=vertex_color, s=vertex_size, label='Vertices')
        if show_edges:
            ax.plot([], [], [], color=edge_color, linewidth=1, label='Edges')
        ax.legend()
        
        plt.tight_layout()
        return fig, ax



if __name__ == "__main__":
    parser = DFISEParser("/Users/yincheangng/worksapce/Github/EDA_competition/data/Silicon_etch_result.bnd")
    parser.parse()
    
    # Print some basic information
    print(f"File version: {parser.info.get('version')}")
    print(f"Type: {parser.info.get('type')}")
    print(f"Dimension: {parser.info.get('dimension')}")
    print(f"Number of vertices: {parser.info.get('nb_vertices')}")
    print(f"Number of edges: {parser.info.get('nb_edges')}")
    print(f"Number of faces: {parser.info.get('nb_faces')}")
    print(f"Number of elements: {parser.info.get('nb_elements')}")
    print(f"Number of regions: {parser.info.get('nb_regions')}")
    
    # Print the first few vertices
    vertices = parser.get_vertices()
    if len(vertices) > 0:
        print("\nFirst 5 vertices:")
        for i in range(min(5, len(vertices))):
            print(f"Vertex {i+1}: {vertices[i]}")
    
    # Print the first few edges
    edges = parser.get_edges()
    if len(edges) > 0:
        print("\nFirst 5 edges:")
        for i in range(min(5, len(edges))):
            print(f"Edge {i+1}: {edges[i]}")
    
    # Print the regions and materials
    print("\nRegions:")
    for i, region in enumerate(parser.get_regions()):
        print(f"Region {i+1}: {region}")
    
    print("\nMaterials:")
    for i, material in enumerate(parser.get_materials()):
        print(f"Material {i+1}: {material}")

    # Plot the geometry
    fig, ax = parser.plot_geometry(True, True, 2)
    plt.show()
