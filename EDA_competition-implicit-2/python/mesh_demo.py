import numpy as np
import matplotlib.pyplot as plt
from readfile import DFISEParser
from cell_cuts import CellCutsMesh

def main():
    # Parse the geometry file
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
    
    # Create a cell-cuts mesh
    # Use a smaller grid for faster computation
    mesh = CellCutsMesh(parser, nx=20, ny=20, nz=20)
    
    # Generate the mesh
    mesh_data = mesh.generate_mesh()
    
    # Print mesh information
    print("\nMesh Information:")
    print(f"Dimensions: {mesh_data['dimensions']}")
    print(f"Bounds: {mesh_data['bounds']}")
    print(f"Number of cut cells: {len(mesh_data['cut_cells'])}")
    
    # Count cell types
    cell_types = mesh_data['cell_types']
    inside_count = np.sum(cell_types == 1)
    cut_count = np.sum(cell_types == 2)
    outside_count = np.sum(cell_types == 0)
    
    print(f"Inside cells: {inside_count}")
    print(f"Cut cells: {cut_count}")
    print(f"Outside cells: {outside_count}")
    
    # Visualize with Taichi GUI
    mesh.visualize_with_gui(show_cut_cells=True, show_inside_cells=True, 
                        show_geometry=True, show_slice=True, 
                        slice_axis='z', slice_pos=0.5)

if __name__ == "__main__":
    main()