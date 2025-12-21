#ifndef ALPHAWRAP_HPP
#define ALPHAWRAP_HPP

#include "output_helper.h"
 
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
 
#include <CGAL/alpha_wrap_3.h>
#include <CGAL/Polygon_mesh_processing/bbox.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Real_timer.h>

#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/remesh_planar_patches.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/detect_features.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
 
#include <iostream>
#include <string>
 
 
void Wrapper(const std::string& inputFile, const double relative_alpha, const double relative_offset){

  namespace PMP = CGAL::Polygon_mesh_processing;
  using K = CGAL::Exact_predicates_inexact_constructions_kernel;
  using Point_3 = K::Point_3;
  using Mesh = CGAL::Surface_mesh<Point_3>;

  // Read the input
  const std::string filename = inputFile;
  std::cout << "Reading " << filename << "..." << std::endl;
 
  Mesh mesh;
  if(!PMP::IO::read_polygon_mesh(filename, mesh) || is_empty(mesh) || !is_triangle_mesh(mesh))
  {
    std::cerr << "Invalid input:" << filename << std::endl;
    return;
  }
 
  std::cout << "Input: " << num_vertices(mesh) << " vertices, " << num_faces(mesh) << " faces" << std::endl;

 
  CGAL::Bbox_3 bbox = CGAL::Polygon_mesh_processing::bbox(mesh);
  const double diag_length = std::sqrt(CGAL::square(bbox.xmax() - bbox.xmin()) +
                                       CGAL::square(bbox.ymax() - bbox.ymin()) +
                                       CGAL::square(bbox.zmax() - bbox.zmin()));
 
  const double alpha = diag_length / relative_alpha;
  const double offset = diag_length / relative_offset;
  std::cout << "alpha: " << alpha << ", offset: " << offset << std::endl;
 
  // Construct the wrap
  CGAL::Real_timer t;
  t.start();
 
  Mesh wrap;
  CGAL::alpha_wrap_3(mesh, alpha, offset, wrap);
 
  t.stop();
  std::cout << "Result: " << num_vertices(wrap) << " vertices, " << num_faces(wrap) << " faces" << std::endl;
  std::cout << "Took " << t.time() << " s." << std::endl;

  // Mesh::Property_map<Mesh::Edge_index, bool> ecm =
    // wrap.add_property_map<Mesh::Edge_index, bool>("ecm",false).first;
 
  // detect sharp edges of the cube
  // PMP::detect_sharp_edges(wrap, 60, ecm);
 
  // create a remeshed version of the cube with many elements
  // PMP::isotropic_remeshing(faces(wrap), 0.1, wrap, CGAL::parameters::edge_is_constrained_map(ecm));
//  
  // decimate the mesh
  // Mesh out;
  // PMP::remesh_planar_patches(wrap, out);
 
  PMP::stitch_borders(wrap);
  if (PMP::does_self_intersect(wrap)) {std::cout << "self intersection found" << std::endl;}
  
  // Save the result
  const std::string output_name = generate_output_name(filename, relative_alpha, relative_offset);
  std::cout << "Writing to " << output_name << std::endl;
  CGAL::IO::write_polygon_mesh(output_name, wrap, CGAL::parameters::stream_precision(17));
 
}

#endif