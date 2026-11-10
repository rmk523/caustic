#ifndef POWER_DIAGRAM_HPP_
#define POWER_DIAGRAM_HPP_

#define BOOST_BIND_NO_PLACEHOLDERS

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Regular_triangulation_adaptation_traits_2.h>
#include <CGAL/Regular_triangulation_adaptation_policies_2.h>
#include <CGAL/Regular_triangulation_2.h>
#include <CGAL/Weighted_point_2.h>
#include <CGAL/Voronoi_diagram_2.h>
#include <CGAL/intersections.h>
#include <CGAL/Polygon_2.h>
#include <Eigen/Dense>
#include <iostream>

typedef CGAL::Exact_predicates_inexact_constructions_kernel             K;
typedef CGAL::Regular_triangulation_vertex_base_2<K>                    Vbase;
typedef CGAL::Triangulation_vertex_base_with_info_2<int, K, Vbase>      Vb;
typedef CGAL::Regular_triangulation_face_base_2<K>                      Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb>                    Tds;
typedef CGAL::Regular_triangulation_2<K, Tds>                           RT;
typedef CGAL::Regular_triangulation_adaptation_traits_2<RT>             AT;
typedef CGAL::Regular_triangulation_degeneracy_removal_policy_2<RT>     DRP;
typedef CGAL::Voronoi_diagram_2<RT, AT, DRP>                            VD;
typedef K::Point_2                                                      Point;
typedef K::Weighted_point_2                                             Wpoint;
typedef K::Iso_rectangle_2                                              Iso_rectangle;
typedef K::Segment_2                                                    Segment;
typedef K::Ray_2                                                        Ray;
typedef std::vector< std::pair<Wpoint, int> >                           VectorWp;

typedef double ScalarType;
const double M_2PI = 2 * M_PI;
// typedef OpenMesh::TriMesh_ArrayKernelT<>  MyMesh;
// typedef MyMesh::VertexHandle VH;
// typedef std::vector<VH> vecVH;
#define max_edge_num 400

using namespace std;

int polygons_in_VD(const int& N, Eigen::VectorXd proj_xy, Eigen::VectorXd weights, Eigen::VectorXd& polygons, int* start_num, int* length,
                     const int& N_threads, std::vector<Eigen::VectorXi> idx_list, ScalarType* proj_corner);
void deal_case( int points_num, Eigen::VectorXd& polygons, int* start_num, int* length, int picture_size, int* case_record, int* case_num);
#endif // POWER_DIAGRAM_HPP_S