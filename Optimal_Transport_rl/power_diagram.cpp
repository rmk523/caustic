#include <omp.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include "power_diagram.hpp"

int polygons_in_VD(const int& N, Eigen::VectorXd proj_xy, Eigen::VectorXd weights, Eigen::VectorXd& polygons, int* start_num, int* length,
                     const int& N_threads, std::vector<Eigen::VectorXi> idx_list, ScalarType* proj_corner)
{
    vector< vector<double> > pgonx(N);
    vector< vector<double> > pgony(N);
    vector< VD::Face_iterator > faceid;
    VectorWp wpoints;
    for(int i = 0; i < N; i ++)
    {
        wpoints.push_back( std::make_pair( Wpoint(Point(proj_xy(2*i), proj_xy(2*i+1)), weights(i)), i ) );
    }
    int extra_num = N;

    Eigen::VectorXi& left_idx = idx_list[0];
    Eigen::VectorXi& right_idx = idx_list[1];
    Eigen::VectorXi& bottom_idx = idx_list[2];
    Eigen::VectorXi& top_idx = idx_list[3];

    for(int i = 0; i < left_idx.size(); i ++)
    {
        int left_id = left_idx(i);
        wpoints.push_back( std::make_pair( Wpoint(Point(2*proj_corner[0] - proj_xy(2*left_id), proj_xy(2*left_id+1)), weights(left_id)), extra_num ) );
        extra_num++;
    }
    for(int i = 0; i < right_idx.size(); i ++)
    {
        int right_id = right_idx(i);
        wpoints.push_back( std::make_pair( Wpoint(Point(2*proj_corner[1] - proj_xy(2*right_id), proj_xy(2*right_id+1)), weights(right_id)), extra_num ) );
        extra_num++;
    }
    for(int i = 0; i < bottom_idx.size(); i ++)
    {
        int bottom_id = bottom_idx(i);
        wpoints.push_back( std::make_pair( Wpoint(Point(proj_xy(2*bottom_id),2*proj_corner[2]-proj_xy(2*bottom_id+1)), weights(bottom_id)), extra_num ) );
        extra_num++;
    }
    for(int i = 0; i < top_idx.size(); i ++)
    {
        int top_id = top_idx(i);
        wpoints.push_back( std::make_pair( Wpoint(Point(proj_xy(2*top_id),2*proj_corner[3]-proj_xy(2*top_id+1)), weights(top_id)), extra_num ) );
        extra_num++;
    }

    RT rt(wpoints.begin(), wpoints.end());
    VD vd(rt);
    
    // set faceid
    faceid.reserve(N);
    for(VD::Face_iterator fit = vd.faces_begin(); fit != vd.faces_end(); ++fit)
    {
        faceid.push_back(fit);
    }

    int N_faces = faceid.size();
#pragma omp parallel for num_threads(N_threads)
    for(int i = 0; i < N; i ++)
    {             
        length[i] = 0;
    }
#pragma omp parallel for num_threads(N_threads)
    for(int i = 0; i < N_faces; i++)
    {

        VD::Face_iterator fit = faceid[i];
        int idx = fit->dual()->info();
        if(idx<N)
        {
            int count = 0;
            auto ec_start = fit->ccb();
            VD::Face::Ccb_halfedge_circulator ec = ec_start;
            do
            {
                const CGAL::Object seg_obj = vd.dual().dual(ec->dual());

                const Segment *dseg = CGAL::object_cast<Segment>(&seg_obj);
                if (dseg) 
                {
                        pgonx[idx].push_back(dseg->target().x());
                        pgony[idx].push_back(dseg->target().y());
                        count ++;
                    }
                    else
                    {
                    const Ray *dray = CGAL::object_cast<Ray>(&seg_obj);
                    if(ec->has_source())
                    {
                        pgonx[idx].push_back(dray->source().x());
                        pgony[idx].push_back(dray->source().y());
                        pgonx[idx].push_back(dray->point(1000).x());
                        pgony[idx].push_back(dray->point(1000).y());
                        count += 2;
                }
                else
                {
                        pgonx[idx].push_back(dray->point(1000).x());
                        pgony[idx].push_back(dray->point(1000).y());
                        count ++;
                    }
                }
            } 
            while ( ++ec != ec_start );

            length[idx] = count;
            if(count > 1000)
            {
                cout << "too many vertices in polygon " << idx << ", length = " << count << endl;
            }
        }
        
    }
    start_num[0] = 0;
    for(int i = 0; i < (N - 1); i++)
    {
        start_num[i + 1] = start_num[i] + length[i];
    }

#pragma omp parallel for num_threads(N_threads)
    for(int i = 0; i < N; i ++)
    {
        for(int j = 0; j < length[i]; ++j)
        {
            polygons(2*(start_num[i] + j)) = pgonx[i][j];
            polygons(2*(start_num[i] + j)+1) = pgony[i][j];
        }
    }
    return (start_num[N - 1] + length[N - 1]);
}

void deal_case( int points_num, Eigen::VectorXd& polygons, int* start_num, int* length, int picture_size, int* case_record, int* case_num)
{
    // 初始化 case_num
    for( int case_iter = 0; case_iter < 4; case_iter++)
    {
        case_num[case_iter] = 0;
    }

    // 处理每个点
    for (int point_iter = 0; point_iter < points_num; point_iter++) {
        int start_index = start_num[point_iter];
        int vertex_count = length[point_iter];
        int boundary_num[4] = {0};

        // 统计边界接触情况
        for (int i = 0; i < vertex_count; i++) {
            double x = polygons[2 * start_index + 2 * i];
            double y = polygons[2 * start_index + 2 * i + 1];

            if (x < 1e-6) boundary_num[0]++;
            if (y < 1e-6) boundary_num[1]++;
            if (x > picture_size - 1e-6) boundary_num[2]++;
            if (y > picture_size - 1e-6) boundary_num[3]++;
        }

        // 记录满足条件的点
        for (int case_iter = 0; case_iter < 4; case_iter++) {
            if (boundary_num[case_iter] > 1) {
                case_record[ case_iter * points_num + case_num[case_iter]] = point_iter;
                case_num[case_iter]++;
            }
        }
    }

    // 输出调试信息
    for (int i = 0; i < 4; i++) {
        std::cerr << "case_num[" << i << "] = " << case_num[i] << std::endl;
    }
}