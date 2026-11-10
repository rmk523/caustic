#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include "clip.h"
using namespace std;

#if !defined (__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

void write_obj(const char* filename, ScalarType* points, int* faces, int points_num, int faces_num) {
    ofstream out(filename);
    if (!out.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return;
    }
    for (int i = 0; i < points_num; i++) {
        out << "v " << points[2*i] << " " << points[2*i + 1] << " 0.0" << endl;
    }
    for (int i = 0; i < faces_num; i++) {
        out << "f " 
            << faces[3*i] + 1 << " " 
            << faces[3*i + 1] + 1 << " " 
            << faces[3*i + 2] + 1 << endl;
    }
    out.close();
    cout << "OBJ file saved as " << filename << endl;
}

__global__ void compute_render_diff( ScalarType* d_render_result, ScalarType* d_real_picture, ScalarType* d_render_grad, int* d_length)
{    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < d_length[0])
    {

        d_render_grad[tid] = d_render_result[tid] - d_real_picture[tid];

    }
}

__global__ void compute_sobel_weight( ScalarType* d_render_grad, ScalarType* d_sobel_weights, ScalarType* d_pic_size, int* d_length)
{
    // (column_num,row_num)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < d_length[0])
    {
        int d_pic_size_int[2] = {(int)d_pic_size[0],(int)d_pic_size[1]};

        int row_num = tid / d_pic_size_int[0];
        int column_num = tid % d_pic_size_int[0];
        d_sobel_weights[tid] = 0;
        if(row_num != 0)
        {
            d_sobel_weights[tid] += d_render_grad[tid] - d_render_grad[tid - d_pic_size_int[0]];
        }
        if(row_num < (d_pic_size_int[0] - 1))
        {
            d_sobel_weights[tid] += d_render_grad[tid] - d_render_grad[tid + d_pic_size_int[0]];
        }
        if(column_num != 0)
        {
            d_sobel_weights[tid] += d_render_grad[tid] - d_render_grad[tid - 1];
        }
        if(column_num < (d_pic_size_int[0] - 1))
        {
            d_sobel_weights[tid] += d_render_grad[tid] - d_render_grad[tid + 1];
        }
    }
}

__global__ void compute_render( ScalarType* d_compute_tri_area, int* d_faces, ScalarType* d_proj_xy, ScalarType* d_total_gray_value,
     ScalarType* d_render_result, int* d_face_num, ScalarType* d_proj_domain, ScalarType* d_pic_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < d_face_num[0])
    {
        int d_pic_size_int[2] = {(int)d_pic_size[0],(int)d_pic_size[1]};

        ScalarType gray_value = d_compute_tri_area[tid]*d_total_gray_value[0];
        ScalarType x_step = (d_proj_domain[1] - d_proj_domain[0])/d_pic_size[0];
        ScalarType y_step = (d_proj_domain[3] - d_proj_domain[2])/d_pic_size[1];

        int point_idx[3] = { d_faces[3*tid], d_faces[3*tid+1], d_faces[3*tid+2]};
        ScalarType tri_coor[6];
        int poly_num = 3;
        for(int i = 0; i < 3; i++)
        {
            tri_coor[2 * i] = d_proj_xy[2 * point_idx[i]];
            tri_coor[2 * i + 1] = d_proj_xy[2 * point_idx[i] + 1];
        }
        ScalarType big_area = compute_area( tri_coor, poly_num);
        
        ScalarType max_x = tri_coor[0], max_y = tri_coor[1], min_x = tri_coor[0], min_y=tri_coor[1];
        for(int i=0;i<3;i++)
        {
            if(tri_coor[2*i]<min_x)
            {
                min_x = tri_coor[2*i];
            }
            if(tri_coor[2*i]>max_x)
            {
                max_x = tri_coor[2*i];
            }
            if(tri_coor[2*i+1]<min_y)
            {
                min_y = tri_coor[2*i+1];
            }
            if(tri_coor[2*i+1]>max_y)
            {
                max_y = tri_coor[2*i+1];
            }
        }
        
        int low_bound_x  = (int)floor((min_x - d_proj_domain[0]) / x_step);
        int low_bound_y  = (int)floor((min_y - d_proj_domain[2]) / y_step);
        int high_bound_x = (int)floor((max_x - d_proj_domain[0]) / x_step) + 1;
        int high_bound_y = (int)floor((max_y - d_proj_domain[2]) / y_step) + 1;
        
        low_bound_x=(low_bound_x>0)?low_bound_x:0;
        low_bound_y=(low_bound_y>0)?low_bound_y:0;
        high_bound_x=(high_bound_x<d_pic_size_int[0])?high_bound_x:d_pic_size_int[0];
        high_bound_y=(high_bound_y<d_pic_size_int[1])?high_bound_y:d_pic_size_int[1];

        ScalarType clipsquare[4];
        ScalarType new_poly[max_edge_num];
        int new_poly_num;

        for(int i_x = low_bound_x; i_x < high_bound_x; i_x++)
        {
            for(int i_y = low_bound_y; i_y < high_bound_y; i_y++)
            {
                int grid = i_y * d_pic_size_int[0] + i_x; 
                clipsquare[0] = (ScalarType)i_x * x_step + d_proj_domain[0];
                clipsquare[1] = (ScalarType)i_y * y_step + d_proj_domain[2];
                clipsquare[2] = (ScalarType)(i_x + 1) * x_step + d_proj_domain[0];
                clipsquare[3] = (ScalarType)(i_y + 1) * y_step + d_proj_domain[2];

                clip_Polygon( tri_coor, poly_num, clipsquare, new_poly, new_poly_num);
                ScalarType area = compute_area( new_poly, new_poly_num);

                if(big_area!=0)
                {
                    ScalarType local_contribution = gray_value*area/big_area;
                    atomicAdd(&d_render_result[grid], local_contribution);
                }  
            }
        }
    }
}

__global__ void compute_grad_render( ScalarType* d_compute_tri_area, ScalarType* d_compute_tri_area_grad, int* d_faces, ScalarType* d_proj_xy,
     ScalarType* d_proj_xy_render_grad, ScalarType* d_total_gray_value, ScalarType* d_render_diff, int* d_face_num, ScalarType* d_proj_domain,
      ScalarType* d_pic_size)
{
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   if(tid < d_face_num[0])
   {
        int d_pic_size_int[2] = {(int)d_pic_size[0],(int)d_pic_size[1]};

        ScalarType gray_value = d_compute_tri_area[tid] * d_total_gray_value[0];
        ScalarType x_step = (d_proj_domain[1] - d_proj_domain[0])/d_pic_size[0];
        ScalarType y_step = (d_proj_domain[3] - d_proj_domain[2])/d_pic_size[1];

        int point_idx[3] = { d_faces[3*tid], d_faces[3*tid+1], d_faces[3*tid+2]};
        ScalarType tri_coor[6];
        ScalarType big_area_grad[6] = {0};
        ScalarType small_area_grad[6];
        ScalarType clip_grad[36] = {0};
        ScalarType render_total_grad[6] = {0};

        int poly_num = 3;
        for(int i = 0; i < 3; i++)
        {
            tri_coor[2 * i] = d_proj_xy[2 * point_idx[i]];
            tri_coor[2 * i + 1] = d_proj_xy[2 * point_idx[i] + 1];
        }
        for(int i = 0; i < 6; i++)
        {
            clip_grad[7 * i] = 1;
        }

        ScalarType big_area = compute_grad_area( tri_coor, big_area_grad, poly_num);
       
        ScalarType max_x = tri_coor[0], max_y = tri_coor[1], min_x = tri_coor[0], min_y=tri_coor[1];
        for(int i=0;i<3;i++)
        {
            if(tri_coor[2*i]<min_x)
            {
                min_x = tri_coor[2*i];
            }
            if(tri_coor[2*i]>max_x)
            {
                max_x = tri_coor[2*i];
            }
            if(tri_coor[2*i+1]<min_y)
            {
                min_y = tri_coor[2*i+1];
            }
            if(tri_coor[2*i+1]>max_y)
            {
                max_y = tri_coor[2*i+1];
            }
        }

        int low_bound_x  = (int)floor((min_x - d_proj_domain[0]) / x_step);
        int low_bound_y  = (int)floor((min_y - d_proj_domain[2]) / y_step);
        int high_bound_x = (int)floor((max_x - d_proj_domain[0]) / x_step) + 1;
        int high_bound_y = (int)floor((max_y - d_proj_domain[2]) / y_step) + 1;
        
        low_bound_x=(low_bound_x>0)?low_bound_x:0;
        low_bound_y=(low_bound_y>0)?low_bound_y:0;
        high_bound_x=(high_bound_x<d_pic_size_int[0])?high_bound_x:d_pic_size_int[0];
        high_bound_y=(high_bound_y<d_pic_size_int[1])?high_bound_y:d_pic_size_int[1];

        ScalarType clipsquare[4];
        ScalarType new_poly[max_edge_num];
        ScalarType clip_poly_grad[6*max_edge_num];
        ScalarType new_poly_grad[max_edge_num] = {0};
        int new_poly_num;
        ScalarType local_area_diff = 0;

        for(int i_x = low_bound_x; i_x < high_bound_x; i_x++)
        {
            for(int i_y = low_bound_y; i_y < high_bound_y; i_y++)
            {
                int grid = i_y * d_pic_size_int[0] + i_x; 
                clipsquare[0] = (ScalarType)i_x * x_step + d_proj_domain[0];
                clipsquare[1] = (ScalarType)i_y * y_step + d_proj_domain[2];
                clipsquare[2] = (ScalarType)(i_x + 1) * x_step + d_proj_domain[0];
                clipsquare[3] = (ScalarType)(i_y + 1) * y_step + d_proj_domain[2];

                for(int i = 0; i < 6; i++)
                {
                    small_area_grad[i] = 0;
                }

                grad_clip_Polygon( tri_coor, clip_grad, poly_num, clipsquare, new_poly, clip_poly_grad, new_poly_num);

                for(int i = 0; i < new_poly_num*2; i++)
                {
                    new_poly_grad[i] = 0;
                }
                ScalarType area = compute_grad_area( new_poly, new_poly_grad, new_poly_num);
                accumulate_grad( small_area_grad, clip_poly_grad, new_poly_grad, new_poly_num);
                if(big_area!=0)
                {
                    ScalarType render_diff = d_render_diff[grid];
                    local_area_diff = local_area_diff + 2 * render_diff * area/big_area;
                    for(int i = 0; i < 6; i++)
                    {
                        render_total_grad[i] += 2 * render_diff * gray_value * (small_area_grad[i]*big_area - big_area_grad[i]*area)/(big_area * big_area);
                    }
                }   
            }
        }

        atomicAdd(&d_proj_xy_render_grad[2*point_idx[0]],render_total_grad[0]);
        atomicAdd(&d_proj_xy_render_grad[2*point_idx[0]+1],render_total_grad[1]);
        atomicAdd(&d_proj_xy_render_grad[2*point_idx[1]],render_total_grad[2]);
        atomicAdd(&d_proj_xy_render_grad[2*point_idx[1]+1],render_total_grad[3]);
        atomicAdd(&d_proj_xy_render_grad[2*point_idx[2]],render_total_grad[4]);
        atomicAdd(&d_proj_xy_render_grad[2*point_idx[2]+1],render_total_grad[5]);
        local_area_diff = local_area_diff * d_total_gray_value[0];
        atomicAdd(&d_compute_tri_area_grad[tid] , local_area_diff);
    }
}

__global__ void compute_grad_render_sobel( ScalarType* d_compute_tri_area, ScalarType* d_compute_tri_area_grad, ScalarType* d_sobel_tri_area_grad, int* d_faces,
     ScalarType* d_proj_xy, ScalarType* d_proj_xy_render_grad, ScalarType* d_proj_xy_sobel_grad, ScalarType* d_total_gray_value, ScalarType* d_render_diff,
     ScalarType* d_sobel_weights, int* d_face_num, ScalarType* d_proj_domain, ScalarType* d_pic_size)
{
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   if(tid < d_face_num[0])
   {
        int d_pic_size_int[2] = {(int)d_pic_size[0],(int)d_pic_size[1]};

        ScalarType gray_value = d_compute_tri_area[tid] * d_total_gray_value[0];
        ScalarType x_step = (d_proj_domain[1] - d_proj_domain[0])/d_pic_size[0];
        ScalarType y_step = (d_proj_domain[3] - d_proj_domain[2])/d_pic_size[1];

        int point_idx[3] = { d_faces[3*tid], d_faces[3*tid+1], d_faces[3*tid+2]};
        ScalarType tri_coor[6];
        ScalarType big_area_grad[6] = {0};
        ScalarType small_area_grad[6];
        ScalarType clip_grad[36] = {0};
        ScalarType render_total_grad[6] = {0};
        ScalarType sobel_total_grad[6] = {0};

        int poly_num = 3;
        for(int i = 0; i < 3; i++)
        {
            tri_coor[2 * i] = d_proj_xy[2 * point_idx[i]];
            tri_coor[2 * i + 1] = d_proj_xy[2 * point_idx[i] + 1];
        }
        for(int i = 0; i < 6; i++)
        {
            clip_grad[7 * i] = 1;
        }

        ScalarType big_area = compute_grad_area( tri_coor, big_area_grad, poly_num);
       
        ScalarType max_x = tri_coor[0], max_y = tri_coor[1], min_x = tri_coor[0], min_y=tri_coor[1];
        for(int i=0;i<3;i++)
        {
            if(tri_coor[2*i]<min_x)
            {
                min_x = tri_coor[2*i];
            }
            if(tri_coor[2*i]>max_x)
            {
                max_x = tri_coor[2*i];
            }
            if(tri_coor[2*i+1]<min_y)
            {
                min_y = tri_coor[2*i+1];
            }
            if(tri_coor[2*i+1]>max_y)
            {
                max_y = tri_coor[2*i+1];
            }
        }

        int low_bound_x  = (int)floor((min_x - d_proj_domain[0]) / x_step);
        int low_bound_y  = (int)floor((min_y - d_proj_domain[2]) / y_step);
        int high_bound_x = (int)floor((max_x - d_proj_domain[0]) / x_step) + 1;
        int high_bound_y = (int)floor((max_y - d_proj_domain[2]) / y_step) + 1;
        
        low_bound_x=(low_bound_x>0)?low_bound_x:0;
        low_bound_y=(low_bound_y>0)?low_bound_y:0;
        high_bound_x=(high_bound_x<d_pic_size_int[0])?high_bound_x:d_pic_size_int[0];
        high_bound_y=(high_bound_y<d_pic_size_int[1])?high_bound_y:d_pic_size_int[1];

       ScalarType clipsquare[4];
       ScalarType new_poly[max_edge_num];
       ScalarType clip_poly_grad[6*max_edge_num];
       ScalarType new_poly_grad[max_edge_num] = {0};
       int new_poly_num;
       ScalarType local_area_diff = 0;
       ScalarType local_sobel_diff = 0;

       for(int i_x = low_bound_x; i_x < high_bound_x; i_x++)
       {
           for(int i_y = low_bound_y; i_y < high_bound_y; i_y++)
           {
                int grid = i_y * d_pic_size_int[0] + i_x; 
                clipsquare[0] = (ScalarType)i_x * x_step + d_proj_domain[0];
                clipsquare[1] = (ScalarType)i_y * y_step + d_proj_domain[2];
                clipsquare[2] = (ScalarType)(i_x + 1) * x_step + d_proj_domain[0];
                clipsquare[3] = (ScalarType)(i_y + 1) * y_step + d_proj_domain[2];

               for(int i = 0; i < 6; i++)
               {
                   small_area_grad[i] = 0;
               }

               grad_clip_Polygon( tri_coor, clip_grad, poly_num, clipsquare, new_poly, clip_poly_grad, new_poly_num);

               for(int i = 0; i < new_poly_num*2; i++)
               {
                   new_poly_grad[i] = 0;
               }
               ScalarType area = compute_grad_area( new_poly, new_poly_grad, new_poly_num);
               accumulate_grad( small_area_grad, clip_poly_grad, new_poly_grad, new_poly_num);
               if(big_area!=0)
               {
                   ScalarType render_diff = d_render_diff[grid];
                   ScalarType sobel_weight = d_sobel_weights[grid];
                   local_area_diff = local_area_diff + 2 * render_diff * area/big_area;
                   local_sobel_diff = local_sobel_diff + 2 * sobel_weight * area/big_area;
                   for(int i = 0; i < 6; i++)
                   {
                        render_total_grad[i] += 2 * render_diff * gray_value * (small_area_grad[i]*big_area - big_area_grad[i]*area)/(big_area * big_area);
                        sobel_total_grad[i] += 2 * sobel_weight * gray_value * (small_area_grad[i]*big_area - big_area_grad[i]*area)/(big_area * big_area);
                   }
               }   
           }
       }

        atomicAdd(&d_proj_xy_render_grad[2*point_idx[0]],render_total_grad[0]);
        atomicAdd(&d_proj_xy_render_grad[2*point_idx[0]+1],render_total_grad[1]);
        atomicAdd(&d_proj_xy_render_grad[2*point_idx[1]],render_total_grad[2]);
        atomicAdd(&d_proj_xy_render_grad[2*point_idx[1]+1],render_total_grad[3]);
        atomicAdd(&d_proj_xy_render_grad[2*point_idx[2]],render_total_grad[4]);
        atomicAdd(&d_proj_xy_render_grad[2*point_idx[2]+1],render_total_grad[5]);

        atomicAdd(&d_proj_xy_sobel_grad[2*point_idx[0]],sobel_total_grad[0]);
        atomicAdd(&d_proj_xy_sobel_grad[2*point_idx[0]+1],sobel_total_grad[1]);
        atomicAdd(&d_proj_xy_sobel_grad[2*point_idx[1]],sobel_total_grad[2]);
        atomicAdd(&d_proj_xy_sobel_grad[2*point_idx[1]+1],sobel_total_grad[3]);
        atomicAdd(&d_proj_xy_sobel_grad[2*point_idx[2]],sobel_total_grad[4]);
        atomicAdd(&d_proj_xy_sobel_grad[2*point_idx[2]+1],sobel_total_grad[5]);
        
        local_area_diff = local_area_diff * d_total_gray_value[0];
        atomicAdd(&d_compute_tri_area_grad[tid] , local_area_diff);
        local_sobel_diff = local_sobel_diff * d_total_gray_value[0];
        atomicAdd(&d_sobel_tri_area_grad[tid] , local_sobel_diff);
   }
}

__global__ void computer_grad_inte( ScalarType* d_compute_tri_area, int* d_faces, ScalarType* d_proj_xy, ScalarType* d_total_gray_value,
    ScalarType* d_real_picture, ScalarType* d_inte_diff, ScalarType* d_inte_result, ScalarType* d_proj_xy_inte_grad, int* d_face_num,
     ScalarType* d_proj_domain, ScalarType* d_pic_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < d_face_num[0])
    {
        int d_pic_size_int[2] = {(int)d_pic_size[0],(int)d_pic_size[1]};

        ScalarType gray_value = d_compute_tri_area[tid] * d_total_gray_value[0];
        ScalarType x_step = (d_proj_domain[1] - d_proj_domain[0])/d_pic_size[0];
        ScalarType y_step = (d_proj_domain[3] - d_proj_domain[2])/d_pic_size[1];
        ScalarType grid_area = x_step * y_step;

        ScalarType inte_gray = 0;

        int point_idx[3] = { d_faces[3*tid], d_faces[3*tid+1], d_faces[3*tid+2]};
        ScalarType tri_coor[6];
        ScalarType small_area_grad[6];
        ScalarType clip_grad[36] = {0};
        ScalarType render_total_grad[6] = {0};

        int poly_num = 3;
        for(int i = 0; i < 3; i++)
        {
            tri_coor[2 * i] = d_proj_xy[2 * point_idx[i]];
            tri_coor[2 * i + 1] = d_proj_xy[2 * point_idx[i] + 1];
        }
        for(int i = 0; i < 6; i++)
        {
            clip_grad[7*i] = 1;
        }

        ScalarType max_x = tri_coor[0], max_y = tri_coor[1], min_x = tri_coor[0], min_y=tri_coor[1];
        for(int i=0;i<3;i++)
        {
            if(tri_coor[2*i]<min_x)
            {
                min_x = tri_coor[2*i];
            }
            if(tri_coor[2*i]>max_x)
            {
                max_x = tri_coor[2*i];
            }
            if(tri_coor[2*i+1]<min_y)
            {
                min_y = tri_coor[2*i+1];
            }
            if(tri_coor[2*i+1]>max_y)
            {
                max_y = tri_coor[2*i+1];
            }
        }

        int low_bound_x  = (int)floor((min_x - d_proj_domain[0]) / x_step);
        int low_bound_y  = (int)floor((min_y - d_proj_domain[2]) / y_step);
        int high_bound_x = (int)floor((max_x - d_proj_domain[0]) / x_step) + 1;
        int high_bound_y = (int)floor((max_y - d_proj_domain[2]) / y_step) + 1;
        
        low_bound_x=(low_bound_x>0)?low_bound_x:0;
        low_bound_y=(low_bound_y>0)?low_bound_y:0;
        high_bound_x=(high_bound_x<d_pic_size_int[0])?high_bound_x:d_pic_size_int[0];
        high_bound_y=(high_bound_y<d_pic_size_int[1])?high_bound_y:d_pic_size_int[1];

        ScalarType clipsquare[4];
        ScalarType new_poly[max_edge_num];
        ScalarType clip_poly_grad[6*max_edge_num];
        ScalarType new_poly_grad[max_edge_num] = {0};
        int new_poly_num;
        
        for(int i_x = low_bound_x; i_x < high_bound_x; i_x++)
        {
            for(int i_y = low_bound_y; i_y < high_bound_y; i_y++)
            {
                int grid = i_y * d_pic_size_int[0] + i_x;
                
                clipsquare[0] = (ScalarType)i_x * x_step + d_proj_domain[0];
                clipsquare[1] = (ScalarType)i_y * y_step + d_proj_domain[2];
                clipsquare[2] = (ScalarType)(i_x + 1) * x_step + d_proj_domain[0];
                clipsquare[3] = (ScalarType)(i_y + 1) * y_step + d_proj_domain[2];
    
                for(int i = 0; i < 6; i++)
                {
                    small_area_grad[i] = 0;
                }
    
                grad_clip_Polygon( tri_coor, clip_grad, poly_num, clipsquare, new_poly, clip_poly_grad, new_poly_num);
    
                for(int i = 0; i < new_poly_num*2; i++)
                {
                    new_poly_grad[i] = 0;
                }
                ScalarType area = compute_grad_area( new_poly, new_poly_grad, new_poly_num);
                accumulate_grad( small_area_grad, clip_poly_grad, new_poly_grad, new_poly_num);

                ScalarType pix_gray = d_real_picture[grid]/grid_area;
                for(int i = 0; i < 6; i++)
                {
                    render_total_grad[i] += 2 * pix_gray * small_area_grad[i];
                }

                inte_gray += area * pix_gray;
            }
        }

        for(int i = 0; i < 6; i++)
        {
            render_total_grad[i] = (inte_gray - gray_value) * render_total_grad[i];
        }
        
        d_inte_diff[tid] = (inte_gray - gray_value);
        d_inte_result[tid] = inte_gray;

        atomicAdd(&d_proj_xy_inte_grad[2*point_idx[0]],render_total_grad[0]);
        atomicAdd(&d_proj_xy_inte_grad[2*point_idx[0]+1],render_total_grad[1]);
        atomicAdd(&d_proj_xy_inte_grad[2*point_idx[1]],render_total_grad[2]);
        atomicAdd(&d_proj_xy_inte_grad[2*point_idx[1]+1],render_total_grad[3]);
        atomicAdd(&d_proj_xy_inte_grad[2*point_idx[2]],render_total_grad[4]);
        atomicAdd(&d_proj_xy_inte_grad[2*point_idx[2]+1],render_total_grad[5]);
    }
}

__global__ void compute_sub( ScalarType* a, ScalarType* b, ScalarType* c, int* d_length)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < d_length[0])
    {
        c[tid] = a[tid] - b[tid];
    }
}

std::vector<torch::Tensor> render( torch::Tensor compute_tri_area, torch::Tensor faces, torch::Tensor proj_xy,
    torch::Tensor total_gray_value, torch::Tensor proj_domain, torch::Tensor pic_size)
{
    cudaSetDevice(compute_tri_area.get_device());

    ScalarType* d_compute_tri_area = compute_tri_area.data_ptr<ScalarType>();
    int* d_faces = faces.data_ptr<int>();
    ScalarType* d_proj_xy = proj_xy.data_ptr<ScalarType>();
    ScalarType* d_total_gray_value = total_gray_value.data_ptr<ScalarType>();
    ScalarType* d_proj_domain = proj_domain.data_ptr<ScalarType>();
    ScalarType* d_pic_size = pic_size.data_ptr<ScalarType>();

    torch::Tensor cpu_pic_size = pic_size.to(torch::kCPU); // Move to CPU
    int H = static_cast<int>(cpu_pic_size[0].item<ScalarType>());
    int W = static_cast<int>(cpu_pic_size[1].item<ScalarType>()); // Convert to int

    ScalarType* d_render_result;
    cudaMalloc(&d_render_result, H * W * sizeof(ScalarType));
    cudaMemset(d_render_result, 0, H * W * sizeof(ScalarType));

    int face_num = compute_tri_area.numel();
    int* d_face_num;
    cudaMalloc(&d_face_num, sizeof(int));
    cudaMemcpy(d_face_num, &face_num, sizeof(int), cudaMemcpyHostToDevice);

    compute_render<<<(face_num + 255) / 256,256>>>( d_compute_tri_area, d_faces, d_proj_xy,
        d_total_gray_value, d_render_result, d_face_num, d_proj_domain, d_pic_size);

    torch::Tensor render_result = torch::from_blob(d_render_result, {W, H}, torch::kFloat).to(torch::kCUDA);
    render_result = render_result.transpose(0, 1);
    render_result = render_result.flip(0); 
    render_result = render_result.clone();

    cudaFree(d_face_num);
    cudaFree(d_render_result);

    return {render_result};
}

std::vector<torch::Tensor> render_grad( torch::Tensor compute_tri_area, torch::Tensor faces, torch::Tensor proj_xy,
    torch::Tensor total_gray_value, torch::Tensor real_picture, torch::Tensor if_use_sobel, torch::Tensor proj_domain, torch::Tensor pic_size)
{
    cudaSetDevice(compute_tri_area.get_device());

    ScalarType* d_compute_tri_area = compute_tri_area.data_ptr<ScalarType>();
    int* d_faces = faces.data_ptr<int>();
    ScalarType* d_proj_xy = proj_xy.data_ptr<ScalarType>();
    ScalarType* d_total_gray_value = total_gray_value.data_ptr<ScalarType>();
    ScalarType* d_real_picture = real_picture.data_ptr<ScalarType>();
    bool h_if_use_sobel = if_use_sobel.to(torch::kCPU).item<bool>();
    ScalarType* d_proj_domain = proj_domain.data_ptr<ScalarType>();
    ScalarType* d_pic_size = pic_size.data_ptr<ScalarType>();

    torch::Tensor cpu_pic_size = pic_size.to(torch::kCPU); // Move to CPU
    int H = static_cast<int>(cpu_pic_size[0].item<ScalarType>());
    int W = static_cast<int>(cpu_pic_size[1].item<ScalarType>()); // Convert to int
    
    int pixel_num = H * W;

    ScalarType* d_render_result;
    cudaMalloc(&d_render_result, pixel_num * sizeof(ScalarType));
    cudaMemset(d_render_result, 0, pixel_num * sizeof(ScalarType));

    ScalarType* d_render_diff;
    cudaMalloc(&d_render_diff, pixel_num * sizeof(ScalarType));
    cudaMemset(d_render_diff, 0, pixel_num * sizeof(ScalarType));

    ScalarType* d_render_grad;
    cudaMalloc(&d_render_grad, pixel_num * sizeof(ScalarType));
    cudaMemset(d_render_grad, 0, pixel_num * sizeof(ScalarType));

    ScalarType* d_sobel_weights;
    cudaMalloc(&d_sobel_weights, pixel_num * sizeof(ScalarType));
    cudaMemset(d_sobel_weights, 0, pixel_num * sizeof(ScalarType));

    int points_num = proj_xy.numel()/2;
    ScalarType* d_proj_xy_render_grad;
    cudaMalloc(&d_proj_xy_render_grad, 2 * points_num * sizeof(ScalarType));
    cudaMemset(d_proj_xy_render_grad, 0, 2 * points_num * sizeof(ScalarType));
    
    ScalarType* d_proj_xy_sobel_grad;
    cudaMalloc(&d_proj_xy_sobel_grad, 2 * points_num * sizeof(ScalarType));
    cudaMemset(d_proj_xy_sobel_grad, 0, 2 * points_num * sizeof(ScalarType));

    ScalarType* d_proj_xy_inte_grad;
    cudaMalloc(&d_proj_xy_inte_grad, 2 * points_num * sizeof(ScalarType));
    cudaMemset(d_proj_xy_inte_grad, 0, 2 * points_num * sizeof(ScalarType));

    int faces_num = compute_tri_area.numel();
    ScalarType* d_render_tri_area_grad;
    cudaMalloc(&d_render_tri_area_grad, faces_num * sizeof(ScalarType));
    cudaMemset(d_render_tri_area_grad, 0, faces_num * sizeof(ScalarType));

    ScalarType* d_sobel_tri_area_grad;
    cudaMalloc(&d_sobel_tri_area_grad, faces_num * sizeof(ScalarType));
    cudaMemset(d_sobel_tri_area_grad, 0, faces_num * sizeof(ScalarType));

    ScalarType* d_inte_diff;
    cudaMalloc(&d_inte_diff, faces_num * sizeof(ScalarType));
    cudaMemset(d_inte_diff, 0, faces_num * sizeof(ScalarType));

    ScalarType* d_inte_result;
    cudaMalloc(&d_inte_result, faces_num * sizeof(ScalarType));
    cudaMemset(d_inte_result, 0, faces_num * sizeof(ScalarType));

    int* d_face_num;
    cudaMalloc(&d_face_num, sizeof(int));
    cudaMemcpy(d_face_num, &faces_num, sizeof(int), cudaMemcpyHostToDevice);

    int* d_pixel_num;
    cudaMalloc(&d_pixel_num, sizeof(int));
    cudaMemcpy(d_pixel_num, &pixel_num, sizeof(int), cudaMemcpyHostToDevice);

    compute_render<<<(faces_num + 255) / 256,256>>>( d_compute_tri_area, d_faces, d_proj_xy,
        d_total_gray_value, d_render_result, d_face_num, d_proj_domain, d_pic_size);
    
    compute_render_diff<<<(pixel_num + 255)/256,256>>>( d_render_result, d_real_picture, d_render_grad, d_pixel_num);

    compute_sub<<<(pixel_num + 255)/256,256>>>( d_render_result, d_real_picture, d_render_diff, d_pixel_num);
    
    computer_grad_inte<<<(faces_num + 255) / 256,256>>>( d_compute_tri_area, d_faces, d_proj_xy, d_total_gray_value,
         d_real_picture, d_inte_diff, d_inte_result, d_proj_xy_inte_grad, d_face_num, d_proj_domain, d_pic_size);

    if(!h_if_use_sobel)
    {
        compute_grad_render<<<(faces_num + 255) / 256,256>>>( d_compute_tri_area, d_render_tri_area_grad, d_faces, d_proj_xy, 
            d_proj_xy_render_grad, d_total_gray_value, d_render_grad, d_face_num, d_proj_domain, d_pic_size);
    }
    else
    {
        compute_sobel_weight<<<(pixel_num + 255)/256,256>>>( d_render_grad, d_sobel_weights, d_pic_size, d_pixel_num);
        compute_grad_render_sobel<<<(faces_num + 255) / 256,256>>>( d_compute_tri_area, d_render_tri_area_grad, d_sobel_tri_area_grad, d_faces,
             d_proj_xy, d_proj_xy_render_grad, d_proj_xy_sobel_grad, d_total_gray_value, d_render_grad, d_sobel_weights, d_face_num,
              d_proj_domain, d_pic_size);
    }

    torch::Tensor render_result = torch::from_blob(d_render_result, {W, H}, torch::kFloat).to(torch::kCUDA);
    render_result = render_result.transpose(0, 1);
    render_result = render_result.flip(0); 
    render_result = render_result.clone();
    torch::Tensor render_diff = torch::from_blob(d_render_diff, {W, H}, torch::kFloat).to(torch::kCUDA);
    render_diff = render_diff.transpose(0, 1);
    render_diff = render_diff.flip(0); 
    render_diff = render_diff.clone();
    torch::Tensor render_proj_xy_grad = torch::from_blob(d_proj_xy_render_grad, {points_num , 2}, torch::kFloat).to(torch::kCUDA);
    render_proj_xy_grad = render_proj_xy_grad.clone();
    torch::Tensor sobel_proj_xy_grad = torch::from_blob(d_proj_xy_sobel_grad, {points_num , 2}, torch::kFloat).to(torch::kCUDA);
    sobel_proj_xy_grad = sobel_proj_xy_grad.clone();
    torch::Tensor render_tri_area_grad = torch::from_blob(d_render_tri_area_grad, {faces_num}, torch::kFloat).to(torch::kCUDA);
    render_tri_area_grad = render_tri_area_grad.clone();
    torch::Tensor sobel_tri_area_grad = torch::from_blob(d_sobel_tri_area_grad, {faces_num}, torch::kFloat).to(torch::kCUDA);
    sobel_tri_area_grad = sobel_tri_area_grad.clone();
    torch::Tensor inte_diff = torch::from_blob(d_inte_diff, {faces_num}, torch::kFloat).to(torch::kCUDA);
    inte_diff = inte_diff.clone();
    torch::Tensor inte_result = torch::from_blob(d_inte_result, {faces_num}, torch::kFloat).to(torch::kCUDA);
    inte_result = inte_result.clone();
    torch::Tensor proj_xy_inte_grad = torch::from_blob(d_proj_xy_inte_grad, {points_num , 2}, torch::kFloat).to(torch::kCUDA);
    proj_xy_inte_grad = proj_xy_inte_grad.clone();

   cudaFree(d_proj_xy_render_grad);
   cudaFree(d_proj_xy_sobel_grad);
   cudaFree(d_render_tri_area_grad);
   cudaFree(d_sobel_tri_area_grad);
   cudaFree(d_render_result);
   cudaFree(d_render_diff);
   cudaFree(d_render_grad);
   cudaFree(d_sobel_weights);
   cudaFree(d_inte_diff);
   cudaFree(d_inte_result);
   cudaFree(d_proj_xy_inte_grad);
   cudaFree(d_face_num);
   cudaFree(d_pixel_num);
   
   return {render_result,render_diff,render_proj_xy_grad,render_tri_area_grad,inte_diff,proj_xy_inte_grad,inte_result,
    sobel_proj_xy_grad,sobel_tri_area_grad};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("render",render);
    m.def("render_grad",render_grad);
}