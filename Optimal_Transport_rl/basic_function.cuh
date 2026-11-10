#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <sstream>
#include <string>
#include <math.h>
#include <omp.h>
// #include <opencv2/opencv.hpp>
#include "clip.cuh"
#include "power_diagram.hpp"

using Vec3 = Eigen::Matrix<ScalarType, 3, 1>;

void writeOBJ(const std::string& filename,const Eigen::VectorXd& polygons,const int* start_num,const int* length,int N)
{
    std::filesystem::path p(filename);
    std::filesystem::create_directories(p.parent_path());

    std::ofstream fout(filename);
    if (!fout.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return;
    }

    int global_idx = 1; // OBJ index 从1开始

    for (int i = 0; i < N; ++i)
    {
        int start = start_num[i];
        int len   = length[i];

        if (len < 3) continue; // 非法面

        // 记录当前polygon的顶点索引
        std::vector<int> face_indices;

        for (int j = 0; j < len; ++j)
        {
            double x = polygons(2 * (start + j));
            double y = polygons(2 * (start + j) + 1);

            fout << "v " << x << " " << y << " 0.0\n";

            face_indices.push_back(global_idx);
            global_idx++;
        }

        // 写 face
        fout << "f ";
        for (int idx : face_indices)
        {
            fout << idx << " ";
        }
        fout << "\n";
    }

    fout.close();
}

void load_triangles(const string& path, Eigen::VectorXi& tri_mesh)
{
    ifstream fin(path);
    if (!fin.is_open()) {
        tri_mesh.resize(0);
        return;
    }

    vector<int> tri_vec;
    string line;

    while (getline(fin, line)) {
        if (line.size() < 2) continue;
        if (line[0] != 'f' || line[1] != ' ') continue;

        stringstream ss(line);
        char c;
        int a, b, c_idx;

        ss >> c >> a >> b >> c_idx;

        tri_vec.push_back(a - 1);  // obj转0-based
        tri_vec.push_back(b - 1);
        tri_vec.push_back(c_idx - 1);
    }

    fin.close();

    // 拷贝到 Eigen::VectorXi
    tri_mesh.resize(tri_vec.size());
    for (int i = 0; i < (int)tri_vec.size(); i++) {
        tri_mesh[i] = tri_vec[i];
    }
}

bool readPointsFlat(const std::string& file_path, Eigen::VectorXd& vec)
{
    std::ifstream fin(file_path);
    if (!fin.is_open()) {
        std::cerr << "无法打开文件: " << file_path << std::endl;
        return false;
    }

    std::vector<double> data;
    double x, y;

    while (fin >> x >> y) {
        data.push_back(x);
        data.push_back(y);
    }

    vec.resize(data.size());
    for (int i = 0; i < data.size(); ++i)
        vec(i) = data[i];

    return true;
}

bool readIntVector(const std::string& file_path,
                   Eigen::VectorXi& vec)
{
    std::ifstream fin(file_path);
    if (!fin.is_open()) {
        std::cerr << "无法打开文件: " << file_path << std::endl;
        return false;
    }

    std::vector<int> data;
    int val;

    while (fin >> val) {
        data.push_back(val);
    }

    fin.close();

    vec.resize(data.size());
    for (int i = 0; i < data.size(); ++i) {
        vec(i) = data[i];
    }

    return true;
}

bool read_pic_txt_to_array(const std::string& file_path, ScalarType*& data, int*& pic_size)
{
    std::ifstream fin(file_path);
    if (!fin.is_open()) {
        std::cerr << "无法打开文件: " << file_path << std::endl;
        return false;
    }

    std::vector<std::vector<ScalarType>> matrix;
    std::string line;

    while (std::getline(fin, line)) {
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::vector<ScalarType> row;
        ScalarType value;

        while (iss >> value) {
            row.push_back(value);
        }

        if (!row.empty()) {
            if (!matrix.empty() && row.size() != matrix[0].size()) {
                std::cerr << "每一行列数不一致，文件格式错误。" << std::endl;
                return false;
            }
            matrix.push_back(row);
        }
    }

    fin.close();

    if (matrix.empty()) {
        std::cerr << "文件为空或没有有效数据。" << std::endl;
        return false;
    }

    int m = static_cast<int>(matrix.size());      // 行数
    int n = static_cast<int>(matrix[0].size());   // 列数

    // 保存行列数
    pic_size = new int[2];
    pic_size[0] = m;
    pic_size[1] = n;

    // 分配一维数组，大小为 m*n
    data = new ScalarType[m * n];

    // 按你的坐标定义存：
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < m; ++x) {
            int row = m - 1 - x;   // txt中从上到下的行号 -> 你定义的x
            int col = y;           // txt中的列号 -> 你定义的y
            data[y * m + x] = matrix[row][col];
        }
    }

    return true;
}
bool save_array_to_txt(const std::string& file_path,
                       const ScalarType* data,
                       const int* picture_size)
{
    if (data == nullptr) {
        std::cerr << "data 为空。" << std::endl;
        return false;
    }

    if (picture_size == nullptr) {
        std::cerr << "picture_size 为空。" << std::endl;
        return false;
    }

    int m = picture_size[0]; // 行数
    int n = picture_size[1]; // 列数

    std::filesystem::path p(file_path);
    std::filesystem::create_directories(p.parent_path());
    std::ofstream fout(file_path);
    if (!fout.is_open()) {
        std::cerr << "无法打开文件: " << file_path << std::endl;
        return false;
    }

    fout << std::fixed << std::setprecision(6);

    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            int x = m - 1 - row;   // 你的数组里 x 从下往上
            int y = col;           // y 从左往右

            fout << data[y * m + x];

            if (col != n - 1) {
                fout << " ";
            }
        }
        fout << "\n";
    }

    fout.close();
    return true;
}

void build_one_ring_neighbors( const Eigen::VectorXi& tri_mesh, int points_num, std::vector<std::vector<int>>& neighbors)
{
    neighbors.clear();
    neighbors.resize(points_num);

    int tri_num = tri_mesh.size() / 3;

    for (int t = 0; t < tri_num; t++)
    {
        int a = tri_mesh[3 * t + 0];
        int b = tri_mesh[3 * t + 1];
        int c = tri_mesh[3 * t + 2];

        neighbors[a].push_back(b);
        neighbors[a].push_back(c);

        neighbors[b].push_back(a);
        neighbors[b].push_back(c);

        neighbors[c].push_back(a);
        neighbors[c].push_back(b);
    }

    // 去重
    for (int i = 0; i < points_num; i++)
    {
        std::vector<int> unique_neighbors;
        for (int j = 0; j < (int)neighbors[i].size(); j++)
        {
            int v = neighbors[i][j];
            bool exist = false;
            for (int k = 0; k < (int)unique_neighbors.size(); k++)
            {
                if (unique_neighbors[k] == v)
                {
                    exist = true;
                    break;
                }
            }
            if (!exist)
            {
                unique_neighbors.push_back(v);
            }
        }
        neighbors[i] = unique_neighbors;
    }
}

void repair_degenerate_weights( Eigen::VectorXd& weights, const int* length, const Eigen::VectorXi& tri_mesh, int points_num, ScalarType repair_para)
{
    std::vector<std::vector<int>> neighbors;
    build_one_ring_neighbors(tri_mesh, points_num, neighbors);

    Eigen::VectorXi state(points_num);
    state.setZero();

    for (int i = 0; i < points_num; i++)
    {
        if (length[i] == 0)
        {
            state[i] = 0;
        }
        else
        {
            state[i] = 1;
        }
    }

    // 第一轮：只看没退化点，赋值 = 平均值 - 200
    for (int i = 0; i < points_num; i++)
    {
        if (state[i] != 0) continue;

        double sum_w = 0.0;
        int cnt = 0;

        cout<<i<<" "<<neighbors[i].size()<<endl;

        for (int j = 0; j < (int)neighbors[i].size(); j++)
        {
            int nb = neighbors[i][j];
            if (state[nb] == 1)   // 只用没退化点
            {
                sum_w += weights[nb];
                cnt++;
            }
        }

        if (cnt > 0)
        {
            weights[i] = sum_w / cnt - repair_para;
            state[i] = 2;
        }
    }

    // 后续迭代：未处理退化点，使用“没退化点 + 已处理点”的平均值
    bool changed = true;
    while (changed)
    {
        changed = false;

        std::vector<int> update_idx;
        std::vector<double> update_val;

        for (int i = 0; i < points_num; i++)
        {
            if (state[i] != 0) continue;  // 只处理还没处理的退化点

            double sum_w = 0.0;
            int cnt = 0;

            for (int j = 0; j < (int)neighbors[i].size(); j++)
            {
                int nb = neighbors[i][j];
                if (state[nb] == 1 || state[nb] == 2)
                {
                    sum_w += weights[nb];
                    cnt++;
                }
            }

            if (cnt > 0)
            {
                update_idx.push_back(i);
                update_val.push_back(sum_w / cnt);
            }
        }

        for (int k = 0; k < (int)update_idx.size(); k++)
        {
            int i = update_idx[k];
            weights[i] = update_val[k];
            state[i] = 2;
            changed = true;
        }
    }

    // 检查是否还有没处理掉的点
    int unresolved = 0;
    for (int i = 0; i < points_num; i++)
    {
        if (state[i] == 0)
        {
            unresolved++;
        }
    }

    std::cout << "degenerate repair finished, unresolved points = " << unresolved << std::endl;
}

inline Vec3 normalize3(const Vec3& v, ScalarType eps = 1e-12)
{
    ScalarType n = v.norm();
    if (n < eps) return Vec3::Zero();
    return v / n;
}

ScalarType fresnel_T_unpolarized(
    const Vec3& wi,
    const Vec3& wt,
    const Vec3& normal,
    ScalarType eta_i,
    ScalarType eta_t,
    ScalarType eps = 1e-12)
{
    ScalarType cos_i = std::abs(wi.dot(normal));
    ScalarType cos_t = std::abs(wt.dot(normal));

    cos_i = std::min((ScalarType)1.0, std::max((ScalarType)0.0, cos_i));
    cos_t = std::min((ScalarType)1.0, std::max((ScalarType)0.0, cos_t));

    ScalarType denom_s = eta_i * cos_i + eta_t * cos_t;
    ScalarType denom_p = eta_i * cos_t + eta_t * cos_i;

    if (std::abs(denom_s) < eps) denom_s = eps;
    if (std::abs(denom_p) < eps) denom_p = eps;

    ScalarType Rs = (eta_i * cos_i - eta_t * cos_t) / denom_s;
    ScalarType Rp = (eta_i * cos_t - eta_t * cos_i) / denom_p;

    Rs *= Rs;
    Rp *= Rp;

    ScalarType R = 0.5 * (Rs + Rp);
    return 1.0 - R;
}

Eigen::VectorXd compute_energy_remain_rate(
    const Eigen::VectorXd& points,          // 2*n
    ScalarType z_of_len,
    const Eigen::VectorXd& input_light,     // 3*n
    const Eigen::VectorXd& inside_light,    // 3*n
    const Eigen::VectorXd& height,
    const Eigen::VectorXd& output_points,   // 2*n
    ScalarType z_of_receiver,
    ScalarType refraction_index)
{
    int n = points.size() / 2;
    Eigen::VectorXd remain_ratio(n);

    Vec3 n1(0, 0, 1);  // 前表面法向固定

    for (int i = 0; i < n; i++)
    {
        // --- 点 ---
        Vec3 p(points[2*i], points[2*i+1], z_of_len);

        // --- 光线 ---
        Vec3 wi(
            input_light[3*i],
            input_light[3*i+1],
            input_light[3*i+2]
        );

        Vec3 ins(
            inside_light[3*i],
            inside_light[3*i+1],
            inside_light[3*i+2]
        );

        // --- 后表面点 ---
        Vec3 out_p = p + height[i] * ins;

        // --- 目标点 ---
        Vec3 receiver(
            output_points[2*i],
            output_points[2*i+1],
            z_of_receiver
        );

        // --- 出射方向 ---
        Vec3 wt = normalize3(receiver - out_p);

        // --- 后表面法向 ---
        Vec3 n2 = normalize3(refraction_index * ins - wt);

        // --- Fresnel ---
        ScalarType T1 = fresnel_T_unpolarized(wi, ins, n1, 1.0, refraction_index);
        ScalarType T2 = fresnel_T_unpolarized(ins, wt, n2, refraction_index, 1.0);

        remain_ratio[i] = T1 * T2;
    }

    return remain_ratio;
}

#if !defined (__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ ScalarType atomicAdd(ScalarType* address, ScalarType val)
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

__device__ ScalarType compute_norm(ScalarType* vector)
{
    ScalarType result = 0;
    for(int i=0; i<3; i++)
    {
        result += vector[i] * vector[i];
    }
    return sqrt(result);
}

__device__ ScalarType compute_area(ScalarType* new_poly, int new_poly_num)
{
    ScalarType area=0;
    for(int i=1;i<new_poly_num-1;i++)
    {
        area+=0.5*((new_poly[2*i+1]-new_poly[1])*(new_poly[2*i+2]-new_poly[0])-(new_poly[2*i]-new_poly[0])*(new_poly[2*i+3]-new_poly[1]));
    }
    if(area<0)
    {
        area=-1*area;
    }
    return area;
}

__device__ ScalarType compute_dot( ScalarType* vector1, ScalarType* vector2)
{
    ScalarType result = 0;
    for(int i = 0; i < 3; i++)
    {
        result += vector1[i] * vector2[i];
    }
    return result;
}

__device__ void compute_cross( ScalarType* vector1, ScalarType* vector2, ScalarType* cross_result)
{
    cross_result[0] = vector1[1] * vector2[2] - vector1[2] * vector2[1];
    cross_result[1] = vector1[2] * vector2[0] - vector1[0] * vector2[2];
    cross_result[2] = vector1[0] * vector2[1] - vector1[1] * vector2[0];
}

__device__ ScalarType compute_solid_angle(ScalarType* new_poly, int new_poly_num, ScalarType* z, 
                                 ScalarType* d_light_coord) 
{
    ScalarType angle = 0;
    for(int i = 0; i < new_poly_num - 2; i++) {
        ScalarType vector1[3] = {new_poly[0] - d_light_coord[0], new_poly[1] - d_light_coord[1], z[0] - d_light_coord[2]};
        ScalarType vector2[3] = {new_poly[2*(i+1)] - d_light_coord[0], new_poly[2*(i+1)+1] - d_light_coord[1], z[0] - d_light_coord[2]};
        ScalarType vector3[3] = {new_poly[2*(i+2)] - d_light_coord[0], new_poly[2*(i+2)+1] - d_light_coord[1], z[0] - d_light_coord[2]};

        ScalarType length_1 = compute_norm(vector1);
        ScalarType length_2 = compute_norm(vector2);
        ScalarType length_3 = compute_norm(vector3);

        vector1[0] /= length_1; vector1[1] /= length_1; vector1[2] /= length_1;
        vector2[0] /= length_2; vector2[1] /= length_2; vector2[2] /= length_2;
        vector3[0] /= length_3; vector3[1] /= length_3; vector3[2] /= length_3;

        ScalarType cross_result[3];
        compute_cross(vector2, vector3, cross_result);
        ScalarType numerator = compute_dot(vector1, cross_result);

        ScalarType dot12 = compute_dot(vector1, vector2);
        ScalarType dot23 = compute_dot(vector2, vector3);
        ScalarType dot13 = compute_dot(vector1, vector3);
        ScalarType denominator = 1.0 + dot12 + dot23 + dot13;

        if (fabs(denominator) < 1e-6) continue;
        angle += 2 * atan(numerator/denominator);
    }
    return fabs(angle);
}

__device__ void d_compute_center(ScalarType* new_poly, int new_poly_num, ScalarType* poly_center)
{
    poly_center[0] = 0;
    poly_center[1] = 0;
    ScalarType big_area = compute_area(new_poly, new_poly_num);
    for(int i=0;i<new_poly_num - 2;i++)
    {
        ScalarType tri[6] = {new_poly[0],new_poly[1],new_poly[2*i+2],new_poly[2*i+3],new_poly[2*i+4],new_poly[2*i+5]};
        ScalarType tri_center[2];
        ScalarType area = compute_area(tri, 3);
        tri_center[0] = (tri[0] + tri[2] + tri[4]) / 3;
        tri_center[1] = (tri[1] + tri[3] + tri[5]) / 3;
        poly_center[0] += tri_center[0] * area;
        poly_center[1] += tri_center[1] * area;
    }
    poly_center[0]/=big_area;
    poly_center[1]/=big_area;
    
}

__global__ void compute_clip_area(ScalarType* d_polygons,ScalarType* d_polygons_inte,int* d_start_num,int* d_length, int* d_points_num, ScalarType* d_simple_domain)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid <d_points_num[0])//
    {
        if(d_length[tid]>1)
        {
            int length_num = d_length[tid];
            ScalarType* polygon = d_polygons+2*d_start_num[tid];

            ScalarType square[4] = {d_simple_domain[0],d_simple_domain[2],d_simple_domain[1],d_simple_domain[3]};
            ScalarType new_poly[max_edge_num];
            int new_poly_num = 0;
            clip_Polygon( polygon, length_num, square, new_poly, new_poly_num);
            ScalarType area = compute_area( new_poly, new_poly_num);
            d_polygons_inte[tid] = area;

        }
        else
        {
            d_polygons_inte[tid] = 0;
        }
    }
}

__global__ void solid_angle(ScalarType* d_polygons, int* d_start_num, int* d_length, ScalarType* d_polygons_solid_angle, int* d_points_num,
     ScalarType* d_simple_domain, ScalarType* d_z, ScalarType* d_light_coord)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < d_points_num[0])
    {
        ScalarType*polygon = d_polygons+2*d_start_num[tid];
        int length_num = d_length[tid];
        ScalarType inside_poly[max_edge_num];
        int inside_poly_num;
        ScalarType square[4] = {d_simple_domain[0],d_simple_domain[2],d_simple_domain[1],d_simple_domain[3]};
        clip_Polygon( polygon, length_num, square, inside_poly, inside_poly_num);
        ScalarType solid_angle = compute_solid_angle( inside_poly, inside_poly_num, d_z, d_light_coord);
        d_polygons_solid_angle[tid] = solid_angle;
    }
}

__global__ void len_solid_angle( ScalarType* d_simple_domain, ScalarType* d_total_solid_angle, ScalarType* d_z, ScalarType* d_light_coord)
{
    ScalarType final_square[8] = {d_simple_domain[0],d_simple_domain[2],d_simple_domain[0],d_simple_domain[3],
                                    d_simple_domain[1],d_simple_domain[3],d_simple_domain[1],d_simple_domain[2]};
    int square_poly_num = 4;
    d_total_solid_angle[0] = compute_solid_angle( final_square, square_poly_num, d_z, d_light_coord);
}

__global__ void render(ScalarType* d_polygons,ScalarType* d_clip_area,int* d_start_num,int* d_length,
     ScalarType* d_render_result, int* d_points_num, ScalarType* d_proj_domain, int* d_picture_size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < d_points_num[0])//
    {
        if(d_length[tid]!=0)
        {
            ScalarType*polygon = d_polygons+2*d_start_num[tid];
            int length_num = d_length[tid];
            ScalarType inside_poly[max_edge_num];
            int inside_poly_num;
            ScalarType new_poly[max_edge_num];
            int new_poly_num;
            ScalarType x_step = (d_proj_domain[1] - d_proj_domain[0])/(ScalarType)d_picture_size[0];
            ScalarType y_step = (d_proj_domain[3] - d_proj_domain[2])/(ScalarType)d_picture_size[1];

            ScalarType square[4] = {d_proj_domain[0],d_proj_domain[2],d_proj_domain[1],d_proj_domain[3]};//xmin ymin xmax ymax
            clip_Polygon( polygon, length_num, square, inside_poly, inside_poly_num);
            ScalarType big_area = compute_area( inside_poly, inside_poly_num);
            ScalarType clipsquare[4];
            
            if(inside_poly_num>=3)
            {
                ScalarType max_x = inside_poly[0], max_y = inside_poly[1],min_x = inside_poly[0], min_y = inside_poly[1];

                for(int i=0;i<inside_poly_num;i++)
                {
                    if(inside_poly[2*i]<min_x)
                    {
                        min_x = inside_poly[2*i];
                    }
                    if(inside_poly[2*i]>max_x)
                    {
                        max_x = inside_poly[2*i];
                    }
                    if(inside_poly[2*i+1]<min_y)
                    {
                        min_y = inside_poly[2*i+1];
                    }
                    if(inside_poly[2*i+1]>max_y)
                    {
                        max_y = inside_poly[2*i+1];
                    }
                }

                int low_bound_x  = (int)floor((min_x - d_proj_domain[0]) / x_step);
                int low_bound_y  = (int)floor((min_y - d_proj_domain[2]) / y_step);
                int high_bound_x = (int)floor((max_x - d_proj_domain[0]) / x_step) + 1;
                int high_bound_y = (int)floor((max_y - d_proj_domain[2]) / y_step) + 1;
                
                low_bound_x=(low_bound_x>0)?low_bound_x:0;
                low_bound_y=(low_bound_y>0)?low_bound_y:0;
                high_bound_x=(high_bound_x<d_picture_size[0])?high_bound_x:d_picture_size[0];
                high_bound_y=(high_bound_y<d_picture_size[1])?high_bound_y:d_picture_size[1];

                for(int i_x = low_bound_x; i_x < high_bound_x; i_x++)
                {
                    for(int i_y = low_bound_y; i_y < high_bound_y; i_y++)
                    {
                        int grid = i_x + i_y * d_picture_size[0]; 
                        clipsquare[0] = (ScalarType)i_x * x_step + d_proj_domain[0];
                        clipsquare[1] = (ScalarType)i_y * y_step + d_proj_domain[2];
                        clipsquare[2] = (ScalarType)(i_x + 1) * x_step + d_proj_domain[0];
                        clipsquare[3] = (ScalarType)(i_y + 1) * y_step + d_proj_domain[2];

                        clip_Polygon( inside_poly, inside_poly_num, clipsquare, new_poly, new_poly_num);
                        ScalarType area = compute_area( new_poly, new_poly_num);

                        if(big_area!=0)
                        {

                            ScalarType local_contribution = d_clip_area[tid]*area/big_area;
                            atomicAdd(&d_render_result[grid],local_contribution);
                        }  
                    }
                }
            }
        }
    }
    return;
}

__global__ void compute_inte(ScalarType* d_polygons, ScalarType* d_polygons_inte, int* d_start_num, int* d_length,
     ScalarType* d_picture, int* d_points_num, ScalarType* d_proj_domain, int* d_picture_size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < d_points_num[0])// == 1217
    {
        if(d_length[tid]!=0)
        {
            d_polygons_inte[tid] = 0;
            ScalarType*polygon = d_polygons+2*d_start_num[tid];
            int length_num = d_length[tid];
            ScalarType inside_poly[max_edge_num];
            int inside_poly_num;
            ScalarType new_poly[max_edge_num];
            int new_poly_num;
            ScalarType x_step = (d_proj_domain[1] - d_proj_domain[0])/(ScalarType)d_picture_size[0];
            ScalarType y_step = (d_proj_domain[3] - d_proj_domain[2])/(ScalarType)d_picture_size[1];
            ScalarType grid_area = x_step*y_step;

            ScalarType square[4] = {d_proj_domain[0],d_proj_domain[2],d_proj_domain[1],d_proj_domain[3]};//xmin ymin xmax ymax
            clip_Polygon( polygon, length_num, square, inside_poly, inside_poly_num);
            ScalarType big_area = compute_area( inside_poly, inside_poly_num);
            ScalarType clipsquare[4];

            if(inside_poly_num>=3)
            {
                ScalarType max_x = inside_poly[0], max_y = inside_poly[1],min_x = inside_poly[0], min_y = inside_poly[1];

                for(int i=0;i<inside_poly_num;i++)
                {
                    if(inside_poly[2*i]<min_x)
                    {
                        min_x = inside_poly[2*i];
                    }
                    if(inside_poly[2*i]>max_x)
                    {
                        max_x = inside_poly[2*i];
                    }
                    if(inside_poly[2*i+1]<min_y)
                    {
                        min_y = inside_poly[2*i+1];
                    }
                    if(inside_poly[2*i+1]>max_y)
                    {
                        max_y = inside_poly[2*i+1];
                    }
                }

                int low_bound_x  = (int)floor((min_x - d_proj_domain[0]) / x_step);
                int low_bound_y  = (int)floor((min_y - d_proj_domain[2]) / y_step);
                int high_bound_x = (int)floor((max_x - d_proj_domain[0]) / x_step) + 1;
                int high_bound_y = (int)floor((max_y - d_proj_domain[2]) / y_step) + 1;
                
                low_bound_x=(low_bound_x>0)?low_bound_x:0;
                low_bound_y=(low_bound_y>0)?low_bound_y:0;
                high_bound_x=(high_bound_x<d_picture_size[0])?high_bound_x:d_picture_size[0];
                high_bound_y=(high_bound_y<d_picture_size[1])?high_bound_y:d_picture_size[1];

                for(int i_x = low_bound_x; i_x < high_bound_x; i_x++)
                {
                    for(int i_y = low_bound_y; i_y < high_bound_y; i_y++)
                    {
                        int grid = i_x + i_y * d_picture_size[0]; 
                        clipsquare[0] = (ScalarType)i_x * x_step + d_proj_domain[0];
                        clipsquare[1] = (ScalarType)i_y * y_step + d_proj_domain[2];
                        clipsquare[2] = (ScalarType)(i_x + 1) * x_step + d_proj_domain[0];
                        clipsquare[3] = (ScalarType)(i_y + 1) * y_step + d_proj_domain[2];

                        clip_Polygon( inside_poly, inside_poly_num, clipsquare, new_poly, new_poly_num);
                        ScalarType area = compute_area( new_poly, new_poly_num);
                        ScalarType local_contribution = (d_picture[grid]) * area/grid_area;
                        d_polygons_inte[tid] += local_contribution;     
                    }
                }
            }
        }
        else
        {
            d_polygons_inte[tid] = 0;
        }
    }
}

__device__ void Edge_ints(ScalarType* p1_origin, ScalarType* p2_origin, ScalarType* s_origin, ScalarType& sq_dist)
{
    ScalarType p1[2] = {p1_origin[0],p1_origin[1]};
    ScalarType p2[2] = {p2_origin[0],p2_origin[1]};
    ScalarType s[2] = {s_origin[0],s_origin[1]};
    ScalarType x2_M_x1 = p2[0] - p1[0];
    ScalarType y1_M_y2 = p1[1] - p2[1];
    ScalarType x1_P_x2 = p1[0] + p2[0];
    ScalarType y1_P_y2 = p1[1] + p2[1];
    ScalarType x1_2_P_x2_2 = p1[0] * p1[0] + p2[0] * p2[0];
    ScalarType y1_2_P_y2_2 = p1[1] * p1[1] + p2[1] * p2[1];
    ScalarType x1_2_P_x2_2_P_x1x2 = x1_2_P_x2_2 + p1[0] * p2[0];
    ScalarType y1_2_P_y2_2_P_y1y2 = y1_2_P_y2_2 + p1[1] * p2[1];
    ScalarType x1y2_M_x2y1 = p1[0] * p2[1] - p2[0] * p1[1];
    ScalarType x0_2 = s[0] * s[0];
    ScalarType y0_2 = s[1] * s[1];

    ScalarType int_Q = x2_M_x1 * y1_P_y2 * y1_2_P_y2_2 / 4. 
        + (x1y2_M_x2y1 - 2. * s[1] * x2_M_x1) * y1_2_P_y2_2_P_y1y2 / 3. 
            + (y0_2 * x2_M_x1 / 2. - s[1] * x1y2_M_x2y1) * y1_P_y2 
                + y0_2 * x1y2_M_x2y1;
    
    ScalarType int_P = y1_M_y2 * x1_P_x2 * x1_2_P_x2_2 / 4. 
        + (x1y2_M_x2y1 - 2. * s[0] * y1_M_y2) * x1_2_P_x2_2_P_x1x2 / 3. 
            + (x0_2 * y1_M_y2 / 2. - s[0] * x1y2_M_x2y1) * x1_P_x2 
                + x0_2 * x1y2_M_x2y1;
    
    sq_dist += int_Q + int_P;
}

__device__ void Polygon_Ints(ScalarType* polygon, int& length, ScalarType* s, ScalarType& sq_dist)
{
    // calculate the Integrals of the polygon
    sq_dist = 0;
    Edge_ints(polygon + (2 * length - 2), polygon, s, sq_dist);
    for(int i = 0; i < (length - 1); i++)
    {
        Edge_ints(polygon + 2 * i, polygon + 2 * i + 2, s, sq_dist);
    }
}

__global__ void compute_f( ScalarType* d_points, ScalarType* d_weights, ScalarType* d_polygons_clip_area, ScalarType* d_polygons_f,
    ScalarType* d_polygons, int* d_start_num,int* d_length, ScalarType* d_picture, int* d_points_num , ScalarType* d_proj_domain, int* d_picture_size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid <d_points_num[0])//
    {
        if(d_length[tid]!=0)
        {
            d_polygons_f[tid] = -d_weights[tid] * d_polygons_clip_area[tid];
            ScalarType sq_dist;
            ScalarType*polygon = d_polygons+2*d_start_num[tid];
            int length_num = d_length[tid];
            ScalarType inside_poly[max_edge_num];
            int inside_poly_num;
            ScalarType new_poly[max_edge_num];
            int new_poly_num;
            ScalarType x_step = (d_proj_domain[1] - d_proj_domain[0])/(ScalarType)d_picture_size[0];
            ScalarType y_step = (d_proj_domain[3] - d_proj_domain[2])/(ScalarType)d_picture_size[1];
            ScalarType grid_area = x_step*y_step;

            ScalarType square[4] = {d_proj_domain[0],d_proj_domain[2],d_proj_domain[1],d_proj_domain[3]};//xmin ymin xmax ymax
            clip_Polygon( polygon, length_num, square, inside_poly, inside_poly_num);
            ScalarType clipsquare[4];
            
            if(inside_poly_num>=3)
            {
                ScalarType max_x = inside_poly[0], max_y = inside_poly[1],min_x = inside_poly[0], min_y = inside_poly[1];

                for(int i=0;i<inside_poly_num;i++)
                {
                    if(inside_poly[2*i]<min_x)
                    {
                        min_x = inside_poly[2*i];
                    }
                    if(inside_poly[2*i]>max_x)
                    {
                        max_x = inside_poly[2*i];
                    }
                    if(inside_poly[2*i+1]<min_y)
                    {
                        min_y = inside_poly[2*i+1];
                    }
                    if(inside_poly[2*i+1]>max_y)
                    {
                        max_y = inside_poly[2*i+1];
                    }
                }

                int low_bound_x  = (int)floor((min_x - d_proj_domain[0]) / x_step);
                int low_bound_y  = (int)floor((min_y - d_proj_domain[2]) / y_step);
                int high_bound_x = (int)floor((max_x - d_proj_domain[0]) / x_step) + 1;
                int high_bound_y = (int)floor((max_y - d_proj_domain[2]) / y_step) + 1;
                
                low_bound_x=(low_bound_x>0)?low_bound_x:0;
                low_bound_y=(low_bound_y>0)?low_bound_y:0;
                high_bound_x=(high_bound_x<d_picture_size[0])?high_bound_x:d_picture_size[0];
                high_bound_y=(high_bound_y<d_picture_size[1])?high_bound_y:d_picture_size[1];

                for(int i_x = low_bound_x; i_x < high_bound_x; i_x++)
                {
                    for(int i_y = low_bound_y; i_y < high_bound_y; i_y++)
                    {
                        int grid = i_x + i_y * d_picture_size[0]; 
                        clipsquare[0] = (ScalarType)i_x * x_step + d_proj_domain[0];
                        clipsquare[1] = (ScalarType)i_y * y_step + d_proj_domain[2];
                        clipsquare[2] = (ScalarType)(i_x + 1) * x_step + d_proj_domain[0];
                        clipsquare[3] = (ScalarType)(i_y + 1) * y_step + d_proj_domain[2];

                        clip_Polygon( inside_poly, inside_poly_num, clipsquare, new_poly, new_poly_num);
                        ScalarType area = compute_area( new_poly, new_poly_num);
                        if(area>0)
                        {
                            Polygon_Ints( new_poly, new_poly_num, d_points + 2 * tid, sq_dist);
                            ScalarType local_contribution = d_picture[grid]/grid_area * (sq_dist - d_weights[tid] * area);
                            d_polygons_f[tid] -= local_contribution;
                        }
                    }
                }
            }
        }
        else
        {
            d_polygons_f[tid] = -d_weights[tid] * d_polygons_clip_area[tid];
        }
    }
}

__global__ void storage(ScalarType* a, ScalarType* b, int* d_points_num)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid <d_points_num[0])//
    {
        a[tid] = b[tid];
    }
}

__global__ void compute_weighted_sum( ScalarType* d_result, ScalarType* d_history, ScalarType* d_alpha, int* history_size, int* d_points_num)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < d_points_num[0])
    {
        ScalarType sum = 0;

        for (int i = 0; i < history_size[0]; ++i)
        {
            sum += d_alpha[i] * d_history[i * d_points_num[0] + tid];
        }

        d_result[tid] = sum;
    }
}

__global__ void compute_add(ScalarType* a, ScalarType* b, ScalarType* result, int* d_points_num)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid <d_points_num[0])//
    {
        result[tid] = a[tid] + b[tid];
    }
}

__global__ void compute_subtract(ScalarType* a, ScalarType* b, ScalarType* result, int* d_points_num)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid <d_points_num[0])//
    {
        result[tid] = a[tid] - b[tid];
    }
}

__global__ void compute_xpay( ScalarType* x, ScalarType* a, ScalarType* y, int* d_points_num)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid <d_points_num[0])//
    {
        x[tid] = x[tid] + a[0]*y[tid];
    }
}

__global__ void compute_divide(ScalarType* d_result, ScalarType* d_data, ScalarType* para, int* d_points_num)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid <d_points_num[0])//
    {
        d_result[tid] = d_data[tid] / para[0];
    }
}

__global__ void compute_multi(ScalarType* d_result, ScalarType* d_data, ScalarType* para, int* d_points_num)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid <d_points_num[0])//
    {
        d_result[tid] = d_data[tid] * para[0];
    }
}

__global__ void compute_dot( ScalarType* d_data_1, ScalarType* d_data_2, ScalarType* d_result, int* d_points_num)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid <d_points_num[0])//
    {
        d_result[tid] = d_data_1[tid] * d_data_2[tid];
    }
}

__global__ void compute_fixed_function( ScalarType* d_weight_history, ScalarType* d_gk_history, ScalarType* d_fixed_para,
    ScalarType* d_fixed_function, int* d_points_num)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid <d_points_num[0])//
    {
        d_fixed_function[tid] = d_weight_history[tid] + d_fixed_para[0] * d_gk_history[tid];
    }
}

__global__ void compute_adam_direc(ScalarType* g_k, ScalarType* m_k, ScalarType* v_k, ScalarType* direc,
     ScalarType* d_beta_1, ScalarType* d_beta_1t, ScalarType* d_beta_2, ScalarType* d_beta_2t, ScalarType* d_epsilon, int* d_points_num)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid <d_points_num[0])//== 127
    {
        m_k[tid] = m_k[tid] * d_beta_1[0] + g_k[tid] * (1 - d_beta_1[0]);
        v_k[tid] = v_k[tid] * d_beta_2[0] + g_k[tid] * g_k[tid] * (1 - d_beta_2[0]);
        ScalarType m_hat = m_k[tid] / (1 - d_beta_1t[0]);
        ScalarType v_hat = v_k[tid] / (1 - d_beta_2t[0]);
        direc[tid] = - m_hat / (sqrt(v_hat) + d_epsilon[0]);
    }
}
__global__ void compute_center(ScalarType* d_polygons,ScalarType* d_polygons_center,int* d_start_num,int* d_length, ScalarType* d_picture,
     int* d_points_num, bool* if_weighting, bool* d_if_exist, ScalarType* d_proj_domain, int* d_picture_size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid <d_points_num[0])//== 127
    {
        if(d_length[tid]!=0)
        {
            if(if_weighting[0] == false)
            {
                ScalarType*polygon = d_polygons+2*d_start_num[tid];
                int length_num = d_length[tid];
                ScalarType inside_poly[max_edge_num];
                int inside_poly_num;

                ScalarType square[4] = {d_proj_domain[0],d_proj_domain[2],d_proj_domain[1],d_proj_domain[3]};

                clip_Polygon( polygon, length_num, square, inside_poly, inside_poly_num);
                if(inside_poly_num >= 3)
                {
                    d_compute_center( inside_poly, inside_poly_num, d_polygons_center+2*tid);
                    d_if_exist[tid] = true;
                }
                else
                {
                    d_if_exist[tid] = false;
                }                
            }
            else
            {
                ScalarType new_center_x = 0;
                ScalarType new_center_y = 0;

                ScalarType*polygon = d_polygons+2*d_start_num[tid];
                int length_num = d_length[tid];
                ScalarType inside_poly[max_edge_num];
                int inside_poly_num;
                ScalarType new_poly[max_edge_num];
                int new_poly_num;
                
                ScalarType square[4] = {d_proj_domain[0],d_proj_domain[2],d_proj_domain[1],d_proj_domain[3]};
                clip_Polygon( polygon, length_num, square, inside_poly, inside_poly_num);
                ScalarType big_area = compute_area( inside_poly, inside_poly_num);
                ScalarType clipsquare[4];

                ScalarType x_step = (d_proj_domain[1] - d_proj_domain[0])/(ScalarType)d_picture_size[0];
                ScalarType y_step = (d_proj_domain[3] - d_proj_domain[2])/(ScalarType)d_picture_size[1];
                ScalarType grid_area = x_step*y_step;

                if(inside_poly_num >= 3)
                {
                    ScalarType max_x = inside_poly[0], max_y = inside_poly[1],min_x = inside_poly[0], min_y = inside_poly[1];

                    for(int i=0;i<inside_poly_num;i++)
                    {
                        if(inside_poly[2*i]<min_x)
                        {
                            min_x = inside_poly[2*i];
                        }
                        if(inside_poly[2*i]>max_x)
                        {
                            max_x = inside_poly[2*i];
                        }
                        if(inside_poly[2*i+1]<min_y)
                        {
                            min_y = inside_poly[2*i+1];
                        }
                        if(inside_poly[2*i+1]>max_y)
                        {
                            max_y = inside_poly[2*i+1];
                        }
                    }

                    int low_bound_x  = (int)floor((min_x - d_proj_domain[0]) / x_step);
                    int low_bound_y  = (int)floor((min_y - d_proj_domain[2]) / y_step);
                    int high_bound_x = (int)floor((max_x - d_proj_domain[0]) / x_step) + 1;
                    int high_bound_y = (int)floor((max_y - d_proj_domain[2]) / y_step) + 1;
                    
                    low_bound_x=(low_bound_x>0)?low_bound_x:0;
                    low_bound_y=(low_bound_y>0)?low_bound_y:0;
                    high_bound_x=(high_bound_x<d_picture_size[0])?high_bound_x:d_picture_size[0];
                    high_bound_y=(high_bound_y<d_picture_size[1])?high_bound_y:d_picture_size[1];

                    ScalarType total_inte = 0;

                    for(int i_x = low_bound_x; i_x < high_bound_x; i_x++)
                    {
                        for(int i_y = low_bound_y; i_y < high_bound_y; i_y++)
                        {
                            int grid = i_x + i_y * d_picture_size[0];
                            clipsquare[0] = (ScalarType)i_x * x_step + d_proj_domain[0];
                            clipsquare[1] = (ScalarType)i_y * y_step + d_proj_domain[2];
                            clipsquare[2] = (ScalarType)(i_x + 1) * x_step + d_proj_domain[0];
                            clipsquare[3] = (ScalarType)(i_y + 1) * y_step + d_proj_domain[2];
                            
                            clip_Polygon( inside_poly, inside_poly_num, clipsquare, new_poly, new_poly_num);
                            ScalarType area = compute_area( new_poly, new_poly_num);
                            if(area>1e-10)
                            {
                                ScalarType local_center[2];
                                d_compute_center( new_poly, new_poly_num, local_center);
                                ScalarType local_contribution = (d_picture[grid]) * area;
                                total_inte += local_contribution;
                                new_center_x += local_center[0]*local_contribution;
                                new_center_y += local_center[1]*local_contribution;
                            }
                        }
                    }
                    if(total_inte!=0)
                    {
                        new_center_x/=total_inte;
                        new_center_y/=total_inte;
                        d_polygons_center[2*tid] = new_center_x;
                        d_polygons_center[2*tid+1] = new_center_y;
                        d_if_exist[tid] = true;
                    }
                    else
                    {
                        ScalarType square[4] = {d_proj_domain[0],d_proj_domain[2],d_proj_domain[1],d_proj_domain[3]};
                        clip_Polygon( polygon, length_num, square, inside_poly, inside_poly_num);
                        d_compute_center( inside_poly, inside_poly_num, d_polygons_center+2*tid);
                        d_if_exist[tid] = true;
                    }
                }
                else
                {
                    d_if_exist[tid] = false;
                } 
            }
        }
        else
        {
            d_if_exist[tid] = false;
        }
    }
}

void read_wpoints_txt(std::string points_txt, Eigen::VectorXd&points, std::string weights_txt, Eigen::VectorXd&weights)
{
    std::ifstream coor_inputFile(points_txt);
    if (!coor_inputFile) 
    {
        std::cerr << "Error opening file" << std::endl;
        return;
    }
    std::ifstream weights_inputFile(weights_txt);
    if (!weights_inputFile) 
    {
        std::cerr << "Error opening file" << std::endl;
        return;
    }

    std::vector<ScalarType> points_data;
    ScalarType x, y;
    std::vector<ScalarType> weights_data;
    ScalarType w;
    while (coor_inputFile >> x >> y) 
    {
        points_data.push_back(x);
        points_data.push_back(y);
    }
    while (weights_inputFile >> w) 
    {
        weights_data.push_back(w);
    }

    coor_inputFile.close();
    weights_inputFile.close();

    points.resize(points_data.size());    
    for (size_t i = 0; i < points_data.size(); ++i) {
        points[i] = points_data[i];
    }
    weights.resize(weights_data.size());
    for (size_t i = 0; i < weights_data.size(); ++i) {
        weights[i] = weights_data[i];
    }
}

void read_weights_txt(std::string weights_txt, Eigen::VectorXd&weights)
{
    std::ifstream weights_inputFile(weights_txt);
    if (!weights_inputFile) 
    {
        std::cerr << "Error opening file" << std::endl;
        return;
    }
    weights_inputFile.precision(std::numeric_limits<ScalarType>::max_digits10);
    std::vector<ScalarType> weights_data;
    ScalarType w;

    while (weights_inputFile >> w) 
    {
        weights_data.push_back(w);
    }

    weights_inputFile.close();

    weights.resize(weights_data.size());
    for (size_t i = 0; i < weights_data.size(); ++i) {
        weights[i] = weights_data[i];
    }
}

void read_points_txt(std::string points_txt, Eigen::VectorXd&points)
{
    std::ifstream coor_inputFile(points_txt);
    if (!coor_inputFile) 
    {
        std::cerr << "Error opening file" << std::endl;
        return;
    }

    std::vector<ScalarType> points_data;
    ScalarType x, y;
    while (coor_inputFile >> x >> y) 
    {
        points_data.push_back(x);
        points_data.push_back(y);
    }

    coor_inputFile.close();

    points.resize(points_data.size());    
    for (size_t i = 0; i < points_data.size(); ++i) {
        points[i] = points_data[i];
    }
}

void read_weights_from_txt(const std::string& filename, Eigen::VectorXd& weights) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return;
    }
    std::vector<double> data;
    double value;

    while (infile >> value) {
        data.push_back(value);
    }

    infile.close();
    weights = Eigen::VectorXd::Zero(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        weights[i] = data[i];
    }
}

void write_selected_cells_to_obj(
    const std::string& obj_path,
    const Eigen::VectorXd& polygons,
    const int* start_num,
    const int* length,
    const std::vector<int>& selected_indices)
{
    std::ofstream obj_file(obj_path);
    if (!obj_file.is_open())
    {
        std::cerr << "Cannot open obj file: " << obj_path << std::endl;
        return;
    }

    int vertex_offset = 1; // obj 索引从 1 开始

    for (int cell_idx : selected_indices)
    {
        int len = length[cell_idx];
        if (len <= 0) continue;

        int start = start_num[cell_idx];

        // 写顶点
        for (int j = 0; j < len; ++j)
        {
            int poly_idx = 2 * (start + j);
            double x = polygons[poly_idx];
            double y = polygons[poly_idx + 1];
            obj_file << "v " << x << " " << y << " 0.0\n";
        }

        // 写面
        obj_file << "f";
        for (int j = 0; j < len; ++j)
        {
            obj_file << " " << (vertex_offset + j);
        }
        obj_file << "\n";

        vertex_offset += len;
    }

    obj_file.close();
    std::cout << "Saved selected cells to: " << obj_path << std::endl;
}