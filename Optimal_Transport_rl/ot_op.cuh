#include "basic_function.cuh"
#include "cublas_v2.h"
#include <filesystem>

#define CUDA_CHECK(call, iter) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::cerr << "[CUDA ERROR] " << __FILE__ << ":" << __LINE__ \
                  << " iter=" << iter \
                  << " : " << cudaGetErrorString(err__) << std::endl; \
        std::abort(); \
    } \
} while(0)

#define CUDA_KERNEL_CHECK(tag, iter, step) do { \
    cudaError_t err1__ = cudaPeekAtLastError(); \
    if (err1__ != cudaSuccess) { \
        std::cerr << "[KERNEL LAUNCH ERROR] " << tag \
                  << " iter=" << iter << " step=" << step \
                  << " : " << cudaGetErrorString(err1__) << std::endl; \
        std::abort(); \
    } \
    cudaError_t err2__ = cudaDeviceSynchronize(); \
    if (err2__ != cudaSuccess) { \
        std::cerr << "[KERNEL EXEC ERROR] " << tag \
                  << " iter=" << iter << " step=" << step \
                  << " : " << cudaGetErrorString(err2__) << std::endl; \
        std::abort(); \
    } \
} while(0)

namespace fs = std::filesystem;
cublasHandle_t m_cublasHandle;  

void initialize_cublas() {  
    cublasCreate(&m_cublasHandle);  
}  
  
void destroy_cublas() {  
    cublasDestroy(m_cublasHandle);  
} 

void dispatch_axpy(const int n, ScalarType *d_dst, const ScalarType *d_y, const ScalarType *d_x, const ScalarType *a)
{
	if (d_dst != d_y)
		cudaMemcpy(d_dst, d_y, n * sizeof(ScalarType), cudaMemcpyDeviceToDevice);

	cublasDaxpy(m_cublasHandle, int(n), a, d_x, 1, d_dst, 1);
}

void dispatch_scale(const int n, ScalarType *d_dst, const ScalarType *d_x, const ScalarType *a)
{
	if (d_dst != d_x)
		cudaMemcpy(d_dst, d_x, n * sizeof(ScalarType), cudaMemcpyDeviceToDevice);

	cublasDscal(m_cublasHandle, int(n), a, d_dst, 1);
}

void dispatch_dot(const int n, ScalarType *dst, const ScalarType *d_x, const ScalarType *d_y)
{
	cublasDdot(m_cublasHandle, int(n), d_x, 1, d_y, 1, dst);
}

int index(int indexIn, int HISTORY_SIZE) { return (indexIn % HISTORY_SIZE); }

void direct_ot(Eigen::VectorXd& points,Eigen::VectorXd&origin_weights,Eigen::VectorXd&proj_xy,Eigen::VectorXd&weights, ScalarType* picture, int* picture_size,
 	int total_iter, ScalarType convergence_tolerance, const ScalarType initial_step, const int history_size,
    ScalarType* sample_domain, ScalarType z_of_len, ScalarType z_of_receiver, Eigen::VectorXd input_light,
    std::string light_type, ScalarType* point_light_coordinate, ScalarType* proj_domain, ScalarType* proj_corner,
    std::vector<Eigen::VectorXi> idx_list, int N_threads, std::string xiushi, std::string ot_time, Eigen::VectorXd& output_points, bool* if_exist)
{
	int prev_time = std::stoi(ot_time);
    const int points_num = points.size()/2;
	const int grid_num = picture_size[0]*picture_size[1];
    ScalarType x_scalar = (sample_domain[1]-sample_domain[0]),y_scalar = (sample_domain[3]-sample_domain[2]);
	const ScalarType area_scalar = x_scalar * y_scalar;
	Eigen::VectorXd polygons (50*points_num);
    int* start_num = new int[points_num];
    int* length = new int[points_num];

	ScalarType picture_total_gray = 0;
    for (int i = 0; i < picture_size[0]; i++) {
        for (int j = 0; j < picture_size[1]; j++) {
            int k = picture_size[1] * i + j;
            picture_total_gray += picture[k];
        }
    }
	ScalarType gray_multiple = picture_total_gray/(ScalarType)area_scalar;

	ScalarType data_corner[4] = {sample_domain[0] - x_scalar/40, sample_domain[1] + x_scalar/40,
                                sample_domain[2] - y_scalar/40, sample_domain[3] + y_scalar/40};
	polygons_in_VD(points_num, points, origin_weights, polygons, start_num, length, N_threads, idx_list, data_corner);

	ScalarType* render_result = new ScalarType[grid_num];
    ScalarType* render_initial = new ScalarType[grid_num];
    for(int i=0;i<grid_num;i++)
    {
        render_result[i] = 0;
        render_initial[i] = 0;
    }

    ScalarType* d_proj_xy;
    ScalarType* d_polygons;
    ScalarType* d_clip_area;
    ScalarType* d_carry_energy;
    ScalarType* d_energy_remain_rate;
    ScalarType* d_weight_energy;
    ScalarType* d_total_energy;
    int* d_start_num;
    int* d_length;
    int* d_points_num;
    ScalarType* d_sample_domain;
    ScalarType* d_gray_multiple;
    ScalarType* d_render_result;
    ScalarType* d_picture;
    int* d_picture_size;
    int* d_grid_num;
    ScalarType* d_proj_domain;
    ScalarType* d_area_scalar;

    cudaMalloc(&d_proj_xy, 2*points_num*sizeof(ScalarType));
    cudaMalloc(&d_polygons, 50*points_num * sizeof(ScalarType));
    cudaMalloc(&d_clip_area, points_num*sizeof(ScalarType));
    cudaMalloc(&d_carry_energy, points_num*sizeof(ScalarType));
    cudaMalloc(&d_energy_remain_rate, points_num*sizeof(ScalarType));
    cudaMalloc(&d_weight_energy, points_num*sizeof(ScalarType));
    cudaMalloc(&d_total_energy, sizeof(ScalarType));
    cudaMalloc(&d_start_num, points_num * sizeof(int));
    cudaMalloc(&d_length, points_num * sizeof(int));	
    cudaMalloc(&d_points_num, sizeof(int));
	cudaMalloc(&d_sample_domain, 4*sizeof(ScalarType));
	cudaMalloc(&d_gray_multiple, sizeof(ScalarType));
    cudaMalloc(&d_render_result, grid_num*sizeof(ScalarType));
    cudaMalloc(&d_picture, grid_num*sizeof(ScalarType));
    cudaMalloc(&d_picture_size, 2*sizeof(int));
    cudaMalloc(&d_grid_num, sizeof(int));
    cudaMalloc(&d_proj_domain, 4 * sizeof(ScalarType));
    cudaMalloc(&d_area_scalar, sizeof(ScalarType));

    cudaMemcpy(d_proj_xy, proj_xy.data(), 2*points_num * sizeof(ScalarType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_polygons, polygons.data(), 50*points_num * sizeof(ScalarType), cudaMemcpyHostToDevice);
	cudaMemcpy(d_start_num, start_num, points_num * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_length, length, points_num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_points_num, &points_num, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sample_domain, sample_domain, 4*sizeof(ScalarType), cudaMemcpyHostToDevice);
	cudaMemcpy(d_gray_multiple, &gray_multiple, sizeof(ScalarType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_render_result, render_initial, grid_num*sizeof(ScalarType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_picture, picture, grid_num*sizeof(ScalarType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_picture_size, picture_size, 2*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid_num, &grid_num, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_proj_domain, proj_domain, 4 * sizeof(ScalarType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_area_scalar, &area_scalar, sizeof(ScalarType), cudaMemcpyHostToDevice);

    compute_divide<<<(grid_num+255)/256,256>>>( d_picture, d_picture, d_gray_multiple, d_grid_num);
    //计算入射能量
	if(light_type == "parallel_light")
    {
        compute_clip_area<<<(points_num+255)/256,256>>>( d_polygons, d_clip_area, d_start_num, d_length, d_points_num, d_sample_domain);
    }
    if(light_type == "point_light")
    {
        ScalarType* d_polygons_solid_angle;
        ScalarType* d_z;
        ScalarType* d_light_coord;
        ScalarType* d_total_solid_angle;

        cudaMalloc(&d_polygons_solid_angle, points_num * sizeof(ScalarType));
        cudaMalloc(&d_z, sizeof(ScalarType));
        cudaMalloc(&d_light_coord, 3 * sizeof(ScalarType));
        cudaMalloc(&d_total_solid_angle, sizeof(ScalarType));

        cudaMemcpy(d_z, &z_of_len, sizeof(ScalarType), cudaMemcpyHostToDevice);
        cudaMemcpy(d_light_coord, point_light_coordinate, 3 * sizeof(ScalarType), cudaMemcpyHostToDevice);

        solid_angle<<<(points_num+255)/256,256>>>( d_polygons, d_start_num, d_length, d_polygons_solid_angle, d_points_num, d_sample_domain, d_z, d_light_coord);
        len_solid_angle<<<1,1>>>( d_sample_domain, d_total_solid_angle, d_z, d_light_coord);
        compute_divide<<<(points_num+255)/256,256>>>( d_polygons_solid_angle, d_polygons_solid_angle, d_total_solid_angle, d_points_num);
        compute_multi<<<(points_num+255)/256,256>>>( d_clip_area, d_polygons_solid_angle, d_area_scalar, d_points_num);
        
        cudaFree(d_polygons_solid_angle);
        cudaFree(d_z);
        cudaFree(d_light_coord);
        cudaFree(d_total_solid_angle);
    }
    
    //中心初始化
    bool* d_if_weighting;
    bool* d_if_exist;
    ScalarType* d_polygons_center;

    bool if_weighting = true;

    cudaMalloc(&d_if_weighting, sizeof(bool));
    cudaMalloc(&d_if_exist,points_num * sizeof(bool));
    cudaMalloc(&d_polygons_center, 2*points_num*sizeof(ScalarType));

    cudaMemcpy(d_if_weighting, &if_weighting, sizeof(bool), cudaMemcpyHostToDevice);

    Eigen::VectorXd weight_energy(points_num);
    ScalarType total_energy;

    cudaMemcpy( weight_energy.data(), d_clip_area, points_num*sizeof(ScalarType), cudaMemcpyDeviceToHost);
    total_energy = weight_energy.sum();
    cudaMemcpy(d_total_energy, &total_energy, sizeof(ScalarType), cudaMemcpyHostToDevice);
    compute_divide<<<(points_num+255)/256,256>>>( d_clip_area, d_clip_area, d_total_energy, d_points_num);
    compute_multi<<<(points_num+255)/256,256>>>( d_carry_energy, d_clip_area, d_area_scalar, d_points_num);

    polygons_in_VD(points_num, proj_xy, weights, polygons, start_num, length, N_threads, idx_list, proj_corner);

    cudaMemcpy(d_polygons, polygons.data(), 50*points_num * sizeof(ScalarType), cudaMemcpyHostToDevice);
	cudaMemcpy(d_start_num, start_num, points_num * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_length, length, points_num * sizeof(int), cudaMemcpyHostToDevice);

    render<<<(points_num+255)/256,256>>>( d_polygons, d_carry_energy, d_start_num, d_length, d_render_result, d_points_num, d_proj_domain, d_picture_size);
    compute_multi<<<(grid_num+255)/256,256>>>( d_render_result, d_render_result, d_gray_multiple, d_grid_num);
    cudaMemcpy(render_result, d_render_result, grid_num*sizeof(ScalarType), cudaMemcpyDeviceToHost);

    std::string render_file_path = "../render_result/"+xiushi+"/"+ot_time+"/render_initial.txt";
    save_array_to_txt( render_file_path, render_result, picture_size);

    int HISTORY_SIZE = history_size;

    Eigen::VectorXd poly_inte(points_num);
    Eigen::VectorXd carry_energy(points_num);
    Eigen::VectorXd deri(points_num);
    Eigen::VectorXd direc(points_num);
    Eigen::VectorXd polygon_f(points_num);
	ScalarType minusOne = -1;							    // minusOne
    Eigen::VectorXd dot_save(points_num);
    Eigen::VectorXd weights_m1;

    ScalarType* d_weights;
    ScalarType* d_polygons_inte;
    ScalarType* d_polygons_f;
    ScalarType fkm1, fk;								    // f_{k-1}, f_k, function values at x_{k-1} and x_k
	ScalarType *d_gkm1, *d_gk;				                // g_{k-1}, g_k, gradients       at x_{k-1} and x_k
	ScalarType *d_direc;							        // search direction
    ScalarType *d_dot_save;
    ScalarType *d_tmp;
    ScalarType *d_H0;
    ScalarType *d_step;
    ScalarType* d_minus_One;

    ScalarType tmp, tmp2, beta;	
	ScalarType alpha[HISTORY_SIZE];							// alpha,        history of alphas (needed for z updates)
	ScalarType rho[HISTORY_SIZE];							// rho,          history of rhos   (needed for z updates)
	ScalarType H0 = 1.0;									// H_0,          initial inverse Hessian (diagonal, same value for all elements)
	ScalarType step = 1.0;									// step          current step length
	ScalarType *d_s;										// s,            history of solution updates
	ScalarType *d_y;										// y,            history of gradient updates
    bool restart = false; 									// a second chance if line search fails
    int loop_iter = 0; 
    ScalarType current_loss;   

    cudaMalloc(&d_weights, points_num*sizeof(ScalarType));
    cudaMalloc(&d_polygons_inte, points_num*sizeof(ScalarType));
    cudaMalloc(&d_polygons_f, points_num*sizeof(ScalarType));
    cudaMalloc(&d_gkm1, points_num*sizeof(ScalarType));
    cudaMalloc(&d_gk, points_num*sizeof(ScalarType));
    cudaMalloc(&d_direc, points_num*sizeof(ScalarType));
    cudaMalloc(&d_dot_save, points_num*sizeof(ScalarType));
    cudaMalloc(&d_tmp, sizeof(ScalarType));
    cudaMalloc(&d_H0, sizeof(ScalarType));
    cudaMalloc(&d_step, sizeof(ScalarType));
    cudaMalloc(&d_minus_One, sizeof(ScalarType));
	cudaMalloc(&d_s,  HISTORY_SIZE * points_num * sizeof(ScalarType));
	cudaMalloc(&d_y,  HISTORY_SIZE * points_num * sizeof(ScalarType));

    cudaMemcpy( carry_energy.data(), d_carry_energy, points_num*sizeof(ScalarType), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_weights, weights.data(), points_num * sizeof(ScalarType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_minus_One, &minusOne, sizeof(ScalarType), cudaMemcpyHostToDevice);

    compute_inte<<<(points_num+255)/256,256>>>( d_polygons, d_polygons_inte, d_start_num, d_length, d_picture, d_points_num, d_proj_domain, d_picture_size);
    compute_subtract<<<(points_num+255)/256,256>>>( d_polygons_inte, d_carry_energy, d_gk, d_points_num);
    compute_f<<<(points_num+255)/256,256>>>( d_proj_xy, d_weights, d_carry_energy, d_polygons_f, d_polygons, d_start_num, d_length, d_picture,
         d_points_num, d_proj_domain, d_picture_size);
    cudaMemcpy( polygon_f.data(), d_polygons_f, points_num*sizeof(ScalarType), cudaMemcpyDeviceToHost);
    fk = polygon_f.sum();//loss函数初始化
    cout<<"fk"<<fk<<endl;

    std::string render_dir_path = "../render_result/" + xiushi + "/" + ot_time + "/";
    std::string mesh_dir_path = "../mesh_output/" + xiushi + "/" + ot_time + "/";
    std::string weight_dir_path = "../weights/" + xiushi + "/" + ot_time + "/";

    string ini_mesh_file_path = mesh_dir_path + "mesh_initial.obj";
    writeOBJ( ini_mesh_file_path, polygons, start_num, length, points_num);
    
    //首次lbfgs
    for(int iter = 0;iter < total_iter; iter++)
    {
        const int MAX_IDX = std::min<int>(loop_iter, HISTORY_SIZE);
        compute_multi<<<(points_num+255)/256,256>>>( d_direc, d_gk, d_minus_One, d_points_num); // z = -gk
        CUDA_KERNEL_CHECK("compute_multi init direct=-gk", iter, -1.0);

        CUDA_CHECK(cudaMemcpy( poly_inte.data(), d_polygons_inte, points_num*sizeof(ScalarType), cudaMemcpyDeviceToHost),iter);
        CUDA_CHECK(cudaMemcpy( deri.data(), d_gk, points_num*sizeof(ScalarType), cudaMemcpyDeviceToHost),iter);
        
		if(!restart)
		{
            for (int i = 1; i <= MAX_IDX ; ++i)
            {
                int idx = index((loop_iter - i),HISTORY_SIZE);

                compute_dot<<<(points_num+255)/256,256>>>( d_s + idx * points_num, d_direc, d_dot_save, d_points_num);
                CUDA_KERNEL_CHECK("compute_dot s·direc", iter, -1.0);

                CUDA_CHECK(cudaMemcpy( dot_save.data(), d_dot_save, points_num*sizeof(ScalarType), cudaMemcpyDeviceToHost), iter);
                tmp = dot_save.sum();
                alpha[idx] = tmp * rho[idx];
                tmp = -alpha[idx];               
                CUDA_CHECK(cudaMemcpy( d_tmp, &tmp, sizeof(ScalarType), cudaMemcpyHostToDevice), iter);

                compute_xpay<<<(points_num+255)/256,256>>>( d_direc, d_tmp, d_y + idx * points_num, d_points_num);
                CUDA_KERNEL_CHECK("compute_xpay direc += tmp*y", iter, -1.0);
            }

            CUDA_CHECK(cudaMemcpy( d_H0, &H0, sizeof(ScalarType), cudaMemcpyHostToDevice), iter);
            compute_multi<<<(points_num+255)/256,256>>>( d_direc, d_direc, d_H0, d_points_num);
            CUDA_KERNEL_CHECK("second loop: direc *= H0", iter, -1.0);

            for (int i = MAX_IDX; i > 0; --i)
            {
                int idx = index((loop_iter - i),HISTORY_SIZE);

                compute_dot<<<(points_num+255)/256,256>>>( d_y + idx * points_num, d_direc, d_dot_save, d_points_num);
                CUDA_KERNEL_CHECK("second loop: dot(y, direc)", iter, -1.0);

                CUDA_CHECK(cudaMemcpy(dot_save.data(), d_dot_save, points_num*sizeof(ScalarType), cudaMemcpyDeviceToHost), iter);
                tmp = dot_save.sum();
                beta = rho[idx] * tmp;
                tmp = alpha[idx] - beta;

                CUDA_CHECK(cudaMemcpy(d_tmp, &tmp, sizeof(ScalarType), cudaMemcpyHostToDevice), iter);
                compute_xpay<<<(points_num+255)/256,256>>>( d_direc, d_tmp, d_s + idx * points_num, d_points_num);
                CUDA_KERNEL_CHECK("second loop: direc += tmp * s", iter, -1.0);
            }
        }

        compute_dot<<<(points_num+255)/256,256>>>( d_gk, d_direc, d_dot_save, d_points_num);
        CUDA_KERNEL_CHECK("check descent: dot(gk, direc)", iter, -1.0);
        CUDA_CHECK(cudaMemcpy(dot_save.data(), d_dot_save, points_num*sizeof(ScalarType), cudaMemcpyDeviceToHost), iter);

        if(dot_save.sum() >= 0)
		{
			compute_multi<<<(points_num+255)/256,256>>>( d_direc, d_gk, d_minus_One, d_points_num);
            CUDA_KERNEL_CHECK("fallback: direc = -gk", iter, -1.0);

			loop_iter = 0;
            H0 = 1.0;
			cout << "iter = " << iter << ", set z = -gk,restart" << endl;
		}

        CUDA_CHECK(cudaMemcpy(d_gkm1, d_gk, points_num * sizeof(ScalarType), cudaMemcpyDeviceToDevice), iter);
        weights_m1 = weights;
        fkm1 = fk;

        CUDA_CHECK(cudaMemcpy( direc.data(), d_direc, points_num*sizeof(ScalarType), cudaMemcpyDeviceToHost), iter);
        step = initial_step;

		for (;;)
		{
            weights = weights_m1 + step*direc;

            polygons_in_VD(points_num, proj_xy, weights, polygons, start_num, length, N_threads, idx_list, proj_corner);

            int max_len = length[0];
            int max_idx = 0;
            for (int i = 1; i < points_num; i++)
            {
                if (length[i] > max_len)
                {
                    max_len = length[i];
                    max_idx = i;
                }
            }
            if(max_len>max_edge_num/2)
            {
                step *= 0.5;

                std::cout << "[LINESEARCH REJECT: polygon too complex] "<< "iter=" << iter<< " new_step=" << step
                        << " max_len=" << max_len << " max_edge_num=" << max_edge_num << " max_idx=" << max_idx << std::endl;
            }
            else
            {               
                CUDA_CHECK(cudaMemcpy(d_polygons, polygons.data(), 50*points_num*sizeof(ScalarType), cudaMemcpyHostToDevice), iter);
                CUDA_CHECK(cudaMemcpy(d_start_num, start_num, points_num*sizeof(int), cudaMemcpyHostToDevice), iter);
                CUDA_CHECK(cudaMemcpy(d_length, length, points_num*sizeof(int), cudaMemcpyHostToDevice), iter);
                CUDA_CHECK(cudaMemcpy(d_weights, weights.data(), points_num*sizeof(ScalarType), cudaMemcpyHostToDevice), iter);

                compute_f<<<(points_num+255)/256,256>>>( d_proj_xy, d_weights, d_carry_energy, d_polygons_f, d_polygons, d_start_num, d_length,
                    d_picture, d_points_num, d_proj_domain, d_picture_size);
                CUDA_KERNEL_CHECK("compute_f", iter, step);

                CUDA_CHECK(cudaMemcpy(polygon_f.data(), d_polygons_f, points_num*sizeof(ScalarType), cudaMemcpyDeviceToHost), iter);
                fk = polygon_f.sum();

                if(fk<fkm1)
                {
                    compute_inte<<<(points_num+255)/256,256>>>( d_polygons, d_polygons_inte, d_start_num, d_length, d_picture, d_points_num,
                        d_proj_domain, d_picture_size);
                    CUDA_KERNEL_CHECK("compute_inte", iter, step);
                    compute_subtract<<<(points_num+255)/256,256>>>( d_polygons_inte, d_carry_energy, d_gk, d_points_num);
                    CUDA_KERNEL_CHECK("compute_subtract gk", iter, step);
                    restart = false;
                    break;
                }
                else
                {
                    step *= 0.5;
                }
            }

            if(step <= 1e-4 * initial_step)
            {
                restart = true;

                CUDA_CHECK(cudaMemcpy(d_gk, d_gkm1, points_num*sizeof(ScalarType), cudaMemcpyDeviceToDevice), iter);

                loop_iter = 0;
                H0 = 1.0;
                weights = weights_m1;
                fk = fkm1;
                cout << "iter = " << iter << ", restart " <<"step"<<step<< endl;
                break;
            }
        }

        if(step <= 1e-4 * initial_step)
        {
            continue;
        }
        else
        {
            int hist_idx = index(loop_iter, HISTORY_SIZE);

            CUDA_CHECK(cudaMemcpy(d_step, &step, sizeof(ScalarType), cudaMemcpyHostToDevice), iter);

            compute_multi<<<(points_num+255)/256,256>>>(d_s + hist_idx * points_num, d_direc, d_step, d_points_num);
            CUDA_KERNEL_CHECK("history update: s = step * direc", iter, step);

            compute_subtract<<<(points_num+255)/256,256>>>(d_gk, d_gkm1, d_y + hist_idx * points_num, d_points_num);
            CUDA_KERNEL_CHECK("history update: y = gk - gkm1", iter, step);

            compute_dot<<<(points_num+255)/256,256>>>(d_y + hist_idx * points_num, d_s + hist_idx * points_num, d_dot_save, d_points_num);
            CUDA_KERNEL_CHECK("history update: dot(y,s)", iter, step);

            CUDA_CHECK(cudaMemcpy(dot_save.data(), d_dot_save, points_num*sizeof(ScalarType), cudaMemcpyDeviceToHost), iter);
            tmp = dot_save.sum();

            compute_dot<<<(points_num+255)/256,256>>>(d_y + hist_idx * points_num, d_y + hist_idx * points_num, d_dot_save, d_points_num);
            CUDA_KERNEL_CHECK("history update: dot(y,y)", iter, step);
            CUDA_CHECK(cudaMemcpy(dot_save.data(), d_dot_save, points_num*sizeof(ScalarType), cudaMemcpyDeviceToHost), iter);
            tmp2 = dot_save.sum();
            
            rho[index(loop_iter,HISTORY_SIZE)] = 1.0 / tmp;            
            if (tmp2 > 1e-5)
            {
                H0 = tmp / tmp2;
            }
            else
            {
                H0 = 1.0;
            }

            loop_iter++;
        }

        current_loss = deri.norm();
        int check_degenerate = 0;
        for(int i=0;i<points_num;i++)
        {
            if(poly_inte[i] == 0)
            {
                check_degenerate++;
            }
        }

        if(iter%20==0)
        {
            cout<<iter<<"次"<<",loss为 "<<fk<<",梯度为 "<<current_loss<<",映射能量为"<<poly_inte.sum()<<",总能量为"<<carry_energy.sum()
            <<",退化点数为"<<check_degenerate<<"step为 "<<step<<endl;
        }

        if(iter%500==0)
        {
            std::string weight_file_path = weight_dir_path + "weights_" + to_string(iter) + ".txt";

            std::filesystem::path p(weight_file_path);
            std::filesystem::create_directories(p.parent_path());

            ofstream weight_file(weight_file_path);
            for(int i=0;i<points_num;i++)
            {
                weight_file<<weights[i]<<std::endl;
            }
            weight_file.close();
        }

        if(iter%500 == 0)
        {
            cudaMemcpy(d_render_result, render_initial, grid_num*sizeof(ScalarType), cudaMemcpyHostToDevice);
            render<<<(points_num+255)/256,256>>>( d_polygons, d_carry_energy, d_start_num, d_length, d_render_result, d_points_num, d_proj_domain, d_picture_size);
            compute_multi<<<(grid_num+255)/256,256>>>( d_render_result, d_render_result, d_gray_multiple, d_grid_num);
            cudaMemcpy(render_result, d_render_result, grid_num*sizeof(ScalarType), cudaMemcpyDeviceToHost);

            std::string render_file_path = render_dir_path + "render_" + to_string(iter) + ".txt";
            save_array_to_txt( render_file_path, render_result, picture_size);
        }

        if(current_loss<convergence_tolerance && check_degenerate == 0)
        {
            cout<<"迭代结束于"<<iter<<endl;
            cout<<iter<<"次"<<",loss为 "<<fk<<",梯度为 "<<current_loss<<",area为"<<poly_inte.sum()<<",总能量为"<<carry_energy.sum()
            <<",退化点数为"<<check_degenerate<<endl;

            std::string weight_file_path = weight_dir_path + "weights_" + to_string(iter) + ".txt";
            ofstream weight_file(weight_file_path);
            for(int i=0;i<points_num;i++)
            {
                weight_file<<weights[i]<<std::endl;
            }
            weight_file.close();

            string mesh_file_path = mesh_dir_path+"mesh_" + to_string(iter) + ".obj";
            writeOBJ( mesh_file_path, polygons, start_num, length, points_num);

            cudaMemcpy(d_render_result, render_initial, grid_num*sizeof(ScalarType), cudaMemcpyHostToDevice);
            render<<<(points_num+255)/256,256>>>( d_polygons, d_carry_energy, d_start_num, d_length, d_render_result, d_points_num, d_proj_domain, d_picture_size);
            compute_multi<<<(grid_num+255)/256,256>>>( d_render_result, d_render_result, d_gray_multiple, d_grid_num);
            cudaMemcpy(render_result, d_render_result, grid_num*sizeof(ScalarType), cudaMemcpyDeviceToHost);

            std::string render_file_path = render_dir_path+"render_" + to_string(iter) + ".txt";
            save_array_to_txt( render_file_path, render_result, picture_size);

            break;
        }
    }

    polygons_in_VD(points_num, proj_xy, weights, polygons, start_num, length, N_threads, idx_list, proj_corner);

    cudaMemcpy(d_polygons, polygons.data(), 50*points_num * sizeof(ScalarType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_start_num, start_num, points_num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_length, length, points_num * sizeof(int), cudaMemcpyHostToDevice);
    compute_center<<<(points_num+255)/256,256>>>( d_polygons, d_polygons_center, d_start_num, d_length, d_picture, d_points_num, d_if_weighting, d_if_exist, d_proj_domain, d_picture_size);
    cudaMemcpy( output_points.data(), d_polygons_center, 2 * points_num * sizeof(ScalarType), cudaMemcpyDeviceToHost);
    cudaMemcpy( if_exist, d_if_exist, points_num * sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_proj_xy);
	cudaFree(d_polygons);
    cudaFree(d_clip_area);
    cudaFree(d_carry_energy);
    cudaFree(d_energy_remain_rate);
    cudaFree(d_weight_energy);
    cudaFree(d_total_energy);
    cudaFree(d_start_num);
    cudaFree(d_length);	
    cudaFree(d_points_num);
	cudaFree(d_sample_domain);
    cudaFree(d_gray_multiple);
    cudaFree(d_render_result);
    cudaFree(d_picture);
    cudaFree(d_picture_size);
    cudaFree(d_grid_num);
    cudaFree(d_proj_domain);
    cudaFree(d_area_scalar);

    cudaFree(d_weights);
    cudaFree(d_polygons_inte);
    cudaFree(d_polygons_f);
    cudaFree(d_gkm1);
    cudaFree(d_gk);
    cudaFree(d_direc);
    cudaFree(d_dot_save);
    cudaFree(d_tmp);
    cudaFree(d_H0);
    cudaFree(d_step);
    cudaFree(d_minus_One);
	cudaFree(d_s);
	cudaFree(d_y);

    cudaFree(d_if_weighting);
    cudaFree(d_if_exist);
    cudaFree(d_polygons_center);

	delete[] render_result;
	delete[] render_initial;
    delete[] start_num;
    delete[] length;
}

void aarl_direct_ot(Eigen::VectorXd& points, Eigen::VectorXd& origin_weights, Eigen::VectorXd&proj_xy, Eigen::VectorXd&weights,
    ScalarType* picture, int* picture_size, int total_iter, ScalarType convergence_tolerance, const ScalarType initial_step, const int history_size,
    const int aa_history_size, ScalarType aa_fixed_para,bool if_aa_op_initial,
    ScalarType* sample_domain, ScalarType z_of_len, ScalarType z_of_receiver,
    Eigen::VectorXd input_light, Eigen::VectorXd& input_point_centers, Eigen::VectorXd& inside_light, Eigen::VectorXd height,
    ScalarType refraction_index, std::string light_type, ScalarType* point_light_coordinate, ScalarType* proj_domain, ScalarType* proj_corner,
    std::vector<Eigen::VectorXi> idx_list, int N_threads, std::string xiushi, std::string ot_time, Eigen::VectorXd& output_points, bool* if_exist)
{
    int prev_time = std::stoi(ot_time);
    const int points_num = points.size()/2;
	const int grid_num = picture_size[0]*picture_size[1];
    ScalarType x_scalar = (sample_domain[1]-sample_domain[0]),y_scalar = (sample_domain[3]-sample_domain[2]);
	const ScalarType area_scalar = x_scalar * y_scalar;
	Eigen::VectorXd polygons (50*points_num);
    int* start_num = new int[points_num];
    int* length = new int[points_num];

	ScalarType picture_total_gray = 0;
    for (int i = 0; i < picture_size[0]; i++) {
        for (int j = 0; j < picture_size[1]; j++) {
            int k = picture_size[1] * i + j;
            picture_total_gray += picture[k];
        }
    }
	ScalarType gray_multiple = picture_total_gray/(ScalarType)area_scalar;

	ScalarType data_corner[4] = {sample_domain[0] - x_scalar/40, sample_domain[1] + x_scalar/40,
                                sample_domain[2] - y_scalar/40, sample_domain[3] + y_scalar/40};
	polygons_in_VD(points_num, points, origin_weights, polygons, start_num, length, N_threads, idx_list, data_corner);

	ScalarType* render_result = new ScalarType[grid_num];
    ScalarType* render_initial = new ScalarType[grid_num];
    for(int i=0;i<grid_num;i++)
    {
        render_result[i] = 0;
        render_initial[i] = 0;
    }

    ScalarType* d_proj_xy;
    ScalarType* d_polygons;
    ScalarType* d_clip_area;
    ScalarType* d_carry_energy;
    ScalarType* d_energy_remain_rate;
    ScalarType* d_weight_energy;
    ScalarType* d_total_energy;
    int* d_start_num;
    int* d_length;
    int* d_points_num;
    ScalarType* d_sample_domain;
    ScalarType* d_gray_multiple;
    ScalarType* d_render_result;
    ScalarType* d_picture;
    int* d_picture_size;
    int* d_grid_num;
    ScalarType* d_proj_domain;
    ScalarType* d_area_scalar;

    cudaMalloc(&d_proj_xy, 2*points_num*sizeof(ScalarType));
    cudaMalloc(&d_polygons, 50*points_num * sizeof(ScalarType));
    cudaMalloc(&d_clip_area, points_num*sizeof(ScalarType));
    cudaMalloc(&d_carry_energy, points_num*sizeof(ScalarType));
    cudaMalloc(&d_energy_remain_rate, points_num*sizeof(ScalarType));
    cudaMalloc(&d_weight_energy, points_num*sizeof(ScalarType));
    cudaMalloc(&d_total_energy, sizeof(ScalarType));
    cudaMalloc(&d_start_num, points_num * sizeof(int));
    cudaMalloc(&d_length, points_num * sizeof(int));	
    cudaMalloc(&d_points_num, sizeof(int));
	cudaMalloc(&d_sample_domain, 4*sizeof(ScalarType));
	cudaMalloc(&d_gray_multiple, sizeof(ScalarType));
    cudaMalloc(&d_render_result, grid_num*sizeof(ScalarType));
    cudaMalloc(&d_picture, grid_num*sizeof(ScalarType));
    cudaMalloc(&d_picture_size, 2*sizeof(int));
    cudaMalloc(&d_grid_num, sizeof(int));
    cudaMalloc(&d_proj_domain, 4 * sizeof(ScalarType));
    cudaMalloc(&d_area_scalar, sizeof(ScalarType));

    cudaMemcpy(d_proj_xy, proj_xy.data(), 2*points_num * sizeof(ScalarType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_polygons, polygons.data(), 50*points_num * sizeof(ScalarType), cudaMemcpyHostToDevice);
	cudaMemcpy(d_start_num, start_num, points_num * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_length, length, points_num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_points_num, &points_num, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sample_domain, sample_domain, 4*sizeof(ScalarType), cudaMemcpyHostToDevice);
	cudaMemcpy(d_gray_multiple, &gray_multiple, sizeof(ScalarType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_render_result, render_initial, grid_num*sizeof(ScalarType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_picture, picture, grid_num*sizeof(ScalarType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_picture_size, picture_size, 2*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid_num, &grid_num, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_proj_domain, proj_domain, 4 * sizeof(ScalarType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_area_scalar, &area_scalar, sizeof(ScalarType), cudaMemcpyHostToDevice);

    compute_divide<<<(grid_num+255)/256,256>>>( d_picture, d_picture, d_gray_multiple, d_grid_num);
    //计算入射能量
	if(light_type == "parallel_light")
    {
        compute_clip_area<<<(points_num+255)/256,256>>>( d_polygons, d_clip_area, d_start_num, d_length, d_points_num, d_sample_domain);
    }
    if(light_type == "point_light")
    {
        ScalarType* d_polygons_solid_angle;
        ScalarType* d_z;
        ScalarType* d_light_coord;
        ScalarType* d_total_solid_angle;

        cudaMalloc(&d_polygons_solid_angle, points_num * sizeof(ScalarType));
        cudaMalloc(&d_z, sizeof(ScalarType));
        cudaMalloc(&d_light_coord, 3 * sizeof(ScalarType));
        cudaMalloc(&d_total_solid_angle, sizeof(ScalarType));

        cudaMemcpy(d_z, &z_of_len, sizeof(ScalarType), cudaMemcpyHostToDevice);
        cudaMemcpy(d_light_coord, point_light_coordinate, 3 * sizeof(ScalarType), cudaMemcpyHostToDevice);

        solid_angle<<<(points_num+255)/256,256>>>( d_polygons, d_start_num, d_length, d_polygons_solid_angle, d_points_num, d_sample_domain, d_z, d_light_coord);
        len_solid_angle<<<1,1>>>( d_sample_domain, d_total_solid_angle, d_z, d_light_coord);
        compute_divide<<<(points_num+255)/256,256>>>( d_polygons_solid_angle, d_polygons_solid_angle, d_total_solid_angle, d_points_num);
        compute_multi<<<(points_num+255)/256,256>>>( d_clip_area, d_polygons_solid_angle, d_area_scalar, d_points_num);
        
        cudaFree(d_polygons_solid_angle);
        cudaFree(d_z);
        cudaFree(d_light_coord);
        cudaFree(d_total_solid_angle);
    }
    
    //中心初始化
    bool* d_weight_true,*d_weight_false;
    bool* d_if_exist;
    ScalarType* d_polygons_center;

    bool weight_true = true, weight_false = false;

    cudaMalloc(&d_weight_true, sizeof(bool));
    cudaMalloc(&d_weight_false, sizeof(bool));
    cudaMalloc(&d_if_exist, points_num * sizeof(bool));
    cudaMalloc(&d_polygons_center, 2*points_num*sizeof(ScalarType));

    cudaMemcpy(d_weight_true, &weight_true, sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_false, &weight_false, sizeof(bool), cudaMemcpyHostToDevice);

    //计算初始损耗
    Eigen::VectorXd weight_energy(points_num);
    Eigen::VectorXd energy_remain_rate(points_num);
    //初始化lbfgs参数
    int HISTORY_SIZE = history_size;
    Eigen::VectorXd poly_inte(points_num);
    Eigen::VectorXd carry_energy(points_num);
    Eigen::VectorXd deri(points_num);
    Eigen::VectorXd direc(points_num);
    Eigen::VectorXd polygon_f(points_num);
	ScalarType minusOne = -1;							    // minusOne
    Eigen::VectorXd dot_save(points_num);
    Eigen::VectorXd weights_m1;

    ScalarType* d_weights;
    ScalarType* d_polygons_inte;
    ScalarType* d_polygons_f;
    ScalarType fkm1, fk;								    // f_{k-1}, f_k, function values at x_{k-1} and x_k
	ScalarType *d_gkm1, *d_gk;				                // g_{k-1}, g_k, gradients       at x_{k-1} and x_k
	ScalarType *d_direc;							        // search direction
    ScalarType *d_dot_save;
    ScalarType *d_tmp;
    ScalarType *d_H0;
    ScalarType *d_step;
    ScalarType* d_minus_One;

    ScalarType tmp, tmp2, beta;	
	ScalarType alpha[HISTORY_SIZE];							// alpha,        history of alphas (needed for z updates)
	ScalarType rho[HISTORY_SIZE];							// rho,          history of rhos   (needed for z updates)
	ScalarType H0 = 1.0;									// H_0,          initial inverse Hessian (diagonal, same value for all elements)
	ScalarType step = 1.0;									// step          current step length
	ScalarType *d_s;										// s,            history of solution updates
	ScalarType *d_y;										// y,            history of gradient updates
    bool restart = false; 									// a second chance if line search fails
    int loop_iter = 0; 
    ScalarType current_loss;   

    cudaMalloc(&d_weights, points_num*sizeof(ScalarType));
    cudaMalloc(&d_polygons_inte, points_num*sizeof(ScalarType));
    cudaMalloc(&d_polygons_f, points_num*sizeof(ScalarType));
    cudaMalloc(&d_gkm1, points_num*sizeof(ScalarType));
    cudaMalloc(&d_gk, points_num*sizeof(ScalarType));
    cudaMalloc(&d_direc, points_num*sizeof(ScalarType));
    cudaMalloc(&d_dot_save, points_num*sizeof(ScalarType));
    cudaMalloc(&d_tmp, sizeof(ScalarType));
    cudaMalloc(&d_H0, sizeof(ScalarType));
    cudaMalloc(&d_step, sizeof(ScalarType));
    cudaMalloc(&d_minus_One, sizeof(ScalarType));
	cudaMalloc(&d_s,  HISTORY_SIZE * points_num * sizeof(ScalarType));
	cudaMalloc(&d_y,  HISTORY_SIZE * points_num * sizeof(ScalarType));

    polygons_in_VD(points_num, proj_xy, weights, polygons, start_num, length, N_threads, idx_list, proj_corner);
    cudaMemcpy(d_polygons, polygons.data(), 50*points_num * sizeof(ScalarType), cudaMemcpyHostToDevice);
	cudaMemcpy(d_start_num, start_num, points_num * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_length, length, points_num * sizeof(int), cudaMemcpyHostToDevice);
    
    compute_center<<<(points_num+255)/256,256>>>( d_polygons, d_polygons_center, d_start_num, d_length, d_picture, d_points_num, d_weight_false, d_if_exist, d_proj_domain, d_picture_size);
    cudaMemcpy( output_points.data(), d_polygons_center, 2 * points_num * sizeof(ScalarType), cudaMemcpyDeviceToHost);
    cudaMemcpy( if_exist, d_if_exist, points_num * sizeof(bool), cudaMemcpyDeviceToHost);
    energy_remain_rate = compute_energy_remain_rate(input_point_centers,z_of_len,input_light,inside_light,height,output_points,z_of_receiver,refraction_index);
    ScalarType total_energy;
    cudaMemcpy(d_energy_remain_rate, energy_remain_rate.data(), points_num * sizeof(ScalarType), cudaMemcpyHostToDevice);
    compute_dot<<<(points_num+255)/256,256>>>( d_clip_area, d_energy_remain_rate, d_weight_energy, d_points_num);
    cudaMemcpy( weight_energy.data(), d_weight_energy, points_num*sizeof(ScalarType), cudaMemcpyDeviceToHost);
    total_energy = weight_energy.sum();
    cudaMemcpy(d_total_energy, &total_energy, sizeof(ScalarType), cudaMemcpyHostToDevice);
    compute_divide<<<(points_num+255)/256,256>>>( d_weight_energy, d_weight_energy, d_total_energy, d_points_num);
    compute_multi<<<(points_num+255)/256,256>>>( d_carry_energy, d_weight_energy, d_area_scalar, d_points_num);

    render<<<(points_num+255)/256,256>>>( d_polygons, d_carry_energy, d_start_num, d_length, d_render_result, d_points_num, d_proj_domain, d_picture_size);
    compute_multi<<<(grid_num+255)/256,256>>>( d_render_result, d_render_result, d_gray_multiple, d_grid_num);
    cudaMemcpy(render_result, d_render_result, grid_num*sizeof(ScalarType), cudaMemcpyDeviceToHost);
    std::string pre_render_file_path = "../render_result/"+xiushi+"/"+ot_time+"/pre_render_initial.txt";
    save_array_to_txt( pre_render_file_path, render_result, picture_size);

    cudaMemcpy(carry_energy.data(), d_carry_energy, points_num*sizeof(ScalarType), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_weights, weights.data(), points_num * sizeof(ScalarType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_minus_One, &minusOne, sizeof(ScalarType), cudaMemcpyHostToDevice);
    
    compute_inte<<<(points_num+255)/256,256>>>( d_polygons, d_polygons_inte, d_start_num, d_length, d_picture, d_points_num, d_proj_domain, d_picture_size);
    compute_subtract<<<(points_num+255)/256,256>>>( d_polygons_inte, d_carry_energy, d_gk, d_points_num);
    compute_f<<<(points_num+255)/256,256>>>( d_proj_xy, d_weights, d_carry_energy, d_polygons_f, d_polygons, d_start_num, d_length, d_picture,
         d_points_num, d_proj_domain, d_picture_size);
    cudaMemcpy( polygon_f.data(), d_polygons_f, points_num*sizeof(ScalarType), cudaMemcpyDeviceToHost);
    fk = polygon_f.sum();//loss函数初始化
    cout<<"fk"<<fk<<endl;

    std::string render_dir_path = "../render_result/" + xiushi + "/" + ot_time + "/";
    std::string mesh_dir_path = "../mesh_output/" + xiushi + "/" + ot_time + "/";
    std::string weight_dir_path = "../weights/" + xiushi + "/" + ot_time + "/";

    std::string ini_mesh_file_path = mesh_dir_path + "pre_initial_mesh.obj";
    writeOBJ( ini_mesh_file_path, polygons, start_num, length, points_num);
    
    //首次lbfgs
    if(if_aa_op_initial == true)
    {
        for(int iter = 0; iter < total_iter; iter++)
        {
            const int MAX_IDX = std::min<int>(loop_iter, HISTORY_SIZE);
            compute_multi<<<(points_num+255)/256,256>>>( d_direc, d_gk, d_minus_One, d_points_num); // z = -gk
            CUDA_KERNEL_CHECK("compute_multi init direct=-gk", iter, -1.0);

            CUDA_CHECK(cudaMemcpy( poly_inte.data(), d_polygons_inte, points_num*sizeof(ScalarType), cudaMemcpyDeviceToHost),iter);
            CUDA_CHECK(cudaMemcpy( deri.data(), d_gk, points_num*sizeof(ScalarType), cudaMemcpyDeviceToHost),iter);
            
            if(!restart)
            {
                for (int i = 1; i <= MAX_IDX ; ++i)
                {
                    int idx = index((loop_iter - i),HISTORY_SIZE);

                    compute_dot<<<(points_num+255)/256,256>>>( d_s + idx * points_num, d_direc, d_dot_save, d_points_num);
                    CUDA_KERNEL_CHECK("compute_dot s·direc", iter, -1.0);

                    CUDA_CHECK(cudaMemcpy( dot_save.data(), d_dot_save, points_num*sizeof(ScalarType), cudaMemcpyDeviceToHost), iter);
                    tmp = dot_save.sum();
                    alpha[idx] = tmp * rho[idx];
                    tmp = -alpha[idx];               
                    CUDA_CHECK(cudaMemcpy( d_tmp, &tmp, sizeof(ScalarType), cudaMemcpyHostToDevice), iter);

                    compute_xpay<<<(points_num+255)/256,256>>>( d_direc, d_tmp, d_y + idx * points_num, d_points_num);
                    CUDA_KERNEL_CHECK("compute_xpay direc += tmp*y", iter, -1.0);
                }

                CUDA_CHECK(cudaMemcpy( d_H0, &H0, sizeof(ScalarType), cudaMemcpyHostToDevice), iter);
                compute_multi<<<(points_num+255)/256,256>>>( d_direc, d_direc, d_H0, d_points_num);
                CUDA_KERNEL_CHECK("second loop: direc *= H0", iter, -1.0);

                for (int i = MAX_IDX; i > 0; --i)
                {
                    int idx = index((loop_iter - i),HISTORY_SIZE);

                    compute_dot<<<(points_num+255)/256,256>>>( d_y + idx * points_num, d_direc, d_dot_save, d_points_num);
                    CUDA_KERNEL_CHECK("second loop: dot(y, direc)", iter, -1.0);

                    CUDA_CHECK(cudaMemcpy(dot_save.data(), d_dot_save, points_num*sizeof(ScalarType), cudaMemcpyDeviceToHost), iter);
                    tmp = dot_save.sum();
                    beta = rho[idx] * tmp;
                    tmp = alpha[idx] - beta;

                    CUDA_CHECK(cudaMemcpy(d_tmp, &tmp, sizeof(ScalarType), cudaMemcpyHostToDevice), iter);
                    compute_xpay<<<(points_num+255)/256,256>>>( d_direc, d_tmp, d_s + idx * points_num, d_points_num);
                    CUDA_KERNEL_CHECK("second loop: direc += tmp * s", iter, -1.0);
                }
            }

            compute_dot<<<(points_num+255)/256,256>>>( d_gk, d_direc, d_dot_save, d_points_num);
            CUDA_KERNEL_CHECK("check descent: dot(gk, direc)", iter, -1.0);
            CUDA_CHECK(cudaMemcpy(dot_save.data(), d_dot_save, points_num*sizeof(ScalarType), cudaMemcpyDeviceToHost), iter);

            if(dot_save.sum() >= 0)
            {
                compute_multi<<<(points_num+255)/256,256>>>( d_direc, d_gk, d_minus_One, d_points_num);
                CUDA_KERNEL_CHECK("fallback: direc = -gk", iter, -1.0);

                loop_iter = 0;
                H0 = 1.0;
                cout << "iter = " << iter << ", set z = -gk,restart" << endl;
            }

            CUDA_CHECK(cudaMemcpy(d_gkm1, d_gk, points_num * sizeof(ScalarType), cudaMemcpyDeviceToDevice), iter);
            weights_m1 = weights;
            fkm1 = fk;

            CUDA_CHECK(cudaMemcpy( direc.data(), d_direc, points_num*sizeof(ScalarType), cudaMemcpyDeviceToHost), iter);
            step = initial_step;

            for (;;)
            {
                weights = weights_m1 + step*direc;

                polygons_in_VD(points_num, proj_xy, weights, polygons, start_num, length, N_threads, idx_list, proj_corner);

                int max_len = length[0];
                int max_idx = 0;
                for (int i = 1; i < points_num; i++)
                {
                    if (length[i] > max_len)
                    {
                        max_len = length[i];
                        max_idx = i;
                    }
                }
                if(max_len>max_edge_num/2)
                {
                    step *= 0.5;

                    std::cout << "[LINESEARCH REJECT: polygon too complex] "<< "iter=" << iter<< " new_step=" << step
                            << " max_len=" << max_len << " max_edge_num=" << max_edge_num << " max_idx=" << max_idx << std::endl;
                }
                else
                {               
                    CUDA_CHECK(cudaMemcpy(d_polygons, polygons.data(), 50*points_num*sizeof(ScalarType), cudaMemcpyHostToDevice), iter);
                    CUDA_CHECK(cudaMemcpy(d_start_num, start_num, points_num*sizeof(int), cudaMemcpyHostToDevice), iter);
                    CUDA_CHECK(cudaMemcpy(d_length, length, points_num*sizeof(int), cudaMemcpyHostToDevice), iter);
                    CUDA_CHECK(cudaMemcpy(d_weights, weights.data(), points_num*sizeof(ScalarType), cudaMemcpyHostToDevice), iter);

                    compute_f<<<(points_num+255)/256,256>>>( d_proj_xy, d_weights, d_carry_energy, d_polygons_f, d_polygons, d_start_num, d_length,
                        d_picture, d_points_num, d_proj_domain, d_picture_size);
                    CUDA_KERNEL_CHECK("compute_f", iter, step);

                    CUDA_CHECK(cudaMemcpy(polygon_f.data(), d_polygons_f, points_num*sizeof(ScalarType), cudaMemcpyDeviceToHost), iter);
                    fk = polygon_f.sum();

                    if(fk<fkm1)
                    {
                        compute_inte<<<(points_num+255)/256,256>>>( d_polygons, d_polygons_inte, d_start_num, d_length, d_picture, d_points_num,
                            d_proj_domain, d_picture_size);
                        CUDA_KERNEL_CHECK("compute_inte", iter, step);
                        compute_subtract<<<(points_num+255)/256,256>>>( d_polygons_inte, d_carry_energy, d_gk, d_points_num);
                        CUDA_KERNEL_CHECK("compute_subtract gk", iter, step);
                        restart = false;
                        break;
                    }
                    else
                    {
                        step *= 0.5;
                    }
                }

                if(step <= 1e-4 * initial_step)
                {
                    restart = true;

                    CUDA_CHECK(cudaMemcpy(d_gk, d_gkm1, points_num*sizeof(ScalarType), cudaMemcpyDeviceToDevice), iter);

                    loop_iter = 0;
                    H0 = 1.0;
                    weights = weights_m1;
                    fk = fkm1;
                    cout << "iter = " << iter << ", restart " <<"step"<<step<< endl;
                    break;
                }
            }

            if(step <= 1e-4 * initial_step)
            {
                continue;
            }
            else
            {
                int hist_idx = index(loop_iter, HISTORY_SIZE);

                CUDA_CHECK(cudaMemcpy(d_step, &step, sizeof(ScalarType), cudaMemcpyHostToDevice), iter);

                compute_multi<<<(points_num+255)/256,256>>>(d_s + hist_idx * points_num, d_direc, d_step, d_points_num);
                CUDA_KERNEL_CHECK("history update: s = step * direc", iter, step);

                compute_subtract<<<(points_num+255)/256,256>>>(d_gk, d_gkm1, d_y + hist_idx * points_num, d_points_num);
                CUDA_KERNEL_CHECK("history update: y = gk - gkm1", iter, step);

                compute_dot<<<(points_num+255)/256,256>>>(d_y + hist_idx * points_num, d_s + hist_idx * points_num, d_dot_save, d_points_num);
                CUDA_KERNEL_CHECK("history update: dot(y,s)", iter, step);

                CUDA_CHECK(cudaMemcpy(dot_save.data(), d_dot_save, points_num*sizeof(ScalarType), cudaMemcpyDeviceToHost), iter);
                tmp = dot_save.sum();

                compute_dot<<<(points_num+255)/256,256>>>(d_y + hist_idx * points_num, d_y + hist_idx * points_num, d_dot_save, d_points_num);
                CUDA_KERNEL_CHECK("history update: dot(y,y)", iter, step);
                CUDA_CHECK(cudaMemcpy(dot_save.data(), d_dot_save, points_num*sizeof(ScalarType), cudaMemcpyDeviceToHost), iter);
                tmp2 = dot_save.sum();
                
                rho[index(loop_iter,HISTORY_SIZE)] = 1.0 / tmp;            
                if (tmp2 > 1e-5)
                {
                    H0 = tmp / tmp2;
                }
                else
                {
                    H0 = 1.0;
                }

                loop_iter++;
            }

            current_loss = deri.norm();
            int check_degenerate = 0;
            for(int i=0;i<points_num;i++)
            {
                if(poly_inte[i] == 0)
                {
                    check_degenerate++;
                }
            }

            if(iter%20==0)
            {
                cout<<iter<<"次"<<",loss为 "<<fk<<",梯度为 "<<current_loss<<",映射能量为"<<poly_inte.sum()<<",总能量为"<<carry_energy.sum()
                <<",退化点数为"<<check_degenerate<<"step为 "<<step<<endl;
            }

            if(iter%500==0)
            {
                std::string weight_file_path = weight_dir_path + "pre_weights_" + to_string(iter) + ".txt";

                std::filesystem::path p(weight_file_path);
                std::filesystem::create_directories(p.parent_path());

                ofstream weight_file(weight_file_path);
                for(int i=0;i<points_num;i++)
                {
                    weight_file<<weights[i]<<std::endl;
                }
                weight_file.close();
            }

            if(iter%500 == 0)
            {
                cudaMemcpy(d_render_result, render_initial, grid_num*sizeof(ScalarType), cudaMemcpyHostToDevice);
                render<<<(points_num+255)/256,256>>>( d_polygons, d_carry_energy, d_start_num, d_length, d_render_result, d_points_num, d_proj_domain, d_picture_size);
                compute_multi<<<(grid_num+255)/256,256>>>( d_render_result, d_render_result, d_gray_multiple, d_grid_num);
                cudaMemcpy(render_result, d_render_result, grid_num*sizeof(ScalarType), cudaMemcpyDeviceToHost);

                std::string render_file_path = render_dir_path + "pre_render_" + to_string(iter) + ".txt";
                save_array_to_txt( render_file_path, render_result, picture_size);
            }

            if(current_loss<convergence_tolerance && check_degenerate == 0)
            {
                cout<<"迭代结束于"<<iter<<endl;
                cout<<iter<<"次"<<",loss为 "<<fk<<",梯度为 "<<current_loss<<",area为"<<poly_inte.sum()<<",总能量为"<<carry_energy.sum()
                <<",退化点数为"<<check_degenerate<<endl;

                std::string weight_file_path = weight_dir_path + "pre_weights_" + to_string(iter) + ".txt";
                ofstream weight_file(weight_file_path);
                for(int i=0;i<points_num;i++)
                {
                    weight_file<<weights[i]<<std::endl;
                }
                weight_file.close();

                string mesh_file_path = mesh_dir_path + "pre_mesh_" + to_string(iter) + ".obj";
                writeOBJ( mesh_file_path, polygons, start_num, length, points_num);

                cudaMemcpy(d_render_result, render_initial, grid_num*sizeof(ScalarType), cudaMemcpyHostToDevice);
                render<<<(points_num+255)/256,256>>>( d_polygons, d_carry_energy, d_start_num, d_length, d_render_result, d_points_num, d_proj_domain, d_picture_size);
                compute_multi<<<(grid_num+255)/256,256>>>( d_render_result, d_render_result, d_gray_multiple, d_grid_num);
                cudaMemcpy(render_result, d_render_result, grid_num*sizeof(ScalarType), cudaMemcpyDeviceToHost);

                std::string render_file_path = render_dir_path + "pre_render_" + to_string(iter) + ".txt";
                save_array_to_txt( render_file_path, render_result, picture_size);

                break;
            }
        }     
    }

    polygons_in_VD(points_num, proj_xy, weights, polygons, start_num, length, N_threads, idx_list, proj_corner);
    cudaMemcpy(d_polygons, polygons.data(), 50*points_num*sizeof(ScalarType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_start_num, start_num, points_num*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_length, length, points_num*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights.data(), points_num*sizeof(ScalarType), cudaMemcpyHostToDevice);

    compute_center<<<(points_num+255)/256,256>>>( d_polygons, d_polygons_center, d_start_num, d_length, d_picture, d_points_num, d_weight_false, d_if_exist, d_proj_domain, d_picture_size);
    cudaMemcpy( output_points.data(), d_polygons_center, 2 * points_num * sizeof(ScalarType), cudaMemcpyDeviceToHost);
    cudaMemcpy( if_exist, d_if_exist, points_num * sizeof(bool), cudaMemcpyDeviceToHost);
    energy_remain_rate = compute_energy_remain_rate(input_point_centers,z_of_len,input_light,inside_light,height,output_points,z_of_receiver,refraction_index);
    cudaMemcpy(d_energy_remain_rate, energy_remain_rate.data(), points_num * sizeof(ScalarType), cudaMemcpyHostToDevice);
    compute_dot<<<(points_num+255)/256,256>>>( d_clip_area, d_energy_remain_rate, d_weight_energy, d_points_num);
    cudaMemcpy( weight_energy.data(), d_weight_energy, points_num*sizeof(ScalarType), cudaMemcpyDeviceToHost);
    total_energy = weight_energy.sum();
    cudaMemcpy(d_total_energy, &total_energy, sizeof(ScalarType), cudaMemcpyHostToDevice);
    compute_divide<<<(points_num+255)/256,256>>>( d_weight_energy, d_weight_energy, d_total_energy, d_points_num);
    compute_multi<<<(points_num+255)/256,256>>>( d_carry_energy, d_weight_energy, d_area_scalar, d_points_num);

    cudaMemcpy(d_render_result, render_initial, grid_num*sizeof(ScalarType), cudaMemcpyHostToDevice);
    render<<<(points_num+255)/256,256>>>( d_polygons, d_carry_energy, d_start_num, d_length, d_render_result, d_points_num, d_proj_domain, d_picture_size);
    compute_multi<<<(grid_num+255)/256,256>>>( d_render_result, d_render_result, d_gray_multiple, d_grid_num);
    cudaMemcpy(render_result, d_render_result, grid_num*sizeof(ScalarType), cudaMemcpyDeviceToHost);
    std::string render_file_path = "../render_result/" + xiushi + "/" + ot_time + "/render_initial.txt";
    save_array_to_txt( render_file_path, render_result, picture_size);

    compute_inte<<<(points_num+255)/256,256>>>( d_polygons, d_polygons_inte, d_start_num, d_length, d_picture, d_points_num, d_proj_domain, d_picture_size);
    compute_subtract<<<(points_num+255)/256,256>>>( d_polygons_inte, d_carry_energy, d_gk, d_points_num);
    cudaMemcpy( deri.data(), d_gk, points_num*sizeof(ScalarType), cudaMemcpyDeviceToHost);
    cout<<"current_loss"<<deri.norm()<<endl;

    compute_f<<<(points_num+255)/256,256>>>( d_proj_xy, d_weights, d_carry_energy, d_polygons_f, d_polygons, d_start_num, d_length, d_picture,
        d_points_num, d_proj_domain, d_picture_size);
    cudaMemcpy( polygon_f.data(), d_polygons_f, points_num*sizeof(ScalarType), cudaMemcpyDeviceToHost);
    fk = polygon_f.sum();//loss函数初始化
    cout<<"fk"<<fk<<endl; 

    // anderson acceleration : w = w + fix_p * (gk) = w + fix_p * (inte - energy)
    int AA_HISTORY_SIZE = aa_history_size;
    int aa_loop_iter = 0;
    ScalarType fixed_para = aa_fixed_para;

    ScalarType* d_aa_weight_history;
    ScalarType* d_aa_gk_history;
    ScalarType* d_aa_fixed_function_history;
    ScalarType* d_fixed_para;
    int* d_current_history_size;
    ScalarType* d_alpha;
    ScalarType* d_aa_new_weights;

	cudaMalloc(&d_aa_weight_history,  AA_HISTORY_SIZE * points_num * sizeof(ScalarType));
	cudaMalloc(&d_aa_gk_history,  AA_HISTORY_SIZE * points_num * sizeof(ScalarType));
	cudaMalloc(&d_aa_fixed_function_history,  AA_HISTORY_SIZE * points_num * sizeof(ScalarType));
    cudaMalloc(&d_fixed_para, sizeof(ScalarType));
    cudaMalloc(&d_current_history_size, sizeof(int));
    cudaMalloc(&d_alpha, AA_HISTORY_SIZE * sizeof(ScalarType));
    cudaMalloc(&d_aa_new_weights, points_num * sizeof(ScalarType));

    cudaMemcpy(d_fixed_para, &fixed_para, sizeof(ScalarType), cudaMemcpyHostToDevice);

    for(int iter = 0;iter < total_iter; iter++)
    {
        int current_history_size = std::min<int>(aa_loop_iter+1, AA_HISTORY_SIZE);
        int hist_idx = index(aa_loop_iter, AA_HISTORY_SIZE);
        storage<<<(points_num+255)/256,256>>>(d_aa_weight_history + hist_idx * points_num, d_weights, d_points_num);
        storage<<<(points_num+255)/256,256>>>(d_aa_gk_history + hist_idx * points_num, d_gk, d_points_num);
        compute_fixed_function<<<(points_num+255)/256,256>>>(d_weights, d_gk, d_fixed_para,
             d_aa_fixed_function_history + hist_idx * points_num, d_points_num);

        Eigen::MatrixXd A(current_history_size, current_history_size);

        for (int i = 0; i < current_history_size; ++i)
        {
            for (int j = 0; j <= i; ++j)
            {
                ScalarType* gi = d_aa_gk_history + i * points_num;
                ScalarType* gj = d_aa_gk_history + j * points_num;

                compute_dot<<<(points_num+255)/256,256>>>( gi, gj, d_dot_save, d_points_num);
                CUDA_CHECK(cudaMemcpy(dot_save.data(), d_dot_save, points_num*sizeof(ScalarType), cudaMemcpyDeviceToHost), iter);
                ScalarType dot_sum = dot_save.sum();
                A(i, j) = static_cast<double>(dot_sum);
                A(j, i) = static_cast<double>(dot_sum);
            }
        }
        A += 1e-10 * Eigen::MatrixXd::Identity(current_history_size, current_history_size);
        Eigen::VectorXd ones = Eigen::VectorXd::Ones(current_history_size);
        Eigen::VectorXd alpha = A.ldlt().solve(ones);
        alpha /= ones.dot(alpha);

        std::vector<ScalarType> h_alpha(AA_HISTORY_SIZE, ScalarType(0));
        for (int i = 0; i < current_history_size; ++i)
        {
            h_alpha[i] = static_cast<ScalarType>(alpha(i));
        }
        cudaMemcpy(d_alpha, h_alpha.data(), AA_HISTORY_SIZE * sizeof(ScalarType), cudaMemcpyHostToDevice);
        cudaMemcpy(d_current_history_size, &current_history_size, sizeof(int), cudaMemcpyHostToDevice);
        compute_weighted_sum<<<(points_num+255)/256,256>>>(d_aa_new_weights, d_aa_fixed_function_history,d_alpha,d_current_history_size,d_points_num);
        compute_subtract<<<(points_num+255)/256,256>>>( d_aa_new_weights, d_weights, d_direc, d_points_num); // z = new_weight - weight

        CUDA_CHECK(cudaMemcpy( poly_inte.data(), d_polygons_inte, points_num*sizeof(ScalarType), cudaMemcpyDeviceToHost),iter);
        CUDA_CHECK(cudaMemcpy(d_gkm1, d_gk, points_num * sizeof(ScalarType), cudaMemcpyDeviceToDevice), iter);
        weights_m1 = weights;
        fkm1 = fk;

        CUDA_CHECK(cudaMemcpy( direc.data(), d_direc, points_num*sizeof(ScalarType), cudaMemcpyDeviceToHost), iter);
        step = 1;

		for (;;)
		{
            weights = weights_m1 + step*direc;

            polygons_in_VD(points_num, proj_xy, weights, polygons, start_num, length, N_threads, idx_list, proj_corner);

            int max_len = length[0];
            int max_idx = 0;
            for (int i = 1; i < points_num; i++)
            {
                if (length[i] > max_len)
                {
                    max_len = length[i];
                    max_idx = i;
                }
            }
            if(max_len>max_edge_num/2)
            {
                step *= 0.5;

                std::cout << "[LINESEARCH REJECT: polygon too complex] "<< "iter=" << iter<< " new_step=" << step
                        << " max_len=" << max_len << " max_edge_num=" << max_edge_num << " max_idx=" << max_idx << std::endl;
            }
            else
            {
                CUDA_CHECK(cudaMemcpy(d_polygons, polygons.data(), 50*points_num*sizeof(ScalarType), cudaMemcpyHostToDevice), iter);
                CUDA_CHECK(cudaMemcpy(d_start_num, start_num, points_num*sizeof(int), cudaMemcpyHostToDevice), iter);
                CUDA_CHECK(cudaMemcpy(d_length, length, points_num*sizeof(int), cudaMemcpyHostToDevice), iter);
                CUDA_CHECK(cudaMemcpy(d_weights, weights.data(), points_num*sizeof(ScalarType), cudaMemcpyHostToDevice), iter);
                compute_center<<<(points_num+255)/256,256>>>( d_polygons, d_polygons_center, d_start_num, d_length, d_picture, d_points_num, d_weight_false, d_if_exist, d_proj_domain, d_picture_size);
                cudaMemcpy( output_points.data(), d_polygons_center, 2 * points_num * sizeof(ScalarType), cudaMemcpyDeviceToHost);
                cudaMemcpy( if_exist, d_if_exist, points_num * sizeof(bool), cudaMemcpyDeviceToHost);

                energy_remain_rate = compute_energy_remain_rate(input_point_centers,z_of_len,input_light,inside_light,height,output_points,z_of_receiver,refraction_index);
                cudaMemcpy(d_energy_remain_rate, energy_remain_rate.data(), points_num * sizeof(ScalarType), cudaMemcpyHostToDevice);
                compute_dot<<<(points_num+255)/256,256>>>( d_clip_area, d_energy_remain_rate, d_weight_energy, d_points_num);
                cudaMemcpy( weight_energy.data(), d_weight_energy, points_num*sizeof(ScalarType), cudaMemcpyDeviceToHost);
                total_energy = weight_energy.sum();
                cudaMemcpy(d_total_energy, &total_energy, sizeof(ScalarType), cudaMemcpyHostToDevice);
                compute_divide<<<(points_num+255)/256,256>>>( d_weight_energy, d_weight_energy, d_total_energy, d_points_num);
                compute_multi<<<(points_num+255)/256,256>>>( d_carry_energy, d_weight_energy, d_area_scalar, d_points_num);
                cudaMemcpy( carry_energy.data(), d_carry_energy, points_num*sizeof(ScalarType), cudaMemcpyDeviceToHost);

                compute_f<<<(points_num+255)/256,256>>>( d_proj_xy, d_weights, d_carry_energy, d_polygons_f, d_polygons, d_start_num, d_length,
                    d_picture, d_points_num, d_proj_domain, d_picture_size);
                CUDA_KERNEL_CHECK("compute_f", iter, step);

                CUDA_CHECK(cudaMemcpy(polygon_f.data(), d_polygons_f, points_num*sizeof(ScalarType), cudaMemcpyDeviceToHost), iter);
                fk = polygon_f.sum();

                compute_inte<<<(points_num+255)/256,256>>>( d_polygons, d_polygons_inte, d_start_num, d_length, d_picture, d_points_num,
                    d_proj_domain, d_picture_size);
                CUDA_KERNEL_CHECK("compute_inte", iter, step);
                compute_subtract<<<(points_num+255)/256,256>>>( d_polygons_inte, d_carry_energy, d_gk, d_points_num);
                CUDA_KERNEL_CHECK("compute_subtract gk", iter, step);
                CUDA_CHECK(cudaMemcpy( deri.data(), d_gk, points_num*sizeof(ScalarType), cudaMemcpyDeviceToHost),iter);
                aa_loop_iter++;
                break;
            }

            if(step <= 1e-4 * initial_step)
            {
                CUDA_CHECK(cudaMemcpy(d_gk, d_gkm1, points_num*sizeof(ScalarType), cudaMemcpyDeviceToDevice), iter);
                aa_loop_iter = 0;
                weights = weights_m1;
                fk = fkm1;
                cout << "iter = " << iter << ", restart " <<"step"<<step<< endl;
                break;
            }
        }

        current_loss = deri.norm();
        
        int check_degenerate = 0;
        if(iter%10==0)
        {
            for(int i=0;i<points_num;i++)
            {
                if(poly_inte[i] == 0)
                {
                    check_degenerate++;
                }
            }
            cout<<iter<<"次"<<",loss为 "<<fk<<",梯度为 "<<current_loss<<",映射能量为"<<poly_inte.sum()<<",总能量为"<<carry_energy.sum()
            <<",退化点数为"<<check_degenerate<<"step为 "<<step<<endl;
        }

        if(iter%200==0)
        {
            std::string weight_file_path = weight_dir_path + std::to_string(prev_time) + "_weights_" + to_string(iter) + ".txt";

            std::filesystem::path p(weight_file_path);
            std::filesystem::create_directories(p.parent_path());

            ofstream weight_file(weight_file_path);
            for(int i=0;i<points_num;i++)
            {
                weight_file<<weights[i]<<std::endl;
            }
            weight_file.close();
        }

        if(iter%500 == 0)
        {
            cudaMemcpy(d_render_result, render_initial, grid_num*sizeof(ScalarType), cudaMemcpyHostToDevice);
            render<<<(points_num+255)/256,256>>>( d_polygons, d_carry_energy, d_start_num, d_length, d_render_result, d_points_num, d_proj_domain, d_picture_size);
            compute_multi<<<(grid_num+255)/256,256>>>( d_render_result, d_render_result, d_gray_multiple, d_grid_num);
            cudaMemcpy(render_result, d_render_result, grid_num*sizeof(ScalarType), cudaMemcpyDeviceToHost);

            std::string render_file_path = render_dir_path + std::to_string(prev_time) + "_render_" + to_string(iter) + ".txt";
            save_array_to_txt( render_file_path, render_result, picture_size);
        }

        if(current_loss<convergence_tolerance && check_degenerate == 0)
        {
            cout<<"迭代结束于"<<iter<<endl;

            cout<<iter<<"次"<<",loss为 "<<fk<<",梯度为 "<<current_loss<<",area为"<<poly_inte.sum()<<",总能量为"<<carry_energy.sum()
            <<",退化点数为"<<check_degenerate<<endl;
            std::string weight_file_path = weight_dir_path + std::to_string(prev_time) + "_weights_" + to_string(iter) + ".txt";
            ofstream weight_file(weight_file_path);
            for(int i=0;i<points_num;i++)
            {
                weight_file<<weights[i]<<std::endl;
            }
            weight_file.close();

            string mesh_file_path = mesh_dir_path + std::to_string(prev_time) + "_mesh_" + to_string(iter) + ".obj";
            writeOBJ( mesh_file_path, polygons, start_num, length, points_num);
            
            cudaMemcpy(d_render_result, render_initial, grid_num*sizeof(ScalarType), cudaMemcpyHostToDevice);
            render<<<(points_num+255)/256,256>>>( d_polygons, d_carry_energy, d_start_num, d_length, d_render_result, d_points_num, d_proj_domain, d_picture_size);
            compute_multi<<<(grid_num+255)/256,256>>>( d_render_result, d_render_result, d_gray_multiple, d_grid_num);
            cudaMemcpy(render_result, d_render_result, grid_num*sizeof(ScalarType), cudaMemcpyDeviceToHost);

            std::string render_file_path = render_dir_path + std::to_string(prev_time) + "_render_" + to_string(iter) + ".txt";
            save_array_to_txt( render_file_path, render_result, picture_size);

            break;
        }
    }
    
    //输出中心点
    polygons_in_VD(points_num, proj_xy, weights, polygons, start_num, length, N_threads, idx_list, proj_corner);

    cudaMemcpy(d_polygons, polygons.data(), 50*points_num * sizeof(ScalarType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_start_num, start_num, points_num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_length, length, points_num * sizeof(int), cudaMemcpyHostToDevice);
    compute_center<<<(points_num+255)/256,256>>>( d_polygons, d_polygons_center, d_start_num, d_length, d_picture, d_points_num, d_weight_true, d_if_exist, d_proj_domain, d_picture_size);
    cudaMemcpy( output_points.data(), d_polygons_center, 2 * points_num * sizeof(ScalarType), cudaMemcpyDeviceToHost);
    cudaMemcpy( if_exist, d_if_exist, points_num * sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_proj_xy);
	cudaFree(d_polygons);
    cudaFree(d_clip_area);
    cudaFree(d_carry_energy);
    cudaFree(d_energy_remain_rate);
    cudaFree(d_weight_energy);
    cudaFree(d_total_energy);
    cudaFree(d_start_num);
    cudaFree(d_length);	
    cudaFree(d_points_num);
	cudaFree(d_sample_domain);
    cudaFree(d_gray_multiple);
    cudaFree(d_render_result);
    cudaFree(d_picture);
    cudaFree(d_picture_size);
    cudaFree(d_grid_num);
    cudaFree(d_proj_domain);
    cudaFree(d_area_scalar);

    cudaFree(d_weights);
    cudaFree(d_polygons_inte);
    cudaFree(d_polygons_f);
    cudaFree(d_gkm1);
    cudaFree(d_gk);
    cudaFree(d_direc);
    cudaFree(d_dot_save);
    cudaFree(d_tmp);
    cudaFree(d_H0);
    cudaFree(d_step);
    cudaFree(d_minus_One);
	cudaFree(d_s);
	cudaFree(d_y);

    cudaFree(d_weight_true);
    cudaFree(d_weight_false);
    cudaFree(d_if_exist);
    cudaFree(d_polygons_center);

	cudaFree(d_aa_weight_history);
	cudaFree(d_aa_gk_history);
	cudaFree(d_aa_fixed_function_history);
    cudaFree(d_fixed_para);
    cudaFree(d_current_history_size);
    cudaFree(d_alpha);
    cudaFree(d_aa_new_weights);

	delete[] render_result;
	delete[] render_initial;
    delete[] start_num;
    delete[] length;
}