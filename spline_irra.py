import os
import sys
import torch
import numpy as np
from utils.ConfigLoader import sp_JSONDataProcessor
import utils.data_utils as data_utils
import utils.basic_function as basic_function
import utils.optimization as optimization
import shutil
import irra_render
import time
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device_ids = [0]
torch.set_num_threads(8)

if __name__=="__main__":
    script_abs_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_abs_path)
    cpp_script_dir = script_dir
    cpp_build_dir = os.path.join(cpp_script_dir,"Optimal_Transport_rl","build")
    cpp_ot_executable = os.path.join(cpp_build_dir, "caustic_GPU")
    
    json_path = os.path.join(script_dir,"configs/_py_sp_settings.json")
    loader = sp_JSONDataProcessor(json_path)
    print(loader.data["xiushi"])
    cpp_json_path = cpp_script_dir+"/Optimal_Transport_rl/set_json/"+loader.data["xiushi"]+"/"
    os.makedirs(cpp_json_path, exist_ok=True)
    shutil.copy(json_path, cpp_json_path)

    data_utils.pic_print_txt(loader)
    tensor_image,total_gray_value,pic_size,max_gray = data_utils.load_pic_data(loader)
    points = data_utils.initial_xy_point(loader)
    proj_points = data_utils.initial_proj_point(loader,points)
    triangles_tensor, left_idx, right_idx, bottom_idx, top_idx = data_utils.find_boundary_points(loader, points)

    ot_seed_path = cpp_script_dir + "/Optimal_Transport_rl/seed_point/"+loader.data['xiushi']+"/"
    os.makedirs(ot_seed_path, exist_ok=True)

    basic_function.define_parameter(ot_seed_path, points, proj_points, left_idx, right_idx, bottom_idx, top_idx)

    xy_tensor = torch.tensor(points, dtype=torch.float32).to("cuda:0").requires_grad_()
    proj_xy_tensor = torch.tensor(proj_points, dtype=torch.float32).to("cuda:0").requires_grad_()

    if loader.data['light_para']['light_type'] == "point_light":
        vertical_direct = torch.zeros((xy_tensor.shape[0], 3), device=xy_tensor.device)
        vertical_direct[:,2] = 1
        inside_light_direct, pq_grad = basic_function.compute_inside_refraction_jacobian(loader, xy_tensor, vertical_direct)
        inside_light_direct = inside_light_direct.detach()
    if loader.data['light_para']['light_type'] == "parallel_light":
        inside_light_direct = torch.zeros((xy_tensor.shape[0], 3)).to("cuda:0")
        inside_light_direct[:,2] = 1

    inside_light_direct_np = inside_light_direct.detach().cpu().numpy()
    np.savetxt(ot_seed_path + "/inside_light_direct.txt", inside_light_direct_np, fmt="%.6f")

    distance = data_utils.initialize_caustic_spline(loader, device_ids, if_requires_grad=True)
    # # 读取半途模型（路径相对于项目根目录）
    # data_path = os.path.join(
    #     script_dir, "result_spline",
    #     "ildTL_tri_standard_sp_point_light_rl_sample",
    #     "control_point", "op_opti_1", "model_60000.pth")
    # distance.module.load_spline(data_path, device="cuda:0")

    # 最优传输过程
    for total_iter in range(0,loader.data['total_opti_para']['total_ot_iter_num']):
        opti_stage = "ot_opti"
        os.makedirs(ot_seed_path + f"{total_iter}" + "/", exist_ok=True)

        height = distance(xy_tensor)
        height_np = height.detach().cpu().numpy()
        np.savetxt(ot_seed_path + f"{total_iter}" + "/height.txt", height_np, fmt="%.6f")

        ot_time = f"{total_iter}"

        basic_function.run_cpp_program( cpp_build_dir, cpp_ot_executable, args=[ot_time,loader.data["xiushi"]])

        point_data_path = script_dir + "/data/point_data/" + loader.data["xiushi"] + "/"
        xy_tensor, proj_xy_tensor = data_utils.load_ot_data(point_data_path, total_iter)

        spline_end_iter = loader.data['total_opti_para']['ot_spline_end_iter'][total_iter]
        spline_learn_rate = loader.data['total_opti_para']['ot_spline_learn_rate'][total_iter]
        optimization.spline_optimization(loader,distance,xy_tensor,proj_xy_tensor,triangles_tensor,spline_end_iter,spline_learn_rate,
                                         total_gray_value,opti_stage,total_iter,pic_size)

    # 防止投影三角形翻转
    proj_xy_tensor,max_threshold,min_threshold = optimization.initial_optimization(loader,proj_xy_tensor,triangles_tensor)
    zeta_1 = optimization.compute_zeta(loader,proj_xy_tensor,triangles_tensor,max_threshold,min_threshold)
    origin_tri_area_0 = basic_function.compute_triangle_area(loader,xy_tensor,triangles_tensor).detach()

    # # 输入半途数据
    # input_name = "ildTL_tri_standard_sp_point_light_rl_sample"
    # iter_num = 1
    # data_num = 30000
    # xy_tensor, proj_xy_tensor, triangles_tensor = data_utils.load_point_data_id(script_dir, input_name, iter_num, data_num)

    for total_iter in range(0,loader.data['total_opti_para']['total_op_iter_num']):
        opti_stage = "op_opti"
        # 采样点细分
        if total_iter == loader.data['total_opti_para']['total_op_iter_num']-1:
            print("细分开始")
            new_xy_tensor, new_proj_xy_tensor, new_triangles_tensor = data_utils.subdivide_two_meshes(xy_tensor, proj_xy_tensor, triangles_tensor)
            zeta_1 = torch.cat([ zeta_1 / 4, zeta_1 / 4, zeta_1 / 4, zeta_1 / 4], dim=0)
            origin_tri_area_0 = torch.cat([ origin_tri_area_0 / 4, origin_tri_area_0 / 4, origin_tri_area_0 / 4, origin_tri_area_0 / 4], dim=0)
            
            xy_tensor = new_xy_tensor.detach()
            proj_xy_tensor = new_proj_xy_tensor.detach()
            triangles_tensor = new_triangles_tensor
            
            xy_tensor.requires_grad_(True)
            proj_xy_tensor.requires_grad_(True)
        
        # 采样点优化
        origin_tri_area = basic_function.compute_triangle_area(loader,xy_tensor, triangles_tensor).detach()
        origin_rot = basic_function.compute_rot(loader,distance,xy_tensor,proj_xy_tensor,triangles_tensor)
        origin_average_rot = origin_rot/origin_tri_area
        origin_average_rot_max = origin_average_rot.mean()
        display_rotation = (origin_average_rot / origin_average_rot_max) * 255
        display_rotation = torch.clamp(display_rotation, min=0, max=255).to(torch.uint8)
        tri_color = torch.stack([torch.full_like(display_rotation, 255), 255 - display_rotation, 255 - display_rotation],  dim=1)

        mesh_foldername = loader.data['data_mesh_folder']+f"{total_iter}/"
        os.makedirs(mesh_foldername, exist_ok=True)
        basic_function.clear_folder(mesh_foldername)
        mesh_xiushi = "initial"
        basic_function.display_color(mesh_foldername, mesh_xiushi, distance, xy_tensor, proj_xy_tensor, triangles_tensor, tri_color)

        inte_end_iter = loader.data['total_opti_para']['total_inte_end_iter'][total_iter]
        inte_origin_learn_rate = loader.data['total_opti_para']['total_inte_origin_learn_rate'][total_iter]
        inte_aim_learn_rate = loader.data['total_opti_para']['total_inte_aim_learn_rate'][total_iter]
        xy_tensor, proj_xy_tensor = optimization.render_inte_optimization(loader,distance,xy_tensor,proj_xy_tensor,triangles_tensor,total_gray_value,pic_size,
                                                                          tensor_image,origin_average_rot_max,inte_end_iter,
                                                                          inte_origin_learn_rate,inte_aim_learn_rate,total_iter,origin_tri_area_0,zeta_1,max_gray)
    
        spline_end_iter = loader.data['total_opti_para']['op_spline_end_iter'][total_iter]
        spline_learn_rate = loader.data['total_opti_para']['op_spline_learn_rate'][total_iter]
        optimization.spline_optimization(loader,distance,xy_tensor,proj_xy_tensor,triangles_tensor,spline_end_iter,spline_learn_rate,
                                         total_gray_value,opti_stage,total_iter,pic_size)

    render_folder = loader.root_dir+"/result_spline/"+loader.data["xiushi"]+"/ray_tracing/render/gauss/"
    module_folder = loader.root_dir+"/result_spline/"+loader.data["xiushi"]+"/ray_tracing/control_point/gauss/"
    optimization.spline_render(loader,distance, total_gray_value, tensor_image, pic_size, render_folder, module_folder)
    