import os
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from itertools import groupby
from PIL import Image
import subprocess
import utils.data_utils as data_utils
import time
from datetime import datetime
import subprocess

def run_cpp_program(cpp_build_dir, cpp_executable, args=None):
    try:
        cmd = [cpp_executable]

        if args is not None:
            cmd += [str(x) for x in args]

        print("Running:", cmd)

        result = subprocess.run(
            cmd,
            cwd=cpp_build_dir,
            text=True,
            check=True
        )

        return result.stdout

    except subprocess.CalledProcessError as e:
        print(f"Error running C++ program: {e}")
        return None
    
def print_time(time_path):
    current_time = time.time()
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(time_path, "a") as f:
        f.write(f"Timestamp: {current_time_str} ({current_time:.6f})\n")

def compute_outside_refract_direct(loader,input_direct,normal_direct):
    input_direct = F.normalize(input_direct)
    normal_direct = F.normalize(normal_direct)
    dot_products = torch.sum(input_direct * normal_direct, dim=1, keepdim=True)
    refraction_index = loader.data["len_para"]["refraction_index"]
    output_direct = torch.sqrt(refraction_index * refraction_index *(dot_products.pow(2) - 1) + 1) * normal_direct + refraction_index * (input_direct - dot_products * normal_direct)
    return output_direct

def compute_outside_refract_direct_mask(loader, input_direct, normal_direct, eps=1e-12):
    input_direct = F.normalize(input_direct, dim=1)
    normal_direct = F.normalize(normal_direct, dim=1)
    dot_products = torch.sum(input_direct * normal_direct, dim=1, keepdim=True)
    refraction_index = loader.data["len_para"]["refraction_index"]

    # 判别项
    discrim = refraction_index**2 * (dot_products.pow(2) - 1.0) + 1.0
    # True = 正常折射，False = 全反射
    valid_mask = (discrim.squeeze(1) >= 0.0)
    # 安全 sqrt（避免 NaN）
    sqrt_term = torch.sqrt(torch.clamp(discrim, min=0.0) + eps)

    output_direct = (
        sqrt_term * normal_direct
        + refraction_index * (input_direct - dot_products * normal_direct)
    )

    # 全反射点设为0（或者你可以改成反射方向）
    output_direct = torch.where(
        valid_mask.unsqueeze(1),
        output_direct,
        torch.zeros_like(output_direct)
    )
    return output_direct, valid_mask

def compute_inside_light(loader, origin_xy_data, normal_direct,if_retain=True, if_create = True):
    device = origin_xy_data.device
    n = origin_xy_data.shape[0]
    
    z_of_len = loader.data["len_para"]["z_of_len"]
    point_light_coordinate = torch.tensor(
        loader.data['light_para']['point_light_coordinate'],
        dtype=torch.float32, device=device
    ).unsqueeze(0)  # (1, 3)
    ri = 1 / loader.data["len_para"]["refraction_index"]
    
    z_tensor = torch.full((n, 1), z_of_len, device=device)
    origin_xyz_data = torch.cat((origin_xy_data, z_tensor), dim=1)  # (n, 3)
    input_light_direct = origin_xyz_data - point_light_coordinate  # (n, 3)
    
    input_norm = F.normalize(input_light_direct, dim=1)
    normal_norm = F.normalize(normal_direct, dim=1)
    
    dot_products = torch.sum(input_norm * normal_norm, dim=1, keepdim=True)  # (n, 1)
    temp = ri**2 * (dot_products**2 - 1) + 1
    sqrt_temp = torch.sqrt(temp)
    output_direct = sqrt_temp * normal_norm + ri * (input_norm - dot_products * normal_norm)
    
    output_direct = output_direct / output_direct[:, 2:3]  # (n, 3)

    return output_direct

def compute_inside_refraction_jacobian(loader, origin_xy_data, normal_direct):
    origin_xy_data.requires_grad_()
    device = origin_xy_data.device
    n = origin_xy_data.shape[0]
    
    z_of_len = loader.data["len_para"]["z_of_len"]
    point_light_coordinate = torch.tensor(
        loader.data['light_para']['point_light_coordinate'],
        dtype=torch.float32, device=device
    ).unsqueeze(0)  # (1, 3)
    ri = 1 / loader.data["len_para"]["refraction_index"]
    
    z_tensor = torch.full((n, 1), z_of_len, device=device)
    origin_xyz_data = torch.cat((origin_xy_data, z_tensor), dim=1)  # (n, 3)
    input_light_direct = origin_xyz_data - point_light_coordinate  # (n, 3)
    
    input_norm = F.normalize(input_light_direct, dim=1)
    normal_norm = F.normalize(normal_direct, dim=1)
    
    dot_products = torch.sum(input_norm * normal_norm, dim=1, keepdim=True)  # (n, 1)
    temp = ri**2 * (dot_products**2 - 1) + 1
    sqrt_temp = torch.sqrt(temp)
    output_direct = sqrt_temp * normal_norm + ri * (input_norm - dot_products * normal_norm)
    
    output_direct = output_direct / output_direct[:, 2:3]  # (n, 3)
    output_pq = output_direct[:, :2]
    
    sum_p = torch.sum(output_pq[:, 0])  # 标量
    sum_q = torch.sum(output_pq[:, 1])  # 标量
    
    grad_p = torch.autograd.grad(
        outputs=sum_p,
        inputs=origin_xy_data,
        grad_outputs=torch.tensor(1.0, device=device),  # 标量梯度
        retain_graph=True,
        create_graph=True
    )[0]
    
    grad_q = torch.autograd.grad(
        outputs=sum_q,
        inputs=origin_xy_data,
        grad_outputs=torch.tensor(1.0, device=device),  # 标量梯度
        retain_graph=True,  # 不需要再保留计算图
        create_graph=True
    )[0]
    
    pq_grad = torch.cat([
        grad_p[:, 0].unsqueeze(1),  # dp/dx
        grad_p[:, 1].unsqueeze(1),  # dp/dy
        grad_q[:, 0].unsqueeze(1),  # dq/dx
        grad_q[:, 1].unsqueeze(1)   # dq/dy
    ], dim=1)  # [n, 4]

    return output_direct, pq_grad

def compute_ray_energy(loader, origin_xy_data, total_energy):
    device = origin_xy_data.device
    n = origin_xy_data.shape[0]
    light_type = loader.data['light_para']['light_type']

    z_of_len = loader.data["len_para"]["z_of_len"]
    z_tensor = torch.full((n, 1), z_of_len, dtype=torch.float32, device=device)
    origin_xyz_data = torch.cat((origin_xy_data, z_tensor), dim=1)  # (n, 3)

    # ============================================================
    # 1) 平行光：每条光线能量相同
    # ============================================================
    if light_type == "parallel_light":
        ray_energy = torch.full(
            (n,),
            float(total_energy) / float(n),
            dtype=torch.float32,
            device=device
        )
        return ray_energy

    # ============================================================
    # 2) 点光源：按微分立体角分配
    # ============================================================
    elif light_type == "point_light":
        point_light_coordinate = torch.tensor(
            loader.data['light_para']['point_light_coordinate'],
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)  # (1, 3)

        r = origin_xyz_data - point_light_coordinate
        R = torch.norm(r, dim=1)
        dz = torch.abs(r[:, 2])

        solid_angle_density = dz / (R ** 3) # (n,)

        weight_sum = torch.sum(solid_angle_density)
        ray_energy = float(total_energy) * solid_angle_density / weight_sum

        return ray_energy

def compute_normal(loader,input_direct,output_direct):
    input_direct = F.normalize(input_direct)
    output_direct = F.normalize(output_direct)
    refraction_index = loader.data["len_para"]["refraction_index"]
    normal_direct = input_direct - output_direct/refraction_index
    return normal_direct

def compute_proj(loader,input_direct,normal_direct,output_point):
    output_direct = compute_outside_refract_direct(loader,input_direct,normal_direct)
    z_of_receiver = torch.tensor(loader.data["len_para"]["z_of_receiver"], device=input_direct.device)

    time=((z_of_receiver - output_point[:,2]).squeeze(-1)/output_direct[:,2])
    x_new = output_point[:, 0] + output_direct[:, 0]*time
    y_new = output_point[:, 1] + output_direct[:, 1]*time
    
    return torch.stack([x_new, y_new], dim=1).to("cuda:0")

def compute_proj_mask(loader, input_direct, normal_direct, output_point, eps=1e-12):
    output_direct, valid_mask = compute_outside_refract_direct_mask(
        loader, input_direct, normal_direct, eps=eps
    )

    device = input_direct.device
    z_of_receiver = torch.tensor(
        loader.data["len_para"]["z_of_receiver"],
        dtype=output_point.dtype,
        device=device
    )

    safe_dz = torch.where(
        valid_mask,
        output_direct[:, 2],
        torch.ones_like(output_direct[:, 2])
    )

    time = (z_of_receiver - output_point[:, 2]) / safe_dz

    x_new = output_point[:, 0] + output_direct[:, 0] * time
    y_new = output_point[:, 1] + output_direct[:, 1] * time

    proj_xy = torch.stack([x_new, y_new], dim=1)

    proj_xy = torch.where(
        valid_mask.unsqueeze(1),
        proj_xy,
        torch.zeros_like(proj_xy)
    )

    return output_direct, proj_xy, valid_mask

def vectorized_light_point( loader,input_direct,normal_direct,output_point, expand_scalar, data_center, expand_center):
    data_scalar = loader.data['sample_point_para']['scalar']
    image_size = loader.data['picture_para']['size']

    output_direct = compute_outside_refract_direct(loader,input_direct,normal_direct)
    z_of_receiver = torch.tensor(loader.data["len_para"]["z_of_receiver"], device=input_direct.device)

    time=((z_of_receiver - output_point[:,2]).squeeze(-1)/output_direct[:,2])
    x_new = output_point[:, 0] + output_direct[:, 0]*time
    y_new = output_point[:, 1] + output_direct[:, 1]*time

    x_new = (x_new - expand_center[:,0]) / expand_scalar + data_center[0]
    y_new = (y_new - expand_center[:,1]) / expand_scalar + data_center[1]

    l_m = (x_new ) // (data_scalar / image_size)+1
    l_n = (y_new ) // (data_scalar / image_size)+1

    result = torch.cat([l_m.unsqueeze(1), l_n.unsqueeze(1)], dim=1)
    return result

def compute_triangle_area(loader, xy_coords, triangles):
        triangles_long = triangles.long()
        x1, y1 = xy_coords[triangles_long[:, 0], 0], xy_coords[triangles_long[:, 0], 1]
        x2, y2 = xy_coords[triangles_long[:, 1], 0], xy_coords[triangles_long[:, 1], 1]
        x3, y3 = xy_coords[triangles_long[:, 2], 0], xy_coords[triangles_long[:, 2], 1]
        area = 0.5 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        return area

def compute_triangle_energy(loader, xy_coords, triangles):
    sample_domain = loader.data["sample_point_para"]["sample_domain"]
    scalar_square = (sample_domain[1] - sample_domain[0]) * (sample_domain[3] - sample_domain[2])
    if(loader.data['light_para']['light_type'] == "parallel_light"):
        triangles_long = triangles.long()
        x1, y1 = xy_coords[triangles_long[:, 0], 0], xy_coords[triangles_long[:, 0], 1]
        x2, y2 = xy_coords[triangles_long[:, 1], 0], xy_coords[triangles_long[:, 1], 1]
        x3, y3 = xy_coords[triangles_long[:, 2], 0], xy_coords[triangles_long[:, 2], 1]
        area = 0.5 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        return area
    if(loader.data['light_para']['light_type'] == "point_light"):
        point_light_coordinate=torch.tensor(loader.data['light_para']['point_light_coordinate'], dtype=torch.float32).unsqueeze(0).to("cuda:0")
        z_of_len = loader.data["len_para"]["z_of_len"]
        z_of_len_tensor = torch.full((xy_coords.shape[0], 1), z_of_len).to("cuda:0")
        xy_coords_3d = torch.cat((xy_coords, z_of_len_tensor), dim=1)
        
        vectors = xy_coords_3d - point_light_coordinate
        vectors = F.normalize(vectors, p=2, dim=1) 

        triangles_long = triangles.long()
        v0 = vectors[triangles_long[:, 0]]
        v1 = vectors[triangles_long[:, 1]]
        v2 = vectors[triangles_long[:, 2]]

        cross_prod = torch.linalg.cross(v1, v2, dim=1)
        dot_prod = torch.sum(v0 * cross_prod, dim=1)
        
        denom = 1 + torch.sum(v0 * v1, dim=1) + torch.sum(v1 * v2, dim=1) + torch.sum(v2 * v0, dim=1)
        area = 2 * torch.atan2(dot_prod, denom)

        total_solid_angle = area.sum()
        area = area/total_solid_angle*scalar_square
        return area

def compute_reflection_loss(loader, distance, origin_xy_data, aim_xy_data):
    z_of_len = loader.data['len_para']['z_of_len']
    z_of_receiver = loader.data['len_para']['z_of_receiver']
    light_type = loader.data['light_para']['light_type']

    device = origin_xy_data.device
    n = origin_xy_data.shape[0]

    with torch.no_grad():
        z_len_tensor = torch.full((n, 1), z_of_len, device=device)
        z_receiver_tensor = torch.full((n, 1), z_of_receiver, device=device)
        origin_xyz_data = torch.cat((origin_xy_data, z_len_tensor), dim=1)
        aim_xyz_data = torch.cat((aim_xy_data, z_receiver_tensor), dim=1)

        if light_type == "point_light":
            vertical_direct = torch.zeros((n, 3), device=device)
            vertical_direct[:, 2] = 1
            inside_light_direct = compute_inside_light(loader, origin_xy_data, vertical_direct)
            
            point_light_coordinate = torch.tensor(
                loader.data['light_para']['point_light_coordinate'],
                dtype=torch.float32, device=device
            ).unsqueeze(0)
            input_light_direct = origin_xyz_data - point_light_coordinate

        elif light_type == "parallel_light":
            inside_light_direct = torch.zeros((n, 3), device=device)
            inside_light_direct[:, 2] = 1
            input_light_direct = torch.zeros((n, 3), device=device)
            input_light_direct[:, 2] = 1

        z_data = distance(origin_xy_data)
        output_point = origin_xyz_data + z_data * inside_light_direct
        output_light_direct = aim_xyz_data - output_point
        ri = loader.data["len_para"]["refraction_index"]

    with torch.no_grad():
        def normalize(v, eps=1e-12):
            return v / torch.clamp(torch.norm(v, dim=1, keepdim=True), min=eps)
        
        input_light_direct = normalize(input_light_direct)
        inside_light_direct = normalize(inside_light_direct)
        output_light_direct = normalize(output_light_direct)
        n1 = torch.zeros((n, 3), device=device)
        n1[:,2] = 1
        n2 = inside_light_direct * ri - output_light_direct
        n2 = normalize(n2)

        def fresnel_T_unpolarized(wi, wt, normal, eta_i, eta_t, eps=1e-12):

            # 不依赖法向朝向，直接取夹角绝对值
            cos_i = torch.clamp(torch.abs(torch.sum(wi * normal, dim=1, keepdim=True)), 0.0, 1.0)
            cos_t = torch.clamp(torch.abs(torch.sum(wt * normal, dim=1, keepdim=True)), 0.0, 1.0)

            Rs = ((eta_i * cos_i - eta_t * cos_t) /
                  torch.clamp(eta_i * cos_i + eta_t * cos_t, min=eps)) ** 2

            Rp = ((eta_i * cos_t - eta_t * cos_i) /
                  torch.clamp(eta_i * cos_t + eta_t * cos_i, min=eps)) ** 2

            R = 0.5 * (Rs + Rp)
            T = 1.0 - R
            return T

        T1 = fresnel_T_unpolarized(
            wi=input_light_direct,
            wt=inside_light_direct,
            normal=n1,
            eta_i=1.0,
            eta_t=ri
        )

        T2 = fresnel_T_unpolarized(
            wi=inside_light_direct,
            wt=output_light_direct,
            normal=n2,
            eta_i=ri,
            eta_t=1.0
        )   # (N, 1)

        # 两次折射后的总剩余比例
        remain_ratio = (T1 * T2).detach()

    return remain_ratio

def compute_rot(loader, distance, origin_xy_data, aim_xy_data, triangles_tensor):
    # 参数设置
    z_of_len = loader.data['len_para']['z_of_len']
    z_of_receiver = loader.data['len_para']['z_of_receiver']
    light_type = loader.data['light_para']['light_type']

    device = origin_xy_data.device
    n = origin_xy_data.shape[0]

    # 拼接成 xyz 坐标
    z_len_tensor = torch.full((n, 1), z_of_len, device=device)
    z_receiver_tensor = torch.full((n, 1), z_of_receiver, device=device)
    origin_xyz_data = torch.cat((origin_xy_data, z_len_tensor), dim=1)
    aim_xyz_data = torch.cat((aim_xy_data, z_receiver_tensor), dim=1)

    # 计算光线方向
    if light_type == "point_light":
        vertical_direct = torch.zeros((n, 3), device=device)
        vertical_direct[:, 2] = 1
        inside_light_direct, pq_grad = compute_inside_refraction_jacobian(loader, origin_xy_data, vertical_direct)

    elif light_type == "parallel_light":
        inside_light_direct = torch.zeros((n, 3), device=device)
        inside_light_direct[:, 2] = 1
        pq_grad = torch.zeros((n, 4), device=device)

    z_data = distance(origin_xy_data)  # requires_grad = True
    output_point = origin_xyz_data + z_data * inside_light_direct
    output_light_direct = aim_xyz_data - output_point

    # 法向 + 光线方向
    normal_direct = compute_normal(loader, inside_light_direct, output_light_direct)
    diff_scalar = torch.sum(inside_light_direct * normal_direct, dim=1, keepdim=True)

    t1 = (1 + pq_grad[:, 0].unsqueeze(1) * z_data)
    t2 = (pq_grad[:, 2].unsqueeze(1) * z_data)
    t3 = (pq_grad[:, 1].unsqueeze(1) * z_data)
    t4 = (1 + pq_grad[:, 3].unsqueeze(1) * z_data)

    real_function_diff = torch.cat([
        t1 * normal_direct[:, 0:1] + t2 * normal_direct[:, 1:2],
        t3 * normal_direct[:, 0:1] + t4 * normal_direct[:, 1:2]
    ], dim=1) / diff_scalar

    triangle_xy_data = origin_xy_data[triangles_tensor]
    triangle_normals = real_function_diff[triangles_tensor]

    base_point = triangle_xy_data[:, 0, :]
    delta_x2 = triangle_xy_data[:, 1, 0] - base_point[:, 0]
    delta_x3 = triangle_xy_data[:, 2, 0] - base_point[:, 0]
    delta_y2 = triangle_xy_data[:, 1, 1] - base_point[:, 1]
    delta_y3 = triangle_xy_data[:, 2, 1] - base_point[:, 1]
    triangle_deltas = torch.stack([delta_x2, delta_x3, delta_y2, delta_y3], dim=1)

    base_normal = triangle_normals[:, 0, :]
    delta_nx2 = triangle_normals[:, 1, 0] - base_normal[:, 0]
    delta_nx3 = triangle_normals[:, 2, 0] - base_normal[:, 0]
    delta_ny2 = triangle_normals[:, 1, 1] - base_normal[:, 1]
    delta_ny3 = triangle_normals[:, 2, 1] - base_normal[:, 1]
    normal_deltas = torch.stack([delta_nx2, delta_nx3, delta_ny2, delta_ny3], dim=1)

    n1_y = triangle_deltas[:, 0] * normal_deltas[:, 1] - triangle_deltas[:, 1] * normal_deltas[:, 0]
    n2_x = triangle_deltas[:, 3] * normal_deltas[:, 2] - triangle_deltas[:, 2] * normal_deltas[:, 3]
    rotation = torch.abs(n1_y - n2_x)

    return rotation

def display_color(mesh_folder, mesh_xiushi, distance, origin_xy_data, aim_xy_data, triangles_tensor, tri_color):
    z_data = distance(origin_xy_data).detach()
    xyz_data = torch.cat([origin_xy_data, torch.zeros(aim_xy_data.shape[0],1, device='cuda')], dim=1)
    aim_xyz_data = torch.cat([aim_xy_data, torch.zeros(aim_xy_data.shape[0],1, device='cuda')], dim=1)

    aim_points = aim_xyz_data.detach().cpu().numpy()        # 形状 (N,3)
    points = xyz_data.detach().cpu().numpy()        # 形状 (N,3)
    faces = triangles_tensor.detach().cpu().numpy() # 形状 (M,3)
    colors = tri_color.detach().cpu().numpy()         # 形状 (M,3)

    colors_float = colors.astype(np.float32) / 255.0 
    colors_rounded = np.round(colors_float, decimals=6)
    unique_colors, color_indices = np.unique(colors_rounded, axis=0, return_inverse=True)

    mtl_filename = os.path.join(mesh_folder, "rotation_"+mesh_xiushi+".mtl")
    with open(mtl_filename, 'w') as mtl_file:
        for i, color in enumerate(unique_colors):
            mtl_file.write(f"newmtl material_{i}\n")
            mtl_file.write(f"Kd {color[0]:f} {color[1]:f} {color[2]:f}\n")

    obj_filename = os.path.join(mesh_folder, "rotation_origin_"+mesh_xiushi+".obj")
    with open(obj_filename, 'w') as obj_file:
        obj_file.write(f"mtllib rotation_"+mesh_xiushi+".mtl\n")
        
        for point in points:
            obj_file.write(f"v {point[0]:f} {point[1]:f} {point[2]:f}\n")
        
        combined = list(zip(color_indices, faces))
        combined.sort(key=lambda x: x[0])
    
        for key, group in groupby(combined, key=lambda x: x[0]):
            mtl_name = f"material_{key}"
            obj_file.write(f"usemtl {mtl_name}\n")
            for item in group:
                face = item[1]
                f_line = "f {} {} {}\n".format(face[0]+1, face[1]+1, face[2]+1)
                obj_file.write(f_line)

    obj_filename = os.path.join(mesh_folder, "rotation_proj_"+mesh_xiushi+".obj")
    with open(obj_filename, 'w') as obj_file:
        obj_file.write(f"mtllib rotation_"+mesh_xiushi+".mtl\n")
        
        for point in aim_points:
            obj_file.write(f"v {point[0]:f} {point[1]:f} {point[2]:f}\n")
        
        combined = list(zip(color_indices, faces))
        combined.sort(key=lambda x: x[0])
        
        for key, group in groupby(combined, key=lambda x: x[0]):
            mtl_name = f"material_{key}"
            obj_file.write(f"usemtl {mtl_name}\n")
            for item in group:
                face = item[1]
                f_line = "f {} {} {}\n".format(face[0]+1, face[1]+1, face[2]+1)
                obj_file.write(f_line)

def clear_folder(foldername):
    for filename in os.listdir(foldername):
        file_path = os.path.join(foldername, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'删除 {file_path} 失败. 原因: {e}')

def normalize(v, eps=1e-12):
    return v / torch.clamp(torch.norm(v, dim=1, keepdim=True), min=eps)

def fresnel_T_unpolarized(wi, wt, normal, eta_i, eta_t, eps=1e-12):

    # 不依赖法向朝向，直接取夹角绝对值
    cos_i = torch.clamp(torch.abs(torch.sum(wi * normal, dim=1, keepdim=True)), 0.0, 1.0)
    cos_t = torch.clamp(torch.abs(torch.sum(wt * normal, dim=1, keepdim=True)), 0.0, 1.0)

    Rs = ((eta_i * cos_i - eta_t * cos_t) /
            torch.clamp(eta_i * cos_i + eta_t * cos_t, min=eps)) ** 2

    Rp = ((eta_i * cos_t - eta_t * cos_i) /
            torch.clamp(eta_i * cos_t + eta_t * cos_i, min=eps)) ** 2

    R = 0.5 * (Rs + Rp)
    T = 1.0 - R
    return T.squeeze(1)

def sample_triangle_energy_centers(loader, xy_coords, triangles, n, total_gray_value_0):
    device = xy_coords.device
    dtype = xy_coords.dtype

    triangles_long = triangles.long()
    face_vertices = xy_coords[triangles_long]  # [F, 3, 2]
    num_faces = triangles_long.shape[0]

    area_bary_tri_list = []

    index_map = {}
    bary_points = []

    idx = 0
    for i in range(n + 1):
        for j in range(n + 1 - i):
            bary = [1.0 - i / n - j / n,i / n,j / n]
            bary_points.append(bary)
            index_map[(i, j)] = idx
            idx += 1

    for i in range(n):
        for j in range(n - i):
            a = index_map[(i, j)]
            b = index_map[(i + 1, j)]
            c = index_map[(i, j + 1)]

            area_bary_tri_list.append([a, b, c])

            if i + j < n - 1:
                d = index_map[(i + 1, j + 1)]
                area_bary_tri_list.append([b, d, c])

    bary_points = torch.tensor(bary_points, dtype=dtype, device=device)
    local_triangles = torch.tensor(area_bary_tri_list, dtype=torch.long, device=device)

    sub_points = torch.einsum("pk,fkd->fpd", bary_points, face_vertices)

    points_per_face = bary_points.shape[0]
    face_offsets = torch.arange(num_faces, device=device).long()[:, None, None] * points_per_face

    sub_triangles = (local_triangles[None, :, :] + face_offsets).reshape(-1, 3).int()
    sub_xy = sub_points.reshape(-1, 2)

    sub_triangles_long = sub_triangles.long()
    center_xy = (
        sub_xy[sub_triangles_long[:, 0]] +
        sub_xy[sub_triangles_long[:, 1]] +
        sub_xy[sub_triangles_long[:, 2]]
    ) / 3.0

    with torch.no_grad():
        sub_area = compute_triangle_energy( loader, sub_xy.double(), sub_triangles)
        sub_area = sub_area.to(dtype)
        min_val, min_idx = torch.min(sub_area, dim=0)
        energy = sub_area / sub_area.sum()
    return center_xy, energy

def compute_mean_curvature(heights, grads, grads_2nd, render_inside_light_direct, render_pq_grad):
    z = heights
    z_x = grads[:, 0:1]
    z_y = grads[:, 1:2]
    z_xx = grads_2nd[:, 0:1]
    z_xy = grads_2nd[:, 1:2]
    z_yy = grads_2nd[:, 2:3]

    p = render_inside_light_direct[:, 0:1]
    q = render_inside_light_direct[:, 1:2]
    p_x = render_pq_grad[:, 0:1]
    p_y = render_pq_grad[:, 1:2]
    q_x = render_pq_grad[:, 2:3]
    q_y = render_pq_grad[:, 3:4]

    qx = torch.cat([
        1 + p_x * z + p * z_x,
        q_x * z + q * z_x,
        z_x
    ], dim=1)

    qy = torch.cat([
        p_y * z + p * z_y,
        1 + q_y * z + q * z_y,
        z_y
    ], dim=1)

    qxx = torch.cat([
        2 * p_x * z_x + p * z_xx,
        2 * q_x * z_x + q * z_xx,
        z_xx
    ], dim=1)

    qxy = torch.cat([
        p_y * z_x + p_x * z_y + p * z_xy,
        q_y * z_x + q_x * z_y + q * z_xy,
        z_xy
    ], dim=1)

    qyy = torch.cat([
        2 * p_y * z_y + p * z_yy,
        2 * q_y * z_y + q * z_yy,
        z_yy
    ], dim=1)

    E = (qx * qx).sum(dim=1, keepdim=True)
    F = (qx * qy).sum(dim=1, keepdim=True)
    G = (qy * qy).sum(dim=1, keepdim=True)

    W2 = E * G - F * F
    W = torch.sqrt(W2 + 1e-12)

    def triple(a, b, c):
        cross_bc = torch.cross(b, c, dim=1)
        return (a * cross_bc).sum(dim=1, keepdim=True)

    L = triple(qxx, qx, qy) / W
    M = triple(qxy, qx, qy) / W
    N = triple(qyy, qx, qy) / W

    H = (E * N - 2 * F * M + G * L) / (2 * W2 + 1e-12)
    H_abs = H.abs()
    avg_H_abs = H_abs.mean()

    return H, H_abs, avg_H_abs

def _gaussian_splat_raw_chunk(points, energy, pic_size, proj_domain, sigma=0.5, radius=2):
    device, dtype = points.device, points.dtype
    res_y, res_x = int(pic_size[0]), int(pic_size[1])
    x_min, x_max, y_min, y_max = proj_domain

    scale_x, scale_y = (x_max - x_min) / res_x, (y_max - y_min) / res_y
    px, py = (points[:, 0] - x_min) / scale_x, (points[:, 1] - y_min) / scale_y

    padded_res_y, padded_res_x = res_y + 2 * radius, res_x + 2 * radius
    chunk_ix = torch.floor(px).long() + radius
    chunk_iy = torch.floor(py).long() + radius

    offsets = torch.arange(-radius, radius + 1, device=device, dtype=torch.long)
    dy_grid, dx_grid = torch.meshgrid(offsets, offsets, indexing="ij")
    dx_flat, dy_flat = dx_grid.reshape(-1), dy_grid.reshape(-1)

    xx = chunk_ix[:, None] + dx_flat[None, :]
    yy = chunk_iy[:, None] + dy_flat[None, :]

    px_pad, py_pad = px + radius, py + radius
    dx_real = xx.to(dtype) + 0.5 - px_pad[:, None]
    dy_real = yy.to(dtype) + 0.5 - py_pad[:, None]

    weights = torch.exp(-(dx_real.square() + dy_real.square()) / (2.0 * sigma * sigma))
    weights = weights / weights.sum(dim=1, keepdim=True).detach().clamp_min(1e-12)
    local_contrib = weights * energy[:, None]

    linear_idx = yy * padded_res_x + xx
    valid_splat = (xx >= 0) & (xx < padded_res_x) & (yy >= 0) & (yy < padded_res_y)

    padded_img_flat = torch.zeros(padded_res_y * padded_res_x, device=device, dtype=dtype)
    padded_img_flat = padded_img_flat.scatter_add(
        0,
        linear_idx[valid_splat].reshape(-1),
        local_contrib[valid_splat].reshape(-1)
    )

    padded_img = padded_img_flat.view(padded_res_y, padded_res_x)
    return padded_img[radius:radius + res_y, radius:radius + res_x]

def render_gaussian_splat_two_pass_backward(loader, total_gray_value, valid_points, ray_energy, pic_size, proj_domain, target_image, target_rms, sigma=0.5, radius=2, chunk_size=300000):
    device, dtype = valid_points.device, valid_points.dtype
    res_y, res_x = int(pic_size[0]), int(pic_size[1])
    x_min, x_max, y_min, y_max = proj_domain

    # 只在生成 mask 时关闭梯度，不复制完整输入数据
    with torch.no_grad():
        legal_mask = torch.isfinite(valid_points).all(dim=1) & torch.isfinite(ray_energy)
        render_mask = legal_mask & (valid_points[:, 0] >= x_min) & (valid_points[:, 0] < x_max) & (valid_points[:, 1] >= y_min) & (valid_points[:, 1] < y_max)
        valid_idx = torch.nonzero(render_mask, as_tuple=False).squeeze(1)
        total_energy = ray_energy[legal_mask].sum()

    n = valid_idx.numel()
    gray_value = torch.as_tensor(total_gray_value, device=device, dtype=dtype).detach()
    target_rms_value = torch.as_tensor(target_rms, device=device, dtype=dtype).detach().clamp_min(1e-12)
    norm_factor = gray_value / total_energy

    with torch.no_grad():
        raw_full_image = torch.zeros((res_y, res_x), device=device, dtype=dtype)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_idx = valid_idx[start:end]
            raw_chunk = _gaussian_splat_raw_chunk(valid_points[chunk_idx], ray_energy[chunk_idx], pic_size, proj_domain, sigma, radius)
            raw_full_image.add_(raw_chunk)
        rendered_image = raw_full_image * norm_factor

    # 根据完整图像计算 dLoss/dImage
    image_leaf = rendered_image.detach().requires_grad_(True)
    loss1_proxy = torch.sqrt(F.mse_loss(image_leaf, target_image.detach(), reduction="mean")) / target_rms_value
    image_grad = torch.autograd.grad(loss1_proxy, image_leaf, create_graph=False, retain_graph=False)[0].detach()
    loss1_value = loss1_proxy.detach()

    del image_leaf, loss1_proxy

    # 第二遍：直接从原始可求导 valid_points 中按 chunk 获取数据
    if n > 0 and valid_points.requires_grad:
        grad_points_all = torch.zeros_like(valid_points)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_idx = valid_idx[start:end]
            points_chunk = valid_points[chunk_idx]

            # 当前流程不优化 ray_energy，所以将其视为常量
            energy_chunk = ray_energy[chunk_idx].detach()

            raw_chunk = _gaussian_splat_raw_chunk(points_chunk, energy_chunk, pic_size, proj_domain, sigma, radius)
            chunk_vjp = torch.sum(raw_chunk * norm_factor * image_grad)

            grad_chunk = torch.autograd.grad(chunk_vjp, points_chunk, create_graph=False, retain_graph=False)[0]
            grad_points_all[chunk_idx] = grad_chunk.detach()

            del points_chunk, energy_chunk, raw_chunk, chunk_vjp, grad_chunk

        # 所有 Gaussian chunk 完成后，只通过 valid_points 向上游反传一次
        torch.autograd.backward(valid_points, grad_points_all)
        del grad_points_all

    return rendered_image, loss1_value

def define_parameter(ot_seed_path, points, proj_points, left_idx, right_idx, bottom_idx, top_idx):
    np.savetxt(ot_seed_path + "points.txt", points, fmt="%f")
    np.savetxt(ot_seed_path + "proj_xy.txt", proj_points, fmt="%f")
    np.savetxt(ot_seed_path + "left_idx.txt", left_idx, fmt="%d")
    np.savetxt(ot_seed_path + "right_idx.txt", right_idx, fmt="%d")
    np.savetxt(ot_seed_path + "bottom_idx.txt", bottom_idx, fmt="%d")
    np.savetxt(ot_seed_path + "top_idx.txt", top_idx, fmt="%d")

def check_each_loss_grad(render_loss, rot_loss, small_area_loss, big_area_loss, proj_area_loss, edge_loss, opt_xy_data, aim_xy_data, full_origin_xy, triangles_tensor, data_foldername, iter_xiushi, iter_step):
    loss_list = [("render_loss", render_loss), ("rot_loss", rot_loss), ("small_area_loss", small_area_loss), ("big_area_loss", big_area_loss), ("proj_area_loss", proj_area_loss), ("edge_loss", edge_loss)]

    for loss_name, loss_value in loss_list:
        if not torch.is_tensor(loss_value) or not loss_value.requires_grad:
            continue

        try:
            grad_opt, grad_aim = torch.autograd.grad(loss_value, [opt_xy_data, aim_xy_data], retain_graph=True, create_graph=False, allow_unused=True)
        except RuntimeError as e:
            print("\n================ gradient error ================")
            print("iter_step =", iter_step)
            print("loss =", loss_name)
            print("autograd.grad error =", e)
            print("保存当前尚未发生错误更新的点数据")
            print("================================================\n")
            with torch.no_grad():
                data_utils.data_save(full_origin_xy.detach(), aim_xy_data.detach(), triangles_tensor, data_foldername, iter_xiushi)
            raise RuntimeError(f"{loss_name} 求导发生错误，优化停止")

        if grad_opt is not None and not torch.isfinite(grad_opt).all():
            nan_num = torch.isnan(grad_opt).sum().item()
            inf_num = torch.isinf(grad_opt).sum().item()
            bad_idx = (~torch.isfinite(grad_opt)).nonzero(as_tuple=False)
            print("\n================ gradient NaN/Inf ================")
            print("iter_step =", iter_step)
            print("problem loss =", loss_name)
            print("problem grad = opt_xy_data.grad")
            print("nan_num =", nan_num)
            print("inf_num =", inf_num)
            if bad_idx.numel() > 0:
                print("first_bad_index =", bad_idx[0].detach().cpu().tolist())
            print("保存当前尚未发生错误更新的点数据")
            print("==================================================\n")
            with torch.no_grad():
                data_utils.data_save(full_origin_xy.detach(), aim_xy_data.detach(), triangles_tensor, data_foldername, iter_xiushi)
            raise RuntimeError(f"{loss_name} 对 opt_xy_data 的梯度出现 NaN/Inf，优化停止")

        if grad_aim is not None and not torch.isfinite(grad_aim).all():
            nan_num = torch.isnan(grad_aim).sum().item()
            inf_num = torch.isinf(grad_aim).sum().item()
            bad_idx = (~torch.isfinite(grad_aim)).nonzero(as_tuple=False)
            print("\n================ gradient NaN/Inf ================")
            print("iter_step =", iter_step)
            print("problem loss =", loss_name)
            print("problem grad = aim_xy_data.grad")
            print("nan_num =", nan_num)
            print("inf_num =", inf_num)
            if bad_idx.numel() > 0:
                print("first_bad_index =", bad_idx[0].detach().cpu().tolist())
            print("保存当前尚未发生错误更新的点数据")
            print("==================================================\n")
            with torch.no_grad():
                data_utils.data_save(full_origin_xy.detach(), aim_xy_data.detach(), triangles_tensor, data_foldername, iter_xiushi)
            raise RuntimeError(f"{loss_name} 对 aim_xy_data 的梯度出现 NaN/Inf，优化停止")