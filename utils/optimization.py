import os
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import subprocess
import utils.basic_function as basic_function
import utils.data_utils as data_utils
import irra_render
import copy

def get_physical_gpu_id(logical_device_id=None):
    """
    logical_device_id: PyTorch 中的 cuda 逻辑编号，例如 cuda:0 就是 0
    返回 nvidia-smi 应该查询的物理 GPU 编号或 UUID
    """
    if logical_device_id is None:
        logical_device_id = torch.cuda.current_device()

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)

    if cuda_visible_devices is None or cuda_visible_devices.strip() == "":
        return str(logical_device_id)

    visible_list = [x.strip() for x in cuda_visible_devices.split(",")]

    if logical_device_id >= len(visible_list):
        raise RuntimeError(
            f"logical_device_id={logical_device_id} 超出 CUDA_VISIBLE_DEVICES 范围: {cuda_visible_devices}"
        )

    return visible_list[logical_device_id]


def get_gpu_status(logical_device_id=None):
    """
    查询当前 PyTorch 使用的 logical GPU 对应的 nvidia-smi 状态
    """
    physical_gpu_id = get_physical_gpu_id(logical_device_id)

    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={physical_gpu_id}",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits"
            ],
            encoding="utf-8"
        )

        gpu_index, gpu_util, mem_used, mem_total = result.strip().split(", ")

        return {
            "physical_gpu_id": physical_gpu_id,
            "nvidia_smi_index": gpu_index,
            "gpu_util": float(gpu_util),
            "mem_used": float(mem_used),
            "mem_total": float(mem_total),
        }

    except Exception as e:
        print(f"Failed to query GPU status: {e}")
        return None
        
def save_tensor_txt(tensor, save_path, fmt="%.8e"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if isinstance(tensor, torch.Tensor):
        data = tensor.detach().cpu().numpy()
    else:
        data = np.asarray(tensor)

    # 1D / 2D 直接保存
    if data.ndim <= 2:
        np.savetxt(save_path, data, fmt=fmt)

    else:
        data_reshape = data.reshape(data.shape[0], -1)
        np.savetxt(save_path, data_reshape, fmt=fmt)

def reset_adam_state_keep_lr(optimizer):
    # 记录每个 param_group 的 lr（以及你可能还关心的 betas/eps）
    lrs = [pg["lr"] for pg in optimizer.param_groups]
    betas = [pg["betas"] for pg in optimizer.param_groups]
    epss = [pg["eps"] for pg in optimizer.param_groups]

    # 清空状态：m, v, step 全部归零
    optimizer.state.clear()

    for i, pg in enumerate(optimizer.param_groups):
        pg["lr"] = lrs[i]
        pg["betas"] = betas[i]
        pg["eps"] = epss[i]

def compute_zeta(loader,proj_xy_tensor_0,triangles_tensor,min_threshold,max_threshold):
    max_area_ratio = loader.data['inte_opti_para']['max_area_ratio']

    with torch.no_grad():
        aim_tri_area = basic_function.compute_triangle_energy(
            loader,
            proj_xy_tensor_0,
            triangles_tensor
        )

        zeta = aim_tri_area * max_area_ratio

        zeta = torch.clamp(
            zeta,
            min=min_threshold,
            max=max_threshold
        )

        return zeta.detach()

def initial_optimization(loader,aim_xy_data,triangles_tensor):
    max_area_ratio = loader.data['inte_opti_para']['max_area_ratio']
    min_area_ratio = loader.data['inte_opti_para']['min_area_ratio']
    aim_tri_area = basic_function.compute_triangle_energy(loader, aim_xy_data, triangles_tensor)
    mean_area = aim_tri_area.mean().detach()
    max_threshold = mean_area * max_area_ratio
    min_threshold = mean_area * min_area_ratio
    print(max_threshold,min_threshold)

    optimizer_aim_xy_data = torch.optim.Adam([aim_xy_data],lr=loader.data['inte_opti_para']['pre_iter_lr'])

    pre_end_iter = loader.data['inte_opti_para']['pre_end_iter']

    for it in range(pre_end_iter):
        optimizer_aim_xy_data.zero_grad()
        aim_tri_area = basic_function.compute_triangle_area(loader,aim_xy_data,triangles_tensor)
        flip_tri_mask = aim_tri_area < min_threshold
        
        if not flip_tri_mask.any().item():
            print(f"min_area={aim_tri_area.min().item():.6e},Early stop at iter {it}")
            break

        area_loss = (max_threshold - aim_tri_area[flip_tri_mask]).pow(2).mean()
        loss = 1e6 * area_loss

        if it % 500 == 0:
            print(
                f"{it} "
                f"loss={loss.item():.6e} "
                f"flip={(aim_tri_area < 0).sum().item()} "
                f"small={(aim_tri_area < min_threshold).sum().item()} "
                f"min_area={aim_tri_area.min().item():.6e}"
            )

        loss.backward()
        optimizer_aim_xy_data.step()

    return aim_xy_data,max_threshold,min_threshold

def render_inte_optimization(loader,distance,origin_xy_data,aim_xy_data,triangles_tensor,total_gray_value,pic_size,tensor_image,
                             origin_average_rot_max,inte_end_iter,inte_origin_learn_rate,inte_aim_learn_rate,optimization_time,origin_tri_area_0,zeta_1,max_gray):
    
    for param in distance.parameters():
        param.requires_grad = False

    print("渲染旋度优化已开始")
    origin_rot_inte = 0
    total_para = loader.data['inte_opti_para']['total_para']
    para_render_1 = loader.data['inte_opti_para']['inte_para_render_1'] * total_para
    para_render_2 = loader.data['inte_opti_para']['inte_para_render_2'] * total_para
    if_use_sobel = torch.tensor(loader.data["inte_opti_para"]["if_use_sobel"], dtype=torch.bool).to("cuda:0")
    para_sobel_1 = loader.data['inte_opti_para']['inte_para_sobel_1'] * total_para
    para_sobel_2 = loader.data['inte_opti_para']['inte_para_sobel_2'] * total_para
    if_use_inte = torch.tensor(loader.data['inte_opti_para']['if_use_inte'], dtype=torch.bool).to("cuda:0")
    para_inte_1 = loader.data['inte_opti_para']['inte_para_inte_1'] * total_para
    para_inte_2 = loader.data['inte_opti_para']['inte_para_inte_2'] * total_para
    para_small_area = loader.data['inte_opti_para']['inte_para_small_area'] * total_para
    para_big_area = loader.data['inte_opti_para']['inte_para_big_area'] * total_para
    para_proj_area = loader.data['inte_opti_para']['inte_para_proj_area'] * total_para
    para_edge = loader.data['inte_opti_para']['inte_para_edge'] * total_para
    base_para_rot = loader.data['inte_opti_para']['inte_para_rot'] * total_para
    if_rot_para_expand = torch.tensor(loader.data['inte_opti_para']['if_rot_para_expand'], dtype=torch.bool).to("cuda:0")
    para_rot_expand = loader.data['inte_opti_para']['inte_para_rot_expand']

    # 跨轮次累积扩张: optimization_time=i 时乘 para_rot_expand[0..i-1]
    if if_rot_para_expand:
        expand_cum = 1.0
        if isinstance(para_rot_expand, list):
            for i in range(min(optimization_time, len(para_rot_expand))):
                expand_cum *= para_rot_expand[i]
        else:
            expand_cum = para_rot_expand ** (optimization_time + 1)
        base_para_rot *= expand_cum
    
    epsilon_1 = torch.tensor(loader.data['inte_opti_para']['inte_epsilon_1'], dtype=torch.float32).to("cuda:0")
    epsilon_2 = torch.tensor(loader.data['inte_opti_para']['inte_epsilon_2'], dtype=torch.float32).to("cuda:0")
    theta_1 = torch.tensor(loader.data['inte_opti_para']['inte_theta_1'], dtype=torch.float32).to("cuda:0")
    theta_2 = torch.tensor(loader.data['inte_opti_para']['inte_theta_2'], dtype=torch.float32).to("cuda:0")
    delta_1 = torch.tensor(loader.data['inte_opti_para']['inte_delta_1'], dtype=torch.float32).to("cuda:0")
    delta_2 = torch.tensor(loader.data['inte_opti_para']['inte_delta_2'], dtype=torch.float32).to("cuda:0")
    if_use_picture_residual = torch.tensor(loader.data["inte_opti_para"]["if_use_picture_residual"], dtype=torch.bool).to("cuda:0")
    picture_residual_para = torch.tensor(loader.data["inte_opti_para"]["picture_residual_para"], dtype=torch.float32).to("cuda:0")
    
    # 投影面积para
    sample_domain = torch.tensor(loader.data["sample_point_para"]["sample_domain"], dtype=torch.float32).to("cuda:0")
    proj_domain = torch.tensor(loader.data["len_para"]["proj_domain"], dtype=torch.float32).to("cuda:0")
    triangles_num = triangles_tensor.shape[0]

    x_mask = ((origin_xy_data[:, 0] == sample_domain[0]) | (origin_xy_data[:, 0] == sample_domain[1]))
    y_mask = ((origin_xy_data[:, 1] == sample_domain[2]) | (origin_xy_data[:, 1] == sample_domain[3]))
    fixed_mask = torch.stack([x_mask, y_mask], axis=1)

    opt_xy_data = torch.nn.Parameter(origin_xy_data[~fixed_mask].clone(), requires_grad=True)
    optimizer_xy_data = torch.optim.Adam([opt_xy_data], lr=inte_origin_learn_rate)
    optimizer_aim_xy_data = torch.optim.Adam([aim_xy_data], lr=inte_aim_learn_rate)

    #渲染尺度
    inte_delta = loader.data['inte_opti_para']['inte_delta']
    render_scalar = torch.tensor(pic_size[0]*pic_size[1]*max_gray*max_gray,device="cuda:0")
    # 旋度尺度
    rot_scalar = (sample_domain[1] - sample_domain[0])*(sample_domain[3] - sample_domain[2])
    # 面积约束
    epsilon_1 = epsilon_1*origin_tri_area_0
    epsilon_2 = epsilon_2*origin_tri_area_0
    theta_1 = theta_1*origin_tri_area_0
    theta_2 = theta_2*origin_tri_area_0
    
    origin_tri_energy = basic_function.compute_triangle_energy(loader, origin_xy_data, triangles_tensor)

    remain_ratio = basic_function.compute_reflection_loss(loader, distance, origin_xy_data, aim_xy_data)
    if remain_ratio.dim() == 2 and remain_ratio.shape[1] == 1:
        remain_ratio = remain_ratio[:, 0]
    tri_ratio = remain_ratio[triangles_tensor].mean(dim=1, keepdim=True).detach()

    if loader.data['inte_opti_para']['if_consider_reflection_loss'] == True:
        equal_tri_energy = origin_tri_energy * tri_ratio[:, 0]
        with torch.no_grad():
            total_energy = equal_tri_energy.sum().detach()
            average_gray_value = total_gray_value/total_energy
    else:
        total_energy = origin_tri_energy.sum().detach()
        average_gray_value = total_gray_value/total_energy
        equal_tri_energy = origin_tri_energy

    if optimization_time==0:
        inte_start_iter = 0
    else:
        inte_start_iter = 0

    for iter_step in range(inte_start_iter,inte_end_iter):
        if iter_step == 0:
            mesh_foldername = loader.data['data_mesh_folder']+f"{optimization_time}/"
            picture_foldername = loader.data['data_render_folder']+f"{optimization_time}/"
            data_foldername = loader.data['data_point_folder']+f"{optimization_time}/"        
            os.makedirs(picture_foldername, exist_ok=True)
            os.makedirs(data_foldername, exist_ok=True)
            basic_function.clear_folder(picture_foldername)
            basic_function.clear_folder(data_foldername)

        iter_xiushi = str(iter_step)
        full_origin_xy = origin_xy_data.clone()
        full_origin_xy[~fixed_mask] = opt_xy_data

        origin_tri_energy = basic_function.compute_triangle_energy(loader, full_origin_xy, triangles_tensor)

        if iter_step % 1 == 0:
            remain_ratio = basic_function.compute_reflection_loss(loader, distance, full_origin_xy, aim_xy_data)
            if remain_ratio.dim() == 2 and remain_ratio.shape[1] == 1:
                remain_ratio = remain_ratio[:, 0]
            tri_ratio = remain_ratio[triangles_tensor].mean(dim=1, keepdim=True).detach()

        if loader.data['inte_opti_para']['if_consider_reflection_loss'] == True:
            equal_tri_energy_coarse = origin_tri_energy * tri_ratio[:, 0]
            if (equal_tri_energy_coarse < 0).any().item():
                print("equal_tri_energy_coarse 存在负值, min =",equal_tri_energy_coarse.min().item())
            equal_tri_energy = equal_tri_energy_coarse.clamp_min(0.0)

            with torch.no_grad():
                total_energy = equal_tri_energy.sum().detach()
                average_gray_value = total_gray_value/total_energy
        else:
            total_energy = origin_tri_energy.sum().detach()
            average_gray_value = total_gray_value/total_energy
            equal_tri_energy = origin_tri_energy

        render_grad = irra_render.render_grad( equal_tri_energy, triangles_tensor, aim_xy_data, average_gray_value, tensor_image,
                                            if_use_sobel, proj_domain, pic_size)

        c_proj_xy_grad = render_grad[2].detach()
        c_tri_area_grad = render_grad[3].detach()
        
        c_tri_area_inte_grad = -2 * average_gray_value * render_grad[4].detach()
        c_proj_xy_inte_grad = render_grad[5].detach()

        # 投影面积loss
        aim_tri_area = basic_function.compute_triangle_area(loader, aim_xy_data, triangles_tensor)
        if iter_step<10:
            print(aim_tri_area.min())
        proj_mask = aim_tri_area < zeta_1
        proj_area_loss = ((0 - zeta_1[proj_mask])/(0 - aim_tri_area[proj_mask]) - 1)**2
        proj_area_loss = proj_area_loss.sum()

        # 入射面积loss
        origin_tri_area = basic_function.compute_triangle_area(loader, full_origin_xy, triangles_tensor)
        small_mask = origin_tri_area < epsilon_1
        small_area_loss = ((epsilon_1[small_mask] - epsilon_2[small_mask])/(origin_tri_area[small_mask] - epsilon_2[small_mask]) - 1)**2
        small_area_loss = (small_area_loss).sum()
        big_mask = origin_tri_area > theta_1
        big_area_loss = ((theta_2[big_mask] - theta_1[big_mask])/(theta_2[big_mask] - origin_tri_area[big_mask]) - 1)**2
        big_area_loss = (big_area_loss).sum()
        small_area_loss = para_small_area * small_area_loss / triangles_num
        big_area_loss = para_big_area * big_area_loss / triangles_num
        proj_area_loss = para_proj_area * proj_area_loss / triangles_num

        # 旋度loss (L2 退火: 先慢后快)
        origin_rot = basic_function.compute_rot(loader,distance,full_origin_xy,aim_xy_data,triangles_tensor)
        rot_loss = origin_rot.pow(2).sum()
        t = iter_step / (inte_end_iter - 1)
        anneal = t * t
        current_expand_para = ((1-anneal) + (anneal)*para_rot_expand[optimization_time])
        rot_loss = base_para_rot * current_expand_para * rot_loss / rot_scalar

        # 投射边缘loss
        edge_loss = 0
        if loader.data['inte_opti_para']['if_edge_freeze']:
            left_loss = torch.relu(proj_domain[0] - aim_xy_data[:, 0]).pow(2).sum()
            right_loss = torch.relu(aim_xy_data[:, 0] - proj_domain[1]).pow(2).sum()
            down_loss = torch.relu(proj_domain[2] - aim_xy_data[:, 1]).pow(2).sum()
            up_loss = torch.relu(aim_xy_data[:, 1] - proj_domain[3]).pow(2).sum()

            edge_loss = para_edge * (left_loss + right_loss + down_loss + up_loss)

        if(if_use_sobel == True):
            c_sobel_proj_xy_grad = render_grad[7].detach()
            c_sobel_tri_area_grad = render_grad[8].detach()
            render_loss = (
                ((para_sobel_1 * c_sobel_proj_xy_grad + para_render_1 * c_proj_xy_grad) * aim_xy_data).sum()
                + ((para_sobel_2 * c_sobel_tri_area_grad + para_render_2 * c_tri_area_grad) * equal_tri_energy).sum()
            )
        else:
            render_loss = (para_render_1 * c_proj_xy_grad * aim_xy_data).sum() + (para_render_2 * c_tri_area_grad * equal_tri_energy).sum()
        render_loss = render_loss/render_scalar

        if if_use_inte == True: 
            inte_loss = (para_inte_1 * c_proj_xy_inte_grad * aim_xy_data).sum() + (para_inte_2 * c_tri_area_inte_grad * equal_tri_energy).sum()
            inte_loss = inte_loss/render_scalar

        proj_f = render_loss + rot_loss + small_area_loss + big_area_loss + proj_area_loss + edge_loss

        basic_function.check_each_loss_grad(render_loss, rot_loss, small_area_loss, big_area_loss, proj_area_loss, edge_loss, opt_xy_data, aim_xy_data, full_origin_xy, triangles_tensor, data_foldername, iter_xiushi, iter_step)
        if if_use_inte == True:
            proj_f = proj_f + inte_loss

        optimizer_xy_data.zero_grad()
        optimizer_aim_xy_data.zero_grad()
        proj_f.backward()

        original_opt_xy_data = opt_xy_data.detach().clone()
        original_aim_xy_data = aim_xy_data.detach().clone()

        origin_xy_data_lr = optimizer_xy_data.param_groups[0]['lr']
        origin_aim_xy_data_lr = optimizer_aim_xy_data.param_groups[0]['lr']

        original_xy_optimizer_state = copy.deepcopy(optimizer_xy_data.state_dict())
        original_aim_optimizer_state = copy.deepcopy(optimizer_aim_xy_data.state_dict())

        current_xy_data_lr = origin_xy_data_lr
        current_aim_xy_data_lr = origin_aim_xy_data_lr

        success_1 = False
        success_2 = False

        for attempt in range(10):
            with torch.no_grad():
                opt_xy_data.copy_(original_opt_xy_data)
            optimizer_xy_data.load_state_dict(original_xy_optimizer_state)

            optimizer_xy_data.param_groups[0]['lr'] = current_xy_data_lr
            optimizer_xy_data.step()

            with torch.no_grad():
                full_origin_xy[~fixed_mask] = opt_xy_data.data
                check_areas = basic_function.compute_triangle_area(loader, full_origin_xy, triangles_tensor)

                if torch.any(check_areas <= 1.0 * epsilon_2) or torch.any(check_areas > (theta_2)):
                    min_area_val, min_idx = check_areas.min(dim=0)
                    max_area_val, max_idx = check_areas.max(dim=0)
                    current_xy_data_lr *= 0.5
                    print( "origin_tri", iter_step, float(min_area_val), min_idx, float(max_area_val), max_idx, float(current_xy_data_lr))
                else:
                    success_1 = True
                    optimizer_xy_data.param_groups[0]['lr'] = origin_xy_data_lr
                    break
        if not success_1:
            with torch.no_grad():
                opt_xy_data.copy_(original_opt_xy_data)
                full_origin_xy[~fixed_mask] = opt_xy_data

            optimizer_xy_data.load_state_dict(original_xy_optimizer_state)
            optimizer_xy_data.param_groups[0]['lr'] = origin_xy_data_lr
        
        for attempt in range(10):
            with torch.no_grad():
                aim_xy_data.copy_(original_aim_xy_data)

            optimizer_aim_xy_data.load_state_dict(original_aim_optimizer_state)
            optimizer_aim_xy_data.param_groups[0]['lr'] = current_aim_xy_data_lr
            optimizer_aim_xy_data.step()

            with torch.no_grad():
                compute_aim_tri_area = basic_function.compute_triangle_area(loader, aim_xy_data, triangles_tensor)
                if torch.min(compute_aim_tri_area) <= 0:
                    min_area_val, min_idx = compute_aim_tri_area.min(dim=0)
                    tri_idx = triangles_tensor[min_idx]
                    current_aim_xy_data_lr = 0.5 * current_aim_xy_data_lr
                    print( "proj_tri", iter_step, float(min_area_val), min_idx, tri_idx.detach().cpu().tolist(), float(current_aim_xy_data_lr))
                else:
                    success_2 = True
                    optimizer_aim_xy_data.param_groups[0]['lr'] = origin_aim_xy_data_lr
                    break
        if not success_2:
            with torch.no_grad():
                aim_xy_data.copy_(original_aim_xy_data)

            optimizer_aim_xy_data.load_state_dict(original_aim_optimizer_state)
            optimizer_aim_xy_data.param_groups[0]['lr'] = origin_aim_xy_data_lr
        
        if iter_step % 200 == 0:
            check_ratio = check_areas / origin_tri_area_0
            min_area = torch.min(check_ratio)
            max_area = torch.max(check_ratio)
            rot_inte = (origin_rot).detach().sum()
            if iter_step == 0:
                origin_rot_inte = rot_inte
            rotation = torch.abs((origin_rot)/origin_tri_area)
            max_rot = torch.max(rotation)
            # min_cos = torch.min(cos_values)
            min_proj_area = torch.min(compute_aim_tri_area)
            print('iter: {:d} | min_area: {:.4f} | max_area: {:.4g} | rot_inte: {:.4g} | rot_inte_rate: {:.2f}% | max_rot: {:.4g} | min_proj_area: {:.4g}'.format(
                iter_step, 
                float(min_area), 
                float(max_area), 
                float(rot_inte), 
                float(rot_inte/origin_rot_inte) * 100,
                float(max_rot), 
                float(min_proj_area)
            ))
            render_loss = torch.norm(render_grad[1], p=1)
            render_loss_squared = torch.sum(render_grad[1] ** 2)
            inte_loss = torch.norm(render_grad[4], p=1)

            msg = (
                    f'iter: {iter_step} | '
                    f'render_loss: {float(render_loss):.4g} | '
                    f'inte_loss: {float(inte_loss):.4g} | '
                    f'rot_loss: {float(rot_loss):.4g} | '
                    f'current_expand_para: {float(current_expand_para):.4g} *\n'
                    f'small_area_loss: {float(small_area_loss/para_small_area):.4g} | '
                    f'big_area_loss: {float(big_area_loss/para_big_area):.4g} | '
                    f'proj_area_loss: {float(proj_area_loss/para_proj_area):.4g} | '
                    f'edge_loss: {float(edge_loss/para_edge):.4g}'
                )
            
            print(msg)

            c_proj_xy_grad_loss = torch.norm(c_proj_xy_grad, p=2)
            c_tri_area_grad_loss = torch.norm(c_tri_area_grad, p=2)
            c_proj_xy_inte_grad_loss = torch.norm(c_proj_xy_inte_grad, p=2)
            c_tri_area_inte_grad_loss = torch.norm(c_tri_area_inte_grad, p=2)
            print(
                f'c_proj_xy_grad_loss: {float(c_proj_xy_grad_loss):.4g} | '
                f'c_tri_area_grad_loss: {float(c_tri_area_grad_loss):.4g} | '
                f'c_proj_xy_inte_grad_loss: {float(c_proj_xy_inte_grad_loss):.4g} | '
                f'c_tri_area_inte_grad_loss: {float(c_tri_area_inte_grad_loss):.4g}'
            )

            print(
                f'xy_step: {float(current_xy_data_lr):.4g} | '
                f'aim_xy_step: {float(current_aim_xy_data_lr):.4g}'
            )
            print("-" * 100)

            log_file = loader.data['data_render_folder']+f"optimization_log.txt"
            with open(log_file, 'a') as f:
                f.write(f"{optimization_time} {iter_step} {float(render_loss):.6f} {float(render_loss_squared):.6f} {float(rot_loss)/(base_para_rot*current_expand_para):.4g}\n")

        if iter_step % 2000 == 0 or iter_step == (inte_end_iter - 1):
            iter_xiushi = str(iter_step)
            picture_foldername = loader.data['data_render_folder']+f"{optimization_time}/"
            data_utils.picture_print(loader,render_grad[0],picture_foldername,iter_xiushi)

        if iter_step % 4000 == 0 or iter_step == (inte_end_iter - 1):
            average_rot = torch.abs((origin_rot)/origin_tri_area)
            display_rotation = (average_rot / origin_average_rot_max) * 255
            display_rotation = torch.clamp(display_rotation, min=0, max=255).to(torch.uint8)
            tri_color = torch.stack([torch.full_like(display_rotation, 255), 255 - display_rotation, 255 - display_rotation],  dim=1)

            iter_xiushi = str(iter_step)
            mesh_foldername = loader.data['data_mesh_folder']+f"{optimization_time}/"
            basic_function.display_color(mesh_foldername, iter_xiushi, distance,full_origin_xy,aim_xy_data,triangles_tensor,tri_color)

            data_foldername = loader.data['data_point_folder']+f"{optimization_time}/"
            data_utils.data_save(full_origin_xy,aim_xy_data,triangles_tensor,data_foldername,iter_xiushi)

    new_origin = origin_xy_data.detach().clone()
    new_aim = aim_xy_data.detach().clone()
    new_origin[~fixed_mask] = opt_xy_data.detach()
    origin_xy_data = new_origin.requires_grad_(True)
    aim_xy_data = new_aim.requires_grad_(True)

    return origin_xy_data,aim_xy_data

def spline_optimization(loader,distance,origin_xy_data,aim_xy_data,triangles_tensor,spline_end_iter,spline_learn_rate, total_gray_value,
                     opti_stage,optimization_time,pic_size):
    
    origin_xy_data.requires_grad_(True)
    for param in distance.parameters():
        param.requires_grad = True
    n = origin_xy_data.shape[0]
    
    regular_para = loader.data['spline_para']['regular_para']
    height_para = loader.data['spline_para']['height_para']
    gpu_regular_para = torch.tensor(regular_para, device="cuda:0")
    gpu_height_para = torch.tensor(height_para, device="cuda:0")
    if opti_stage == "ot_opti" and optimization_time == 0:
        gpu_height_para = gpu_height_para[0]
    else:
        gpu_height_para = gpu_height_para[1]

    thick_tolerant = torch.tensor(loader.data['total_opti_para']['thick_tolerant'], device="cuda:0")
        
    sample_domain = torch.tensor(loader.data["sample_point_para"]["sample_domain"], dtype=torch.float32).to("cuda:0")
    proj_domain = torch.tensor(loader.data["len_para"]["proj_domain"], dtype=torch.float32).to("cuda:0")

    origin_tri_energy = basic_function.compute_triangle_energy(loader, origin_xy_data, triangles_tensor)
    if loader.data['inte_opti_para']['if_consider_reflection_loss'] == True:
        remain_ratio = basic_function.compute_reflection_loss(loader, distance, origin_xy_data, aim_xy_data)
        if remain_ratio.dim() == 2 and remain_ratio.shape[1] == 1:
            remain_ratio = remain_ratio[:, 0]

        tri_ratio = remain_ratio[triangles_tensor].mean(dim=1, keepdim=True).detach()
        equal_tri_energy = origin_tri_energy * tri_ratio[:, 0]       
        with torch.no_grad():
            total_area = equal_tri_energy.sum().detach()
            average_gray_value = total_gray_value/total_area
    else:
        equal_tri_energy = origin_tri_energy
        with torch.no_grad():
            total_area = equal_tri_energy.sum().detach()
            average_gray_value = total_gray_value/total_area
    
    initial_params = [p for p in distance.parameters()]
    optimizer = torch.optim.Adam(initial_params, lr=spline_learn_rate, eps=1e-12)

    z_of_receiver = torch.full((aim_xy_data.shape[0], 1), loader.data["len_para"]["z_of_receiver"], device=aim_xy_data.device)
    aim_xyz_data = torch.cat((aim_xy_data,z_of_receiver),dim=1)
    z_of_len = torch.full((origin_xy_data.shape[0], 1), loader.data["len_para"]["z_of_len"], device=aim_xy_data.device)
    origin_xyz_data = torch.cat((origin_xy_data,z_of_len),dim=1)

    if loader.data['light_para']['light_type'] == "point_light":
        vertical_direct = torch.zeros((n, 3), device=origin_xy_data.device)
        vertical_direct[:,2] = 1
        inside_light_direct, pq_grad = basic_function.compute_inside_refraction_jacobian(loader, origin_xy_data, vertical_direct)
        inside_light_direct = inside_light_direct.detach().to("cuda:0")
        pq_grad = pq_grad.detach().to("cuda:0")

    if loader.data['light_para']['light_type'] == "parallel_light":
        inside_light_direct = torch.zeros((n, 3), device=origin_xy_data.device)
        inside_light_direct[:,2] = 1
        pq_grad = torch.zeros((n, 4), device=origin_xy_data.device)

    real_function_diff = torch.zeros_like(origin_xy_data[:, :2])
    aim_thickness = torch.tensor(loader.data['sample_point_para']['thickness'], dtype=torch.float32, device='cuda')
    
    if loader.data['spline_para']['if_regular'] == True:
        regular_x_num = distance.module.num_u - distance.module.p
        regular_y_num = distance.module.num_v - distance.module.p
        print(regular_x_num,regular_y_num)
        dx = (sample_domain[1] - sample_domain[0]) / regular_x_num
        dy = (sample_domain[3] - sample_domain[2]) / regular_y_num
        regular_x = torch.arange(regular_x_num, device="cuda:0") * dx + sample_domain[0] + dx * 0.5
        regular_y = torch.arange(regular_y_num, device="cuda:0") * dy + sample_domain[2] + dy * 0.5
        regular_X = regular_x.repeat(regular_y_num, 1).reshape(-1)
        regular_Y = regular_y.repeat(regular_x_num, 1).t().reshape(-1)
        regular_xy_data = torch.stack((regular_X, regular_Y), dim=1)
        regular_n = regular_xy_data.shape[0]
        # gpu_regular_para = origin_gpu_regular_para / regular_n

        if loader.data['light_para']['light_type'] == "point_light":
            regular_vertical_direct = torch.zeros((regular_n, 3), device=origin_xy_data.device)
            regular_vertical_direct[:,2] = 1
            regular_inside_light_direct, regular_pq_grad = basic_function.compute_inside_refraction_jacobian(loader, regular_xy_data, regular_vertical_direct)
            regular_inside_light_direct = regular_inside_light_direct.detach().to("cuda:0")
            regular_pq_grad = regular_pq_grad.detach().to("cuda:0")
        if loader.data['light_para']['light_type'] == "parallel_light":
            regular_inside_light_direct = torch.zeros((regular_n, 3), device=origin_xy_data.device)
            regular_inside_light_direct[:,2] = 1
            regular_pq_grad = torch.zeros((regular_n, 4), device=origin_xy_data.device)
    
    if opti_stage == "ot_opti":
        refine_schedule = { tuple(x) for x in loader.data["total_opti_para"]["ot_refine_time"]}
    if opti_stage == "op_opti":
        refine_schedule = { tuple(x) for x in loader.data["total_opti_para"]["op_refine_time"]}
        convergence_time = { tuple(x) for x in loader.data["total_opti_para"]["op_convergence_time"]}
    
    for iter_step in range(spline_end_iter):
        if iter_step == 0:
            picture_foldername = loader.data['spline_render_folder']+f"{opti_stage}_{optimization_time}/"
            control_point_foldername = loader.data['spline_control_point_folder']+f"{opti_stage}_{optimization_time}/"
            os.makedirs(picture_foldername, exist_ok=True)
            os.makedirs(control_point_foldername, exist_ok=True)
            basic_function.clear_folder(picture_foldername)
            basic_function.clear_folder(control_point_foldername)

            if loader.data['spline_para']['if_regular'] == True:
                precomputed_data = distance.module.precompute_basis_data_1(origin_xy_data)
                precomputed_regular_data = distance.module.precompute_basis_data_2(regular_xy_data)
            if loader.data['spline_para']['if_regular'] == False:
                precomputed_data = distance.module.precompute_basis_data_1(origin_xy_data)

        if (optimization_time, iter_step) in refine_schedule:
            distance.module.refine_surface()
            initial_params = [p for p in distance.parameters()]
            spline_learn_rate /= 2
            optimizer = torch.optim.Adam(initial_params, lr=spline_learn_rate)

            regular_x_num = distance.module.num_u - distance.module.p
            regular_y_num = distance.module.num_v - distance.module.p
            print(regular_x_num,regular_y_num)
            dx = (sample_domain[1] - sample_domain[0]) / regular_x_num
            dy = (sample_domain[3] - sample_domain[2]) / regular_y_num
            regular_x = torch.arange(regular_x_num, device="cuda:0") * dx + sample_domain[0] + dx * 0.5
            regular_y = torch.arange(regular_y_num, device="cuda:0") * dy + sample_domain[2] + dy * 0.5
            regular_X = regular_x.repeat(regular_y_num, 1).reshape(-1)
            regular_Y = regular_y.repeat(regular_x_num, 1).t().reshape(-1)
            regular_xy_data = torch.stack((regular_X, regular_Y), dim=1)

            regular_n = regular_xy_data.shape[0]
            gpu_regular_para = gpu_regular_para * 4

            if loader.data['light_para']['light_type'] == "point_light":
                regular_vertical_direct = torch.zeros((regular_n, 3), device=origin_xy_data.device)
                regular_vertical_direct[:,2] = 1
                regular_inside_light_direct, regular_pq_grad = basic_function.compute_inside_refraction_jacobian(loader, regular_xy_data, regular_vertical_direct)
                regular_inside_light_direct = regular_inside_light_direct.detach().to("cuda:0")
                regular_pq_grad = regular_pq_grad.detach().to("cuda:0")
            if loader.data['light_para']['light_type'] == "parallel_light":
                regular_inside_light_direct = torch.zeros((regular_n, 3), device=origin_xy_data.device)
                regular_inside_light_direct[:,2] = 1
                regular_pq_grad = torch.zeros((regular_n, 4), device=origin_xy_data.device)
            
            if loader.data['spline_para']['spline_type'] == "B_spline":
                if loader.data['spline_para']['if_regular'] == True:
                    precomputed_data = distance.module.precompute_basis_data_1(origin_xy_data)
                    precomputed_regular_data = distance.module.precompute_basis_data_2(regular_xy_data)
                if loader.data['spline_para']['if_regular'] == False:
                    precomputed_data = distance.module.precompute_basis_data_1(origin_xy_data)

        if opti_stage == "op_opti":
            if (optimization_time, iter_step) in convergence_time:
                initial_params = [p for p in distance.parameters()]
                spline_learn_rate = 1e-6
                optimizer = torch.optim.Adam(initial_params, lr=spline_learn_rate)

        if(iter_step % 100 == 0) :
            z_data = distance(origin_xy_data).detach()
            output_point = origin_xyz_data + z_data * inside_light_direct
            output_light_direct = aim_xyz_data - output_point
            normal_direct = basic_function.compute_normal(loader,inside_light_direct,output_light_direct).to("cuda:0")

            t1 = (1 + pq_grad[:,0].unsqueeze(1) * z_data).detach()
            t2 = (pq_grad[:,2].unsqueeze(1) * z_data).detach()
            t3 = (pq_grad[:,1].unsqueeze(1) * z_data).detach()
            t4 = (1 + pq_grad[:,3].unsqueeze(1) * z_data).detach()

            diff_scalar = torch.sum(inside_light_direct * normal_direct, dim=1, keepdim=True)
            real_function_diff = torch.cat([
                t1 * normal_direct[:, 0].unsqueeze(1) + t2 * normal_direct[:, 1].unsqueeze(1),
                t3 * normal_direct[:, 0].unsqueeze(1) + t4 * normal_direct[:, 1].unsqueeze(1)
            ], dim=1)/diff_scalar
            real_function_diff = real_function_diff.detach()

        if loader.data['spline_para']['if_regular'] == True:
            out_put = distance.module.forward_precomputed_1(precomputed_data)
            regular_out_put = distance.module.forward_precomputed_2(precomputed_regular_data)
        if loader.data['spline_para']['if_regular'] == False:
            out_put = distance.module.forward_precomputed_1(precomputed_data)
        
        optimizer.zero_grad()

        error = out_put["grads"] + real_function_diff
        loss1 = torch.norm(error, dim=1).sum()

        heights = regular_out_put["z"]
        grads = regular_out_put["grads"]
        grads_2nd = regular_out_put["grads_2nd"] 
        H, H_abs, avg_H_abs = basic_function.compute_mean_curvature(heights, grads, grads_2nd, regular_inside_light_direct, regular_pq_grad)
        
        height_mean = heights.mean()
        min_height = heights.min().detach()
        target_mean = (height_mean + thick_tolerant - min_height).detach()
        loss2 = gpu_height_para * ((height_mean - target_mean)**2).mean()

        loss3 = gpu_regular_para * avg_H_abs
            
        # 总损失 = 损失1 + 损失2 + 损失3 (正则项)
        loss = (loss1 + loss2 + loss3)
        loss.backward()

        if iter_step % 300 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print('iter: ',iter_step,
                        '| normal_loss: %.4g' % loss1.data,
                        '| height_loss: %.4f' % (loss2.data/gpu_height_para),
                        '| regular_loss: %.4f' % (loss3.data/gpu_regular_para),
                        '| lr: %.5e' % current_lr)
            height_amount=distance.forward(origin_xy_data)
            max_height=torch.max(height_amount)
            min_height=torch.min(height_amount)
            print("max_height",max_height,"min_height",min_height)

        if iter_step % 3000 == 0:
            origin_xy_data.requires_grad_(True)
            out_put = distance(origin_xy_data).sum()
            grad = torch.autograd.grad(out_put, origin_xy_data,create_graph=True)[0]
            n_data = torch.cat([t2 * grad[:, 1:2] - t4 * grad[:, 0:1], t3 * grad[:, 0:1] - t1 * grad[:, 1:2],
                                   (t1*t4-t3*t2) + (t4 * inside_light_direct[:,0:1] - t3 * inside_light_direct[:,1:2]) * grad[:,0:1]
                                   + (t1 * inside_light_direct[:,1:2] - t2 * inside_light_direct[:,0:1])*grad[:,1:2]], dim=1)
            z_data = distance(origin_xy_data).detach()
            output_point = origin_xyz_data + z_data * inside_light_direct
            
            display_proj_xy = basic_function.compute_proj(loader,inside_light_direct,n_data,output_point)

            render_result = irra_render.render(equal_tri_energy, triangles_tensor,display_proj_xy,average_gray_value,proj_domain,pic_size)

            picture_foldername = loader.data['spline_render_folder']+f"{opti_stage}_{optimization_time}/"
            iter_xiushi = f"{iter_step}_{loss1.item():.1f}"

            if iter_step % 3000 == 0:
                data_utils.picture_print(loader,render_result[0],picture_foldername,iter_xiushi)
                    
        if iter_step == (spline_end_iter - 1):
            control_point_foldername = loader.data['spline_control_point_folder']+f"{opti_stage}_{optimization_time}/"
            control_point_path = os.path.join(control_point_foldername, f"model_{iter_step}.pth")
            torch.save(distance.module , control_point_path)
        optimizer.step()

def spline_render(loader, distance, total_gray_value, tensor_image, pic_size, render_folder, module_folder):
    sample_domain = torch.tensor(loader.data["sample_point_para"]["sample_domain"], dtype=torch.float32).to("cuda:0")

    render_sample_num = loader.data["total_opti_para"]["render_sample_num"]
    render_x = torch.linspace(sample_domain[0], sample_domain[1], render_sample_num[0]).repeat(render_sample_num[1], 1).to("cuda:0").reshape(-1)
    render_y = torch.linspace(sample_domain[2], sample_domain[3], render_sample_num[1]).repeat(render_sample_num[0], 1).to("cuda:0").t().reshape(-1)
    render_xy = torch.stack((render_x, render_y), dim=1).requires_grad_(True)
    ray_energy = basic_function.compute_ray_energy(loader, render_xy, total_gray_value).detach()

    proj_domain = torch.tensor(loader.data["len_para"]["proj_domain"], dtype=torch.float32).to("cuda:0")
    render_z_of_len = torch.full((render_xy.shape[0], 1), loader.data["len_para"]["z_of_len"], device='cuda:0')
    render_xyz = torch.cat((render_xy, render_z_of_len), dim=1)

    if loader.data['light_para']['light_type'] == "point_light":
        render_vertical_direct = torch.zeros((render_xy.shape[0], 3), device='cuda:0')
        render_vertical_direct[:, 2] = 1
        render_inside_light_direct, render_pq_grad = basic_function.compute_inside_refraction_jacobian(
            loader, render_xy, render_vertical_direct)
        render_inside_light_direct = render_inside_light_direct.detach()
        render_pq_grad = render_pq_grad.detach()

        point_light_coordinate = torch.tensor(
            loader.data['light_para']['point_light_coordinate'],
            dtype=torch.float32, device='cuda:0'
        ).unsqueeze(0)
        input_light_direct = render_xyz - point_light_coordinate
    else:
        render_inside_light_direct = torch.zeros((render_xy.shape[0], 3), device='cuda:0')
        render_inside_light_direct[:, 2] = 1
        render_pq_grad = torch.zeros((render_xy.shape[0], 4), device='cuda:0')
        input_light_direct = torch.zeros((render_xy.shape[0], 3), device='cuda:0')
        input_light_direct[:, 2] = 1
        
    # 归一化并计算第一界面菲涅尔透射率 T1
    ri = loader.data["len_para"]["refraction_index"]
    input_light_direct = basic_function.normalize(input_light_direct)
    inside_light_direct = basic_function.normalize(render_inside_light_direct)
    n1 = torch.zeros((render_xy.shape[0], 3), device='cuda:0')
    n1[:, 2] = 1
    T1 = basic_function.fresnel_T_unpolarized(
        wi=input_light_direct, wt=inside_light_direct, normal=n1,
        eta_i=1.0, eta_t=ri
    )

    # 设置正则采样点
    regular_x_num = distance.module.num_u - distance.module.p
    regular_y_num = distance.module.num_v - distance.module.p
    dx = (sample_domain[1] - sample_domain[0]) / regular_x_num
    dy = (sample_domain[3] - sample_domain[2]) / regular_y_num
    regular_x = torch.arange(regular_x_num, device="cuda:0") * dx + sample_domain[0] + dx * 0.5
    regular_y = torch.arange(regular_y_num, device="cuda:0") * dy + sample_domain[2] + dy * 0.5
    regular_X = regular_x.repeat(regular_y_num, 1).reshape(-1)
    regular_Y = regular_y.repeat(regular_x_num, 1).t().reshape(-1)
    regular_xy_data = torch.stack((regular_X, regular_Y), dim=1)
    regular_n = regular_xy_data.shape[0]

    if loader.data['light_para']['light_type'] == "point_light":
        regular_vertical_direct = torch.zeros((regular_n, 3)).to("cuda:0")
        regular_vertical_direct[:,2] = 1
        regular_inside_light_direct, regular_pq_grad = basic_function.compute_inside_refraction_jacobian(loader, regular_xy_data, regular_vertical_direct)
        regular_inside_light_direct = regular_inside_light_direct.detach().to("cuda:0")
        regular_pq_grad = regular_pq_grad.detach().to("cuda:0")
    if loader.data['light_para']['light_type'] == "parallel_light":
        regular_inside_light_direct = torch.zeros((regular_n, 3)).to("cuda:0")
        regular_inside_light_direct[:,2] = 1
        regular_pq_grad = torch.zeros((regular_n, 4)).to("cuda:0")

    # ---------- 预计算样条基函数数据（一次性） ----------
    print("Precomputing basis data for all samples (this may take a while)...")
    render_xy = render_xy.detach()
    render_precomputed_dict = distance.module.precompute_basis_data_1(render_xy)
    regular_precomputed_dict = distance.module.precompute_basis_data_2(regular_xy_data)

    render_end_iter = loader.data["total_opti_para"]["render_end_iter"]
    learn_rate = loader.data["total_opti_para"]["learn_rate"]
    chunk_size = loader.data["total_opti_para"]["chunk_size"]
    sigma = loader.data["total_opti_para"]["sigma"]
    radius = loader.data["total_opti_para"]["radius"]
    H_para = loader.data["total_opti_para"]["H_para"]
    sharp_sigma = loader.data["total_opti_para"]["sharp_sigma"]
    sharp_radius = loader.data["total_opti_para"]["sharp_radius"]

    optimizer = torch.optim.Adam(distance.parameters(), lr=learn_rate)
    criterion = nn.MSELoss()
    target_rms = torch.sqrt(torch.mean(tensor_image ** 2))

    for iter_step in range(render_end_iter):
        if iter_step == 0:
            render_folder = render_folder
            os.makedirs(render_folder, exist_ok=True)
            basic_function.clear_folder(render_folder)
            module_folder = module_folder
            os.makedirs(module_folder, exist_ok=True)
            basic_function.clear_folder(module_folder) 
        
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)

        regular_out_put = distance.module.forward_precomputed_2(regular_precomputed_dict)

        regular_heights = regular_out_put["z"]
        regular_grads = regular_out_put["grads"]
        regular_grads_2nd = regular_out_put["grads_2nd"]
        H, H_abs, avg_H_abs = basic_function.compute_mean_curvature(regular_heights, regular_grads, regular_grads_2nd,
                                                                     regular_inside_light_direct, regular_pq_grad)
        # ---- 计算出射点与表面法向量 ----
        render_out_put = distance.module.forward_precomputed_1(render_precomputed_dict)
        render_heights = render_out_put["z"]
        render_grads = render_out_put["grads"]
        t1 = (1 + render_pq_grad[:, 0:1] * render_heights).detach()
        t2 = (render_pq_grad[:, 2:3] * render_heights).detach()
        t3 = (render_pq_grad[:, 1:2] * render_heights).detach()
        t4 = (1 + render_pq_grad[:, 3:4] * render_heights).detach()

        output_point = render_xyz + render_heights * render_inside_light_direct

        render_n = torch.cat([
            t2 * render_grads[:, 1:2] - t4 * render_grads[:, 0:1],
            t3 * render_grads[:, 0:1] - t1 * render_grads[:, 1:2],
            (t1*t4 - t3*t2) + (t4*render_inside_light_direct[:,0:1] - t3*render_inside_light_direct[:,1:2]) * render_grads[:,0:1]
            + (t1*render_inside_light_direct[:,1:2] - t2*render_inside_light_direct[:,0:1]) * render_grads[:,1:2]
        ], dim=1)

        output_direct, proj_xy, valid_mask = basic_function.compute_proj_mask(
            loader, render_inside_light_direct, render_n, output_point, eps=1e-12
        )

        # ---- 计算第二界面菲涅尔透射率与有效光线能量 ----
        with torch.no_grad():
            out_dir_valid = output_direct[valid_mask]
            inside_valid = inside_light_direct[valid_mask]
            n2_valid = render_n[valid_mask]
            out_dir_valid = basic_function.normalize(out_dir_valid)
            inside_valid = basic_function.normalize(inside_valid)
            n2_valid = basic_function.normalize(n2_valid)
            T2 = basic_function.fresnel_T_unpolarized(
                wi=inside_valid, wt=out_dir_valid, normal=n2_valid,
                eta_i=ri, eta_t=1.0
            )
            remain_ratio = (T1[valid_mask] * T2).detach()
            valid_energy = ray_energy[valid_mask] * remain_ratio

        valid_points = proj_xy[valid_mask]

        rendered_image, loss1 = (
            basic_function.render_gaussian_splat_two_pass_backward(
                loader=loader, total_gray_value=total_gray_value, valid_points=valid_points, ray_energy=valid_energy,
                pic_size=pic_size, proj_domain=proj_domain, target_image=tensor_image, target_rms=target_rms,
                sigma=sigma, radius=radius, chunk_size=chunk_size,
            )
        )
        
        loss2 = H_para * avg_H_abs
        loss2.backward()

        optimizer.step()

        # 日志与保存
        if iter_step % 50 == 0:
            torch.cuda.synchronize()
            logical_id = torch.cuda.current_device()
            gpu_status = get_gpu_status(logical_id)

            torch_mem_alloc = torch.cuda.memory_allocated(logical_id) / 1024**2
            torch_mem_reserved = torch.cuda.memory_reserved(logical_id) / 1024**2
            torch_mem_peak = torch.cuda.max_memory_allocated(logical_id) / 1024**2

            lr = optimizer.param_groups[0]["lr"]

            if gpu_status is not None:
                print(
                    f"iter_step: {iter_step}, "
                    f"loss1: {loss1.item():.8f}, "
                    f"loss2: {loss2.item() if torch.is_tensor(loss2) else loss2:.8f}, "
                    f"lr: {lr:.3e}, "
                    f"cuda logical id: {logical_id}, "
                    f"physical gpu id: {gpu_status['physical_gpu_id']}, "
                    f"nvidia-smi index: {gpu_status['nvidia_smi_index']}, "
                    f"GPU util: {gpu_status['gpu_util']:.1f}%, "
                    f"GPU mem: {gpu_status['mem_used']:.0f}/{gpu_status['mem_total']:.0f} MB, "
                    f"torch alloc: {torch_mem_alloc:.0f} MB, "
                    f"torch reserved: {torch_mem_reserved:.0f} MB, "
                    f"torch peak: {torch_mem_peak:.0f} MB"
                )

        if iter_step % 100 == 0:
            iter_xiushi = f"{iter_step}"
            rendered_image = rendered_image.detach().transpose(0, 1).flip(0)
            data_utils.picture_print(loader,rendered_image.detach(),render_folder,iter_xiushi)
        if iter_step == (render_end_iter - 1):
            control_point_path = os.path.join(module_folder, f"model_{iter_step}.pth")
            torch.save(distance.module , control_point_path)

        if iter_step % ((render_end_iter-1)/3) == 0 and iter_step > 0:
            learn_rate=learn_rate/2
            optimizer = torch.optim.Adam(distance.parameters(), lr=learn_rate)
            print(learn_rate)