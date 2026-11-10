def render_gaussian_splat(loader, total_gray_value, valid_points, ray_energy, pic_size, proj_domain,
                                 sigma=0.5, radius=2, chunk_size=3000000):
    device = points.device
    dtype = points.dtype

    res_y, res_x = int(pic_size[0]), int(pic_size[1])
    x_min, x_max, y_min, y_max = proj_domain

    legal_mask = (torch.isfinite(valid_points).all(dim=1) & torch.isfinite(ray_energy))
    total_energy = ray_energy[legal_mask].sum()

    # --- 0. 过滤非法点 ---
    mask = (
        torch.isfinite(valid_points).all(dim=1) &
        torch.isfinite(ray_energy) &
        (valid_points[:, 0] >= x_min) & (valid_points[:, 0] < x_max) &
        (valid_points[:, 1] >= y_min) & (valid_points[:, 1] < y_max)
    )

    valid_points = valid_points[mask]
    ray_energy = ray_energy[mask]

    n = valid_points.shape[0]
    if n == 0:
        return torch.zeros((res_y, res_x), device=device, dtype=dtype)

    # 物理坐标 -> 像素坐标
    scale_x = (x_max - x_min) / res_x
    scale_y = (y_max - y_min) / res_y

    px = (valid_points[:, 0] - x_min) / scale_x
    py = (valid_points[:, 1] - y_min) / scale_y

    padded_res_y = res_y + 2 * radius
    padded_res_x = res_x + 2 * radius

    padded_img = torch.zeros(
        (padded_res_y, padded_res_x),
        device=device,
        dtype=dtype
    )

    offset_range = torch.arange(-radius, radius + 1, device=device)
    dy_grid, dx_grid = torch.meshgrid(offset_range, offset_range, indexing="ij")

    dx_flat = dx_grid.reshape(-1)
    dy_flat = dy_grid.reshape(-1)

    padded_img_flat = padded_img.view(-1)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        chunk_ix = torch.floor(px[start:end]).long() + radius
        chunk_iy = torch.floor(py[start:end]).long() + radius
        chunk_energy = ray_energy[start:end]

        xx = chunk_ix[:, None] + dx_flat[None, :]
        yy = chunk_iy[:, None] + dy_flat[None, :]

        px_pad = px[start:end] + radius
        py_pad = py[start:end] + radius

        dx_real = (xx.to(dtype) + 0.5) - px_pad[:, None]
        dy_real = (yy.to(dtype) + 0.5) - py_pad[:, None]

        d2 = dx_real * dx_real + dy_real * dy_real
        weights = torch.exp(-d2 / (2 * sigma * sigma))

        sum_w = weights.sum(dim=1, keepdim=True).detach()
        weights = weights / (sum_w)
        local_contrib = weights * chunk_energy[:, None]

        linear_idx = yy * padded_res_x + xx

        valid_splat = (
            (xx >= 0) & (xx < padded_res_x) &
            (yy >= 0) & (yy < padded_res_y)
        )

        padded_img_flat.scatter_add_(
            0,
            linear_idx[valid_splat].reshape(-1),
            local_contrib[valid_splat].reshape(-1)
        )

    img = padded_img[
        radius:radius + res_y,
        radius:radius + res_x
    ]

    img = img / (total_energy) * total_gray_value

    return img
    
def render_mitsuba_box_splat( loader, total_gray_value, valid_points, ray_energy, pic_size, proj_domain, chunk_size=3000000):

    device = valid_points.device
    dtype = valid_points.dtype

    res_y, res_x = int(pic_size[0]), int(pic_size[1])
    x_min, x_max, y_min, y_max = proj_domain

    legal_mask = (torch.isfinite(valid_points).all(dim=1) & torch.isfinite(ray_energy))
    total_energy = ray_energy[legal_mask].sum()

    # ---- 0. filter invalid samples ----
    mask = (
        torch.isfinite(valid_points).all(dim=1) &
        torch.isfinite(ray_energy) &
        (valid_points[:, 0] >= x_min) & (valid_points[:, 0] < x_max) &
        (valid_points[:, 1] >= y_min) & (valid_points[:, 1] < y_max)
    )

    valid_points = valid_points[mask]
    ray_energy = ray_energy[mask]

    n = valid_points.shape[0]
    if n == 0:
        return torch.zeros((res_y, res_x), device=device, dtype=dtype)

    # ---- 1. physical coordinate -> fractional pixel coordinate ----
    scale_x = (x_max - x_min) / res_x
    scale_y = (y_max - y_min) / res_y

    px = (valid_points[:, 0] - x_min) / scale_x
    py = (valid_points[:, 1] - y_min) / scale_y

    # ---- 2. Mitsuba-style box: floor to one pixel ----
    ix = torch.floor(px).long()
    iy = torch.floor(py).long()

    inside = (
        (ix >= 0) & (ix < res_x) &
        (iy >= 0) & (iy < res_y)
    )

    ix = ix[inside]
    iy = iy[inside]
    ray_energy = ray_energy[inside]

    img = torch.zeros((res_y, res_x), device=device, dtype=dtype)
    img_flat = img.view(-1)

    linear_idx = iy * res_x + ix

    for start in range(0, linear_idx.shape[0], chunk_size):
        end = min(start + chunk_size, linear_idx.shape[0])

        img_flat.scatter_add_(
            0,
            linear_idx[start:end],
            ray_energy[start:end]
        )

    # keep your current normalization convention
    img = img / (total_energy) * total_gray_value

    return img

def render_differentiable_box_splat( loader, total_gray_value, valid_points, ray_energy, pic_size, proj_domain, chunk_size=3000000):
    device = valid_points.device
    dtype = valid_points.dtype

    res_y, res_x = int(pic_size[0]), int(pic_size[1])
    x_min, x_max, y_min, y_max = proj_domain

    legal_mask = (
        torch.isfinite(valid_points).all(dim=1) &
        torch.isfinite(ray_energy)
    )

    total_energy = ray_energy[legal_mask].sum()

    scale_x = (x_max - x_min) / res_x
    scale_y = (y_max - y_min) / res_y

    px_all = (valid_points[:, 0] - x_min) / scale_x
    py_all = (valid_points[:, 1] - y_min) / scale_y

    mask = (
        torch.isfinite(valid_points).all(dim=1) &
        torch.isfinite(ray_energy) &
        torch.isfinite(px_all) &
        torch.isfinite(py_all) &
        (px_all >= -1.0) & (px_all < res_x + 1.0) &
        (py_all >= -1.0) & (py_all < res_y + 1.0)
    )

    px = px_all[mask]
    py = py_all[mask]
    ray_energy = ray_energy[mask]
    n = px.shape[0]
    
    pad = 2
    padded_res_y = res_y + 2 * pad
    padded_res_x = res_x + 2 * pad

    padded_img = torch.zeros(
        (padded_res_y, padded_res_x),
        device=device,
        dtype=dtype
    )

    padded_img_flat = padded_img.view(-1)

    # --------------------------------------------------
    # 5. 分 chunk 做 2x2 box splat
    # --------------------------------------------------
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        chunk_px = px[start:end]
        chunk_py = py[start:end]
        chunk_energy = ray_energy[start:end]

        # --------------------------------------------------
        # box_width = 1.0
        #
        # box footprint:
        #   [px - 0.5, px + 0.5]
        #
        # 等价于对 ux = px - 0.5 做双线性面积权重。
        # --------------------------------------------------
        ux = chunk_px - 0.5
        uy = chunk_py - 0.5

        ix0 = torch.floor(ux).long()
        iy0 = torch.floor(uy).long()

        fx = ux - ix0.to(dtype)
        fy = uy - iy0.to(dtype)

        ix1 = ix0 + 1
        iy1 = iy0 + 1

        # 四个像素的面积权重
        w00 = (1.0 - fx) * (1.0 - fy)
        w10 = fx * (1.0 - fy)
        w01 = (1.0 - fx) * fy
        w11 = fx * fy

        # --------------------------------------------------
        # 原始像素坐标 -> padded 像素坐标
        # 原始像素 (0,0) 对应 padded 像素 (pad,pad)
        # --------------------------------------------------
        ix0_p = ix0 + pad
        ix1_p = ix1 + pad
        iy0_p = iy0 + pad
        iy1_p = iy1 + pad

        idx00 = iy0_p * padded_res_x + ix0_p
        idx10 = iy0_p * padded_res_x + ix1_p
        idx01 = iy1_p * padded_res_x + ix0_p
        idx11 = iy1_p * padded_res_x + ix1_p

        valid00 = (
            (ix0_p >= 0) & (ix0_p < padded_res_x) &
            (iy0_p >= 0) & (iy0_p < padded_res_y)
        )
        valid10 = (
            (ix1_p >= 0) & (ix1_p < padded_res_x) &
            (iy0_p >= 0) & (iy0_p < padded_res_y)
        )
        valid01 = (
            (ix0_p >= 0) & (ix0_p < padded_res_x) &
            (iy1_p >= 0) & (iy1_p < padded_res_y)
        )
        valid11 = (
            (ix1_p >= 0) & (ix1_p < padded_res_x) &
            (iy1_p >= 0) & (iy1_p < padded_res_y)
        )

        c00 = w00 * chunk_energy
        c10 = w10 * chunk_energy
        c01 = w01 * chunk_energy
        c11 = w11 * chunk_energy

        padded_img_flat.scatter_add_(0, idx00[valid00], c00[valid00])
        padded_img_flat.scatter_add_(0, idx10[valid10], c10[valid10])
        padded_img_flat.scatter_add_(0, idx01[valid01], c01[valid01])
        padded_img_flat.scatter_add_(0, idx11[valid11], c11[valid11])

    img = padded_img[
        pad:pad + res_y,
        pad:pad + res_x
    ]

    img = img / (total_energy) * total_gray_value

    return img
    
def spline_render_box(loader, distance, origin_xy_data, triangles_tensor, total_gray_value, tensor_image, pic_size):
    sample_domain = torch.tensor(loader.data["sample_point_para"]["sample_domain"], dtype=torch.float32).to("cuda:0")
    origin_xy_data = origin_xy_data.detach()
    if loader.data["total_opti_para"]["if_triangle_refine"] == True:
        render_refine_n = loader.data["total_opti_para"]["render_refine_n"]
        render_xy, ray_energy = basic_function.sample_triangle_energy_centers(loader, origin_xy_data, triangles_tensor, render_refine_n, total_gray_value)
    else:
        render_sample_num = loader.data["total_opti_para"]["render_box_sample_num"]
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
        regular_vertical_direct = torch.zeros((regular_n, 3), device=origin_xy_data.device)
        regular_vertical_direct[:,2] = 1
        regular_inside_light_direct, regular_pq_grad = basic_function.compute_inside_refraction_jacobian(loader, regular_xy_data, regular_vertical_direct)
        regular_inside_light_direct = regular_inside_light_direct.detach().to("cuda:0")
        regular_pq_grad = regular_pq_grad.detach().to("cuda:0")
    if loader.data['light_para']['light_type'] == "parallel_light":
        regular_inside_light_direct = torch.zeros((regular_n, 3), device=origin_xy_data.device)
        regular_inside_light_direct[:,2] = 1
        regular_pq_grad = torch.zeros((regular_n, 4), device=origin_xy_data.device)

    # ---------- 预计算样条基函数数据（一次性） ----------
    print("Precomputing basis data for all samples (this may take a while)...")
    render_xy = render_xy.detach()
    render_precomputed_dict = distance.module.precompute_basis_data_1(render_xy)
    regular_precomputed_dict = distance.module.precompute_basis_data_2(regular_xy_data)
    # test_precomputed_dict = distance.module.precompute_basis_data_1(test_xy)

    render_end_iter = loader.data["total_opti_para"]["render_box_end_iter"]
    learn_rate = loader.data["total_opti_para"]["box_learn_rate"]
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
            render_folder = loader.root_dir+"/result_spline/"+loader.data["xiushi"]+"/ray_tracing/render/box/"
            os.makedirs(render_folder, exist_ok=True)
            basic_function.clear_folder(render_folder)
            module_folder = loader.root_dir+"/result_spline/"+loader.data["xiushi"]+"/ray_tracing/control_point/box/"
            os.makedirs(module_folder, exist_ok=True)
            basic_function.clear_folder(module_folder) 
        
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)

        if iter_step == (render_end_iter-1)/2:
            sigma = sharp_sigma
            radius = sharp_radius
            optimizer = torch.optim.Adam(distance.parameters(), lr=learn_rate)
            print(f"Switch Gaussian filter at iter {iter_step}: sigma={sigma}, radius={radius}, reset optimizer.")

        # if(iter_step != 6):
        regular_out_put = distance.module.forward_precomputed_2(regular_precomputed_dict)
        # else:
        #     regular_out_put = distance.module.forward_precomputed_2_check(regular_precomputed_dict)

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
        rendered_image = basic_function.render_differentiable_box_splat( loader, total_gray_value, valid_points, valid_energy, pic_size, proj_domain, chunk_size)

        loss1 = torch.sqrt(criterion(rendered_image, tensor_image))/target_rms
        
        loss2 = H_para * avg_H_abs
        loss = loss1+loss2
        loss.backward()
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

            rendered_image = basic_function.render_mitsuba_box_splat( loader, total_gray_value, valid_points, valid_energy, pic_size, proj_domain, chunk_size)
            iter_xiushi = f"{iter_step}_mitsuba_box"
            rendered_image = rendered_image.detach().transpose(0, 1).flip(0)
            data_utils.picture_print(loader,rendered_image.detach(),render_folder,iter_xiushi)
        
        if iter_step == (render_end_iter - 1):
            control_point_path = os.path.join(module_folder, f"model_{iter_step}.pth")
            torch.save(distance.module , control_point_path)

        if iter_step % ((render_end_iter-1)/3) == 0 and iter_step > 0:
            learn_rate=learn_rate/2
            optimizer = torch.optim.Adam(distance.parameters(), lr=learn_rate)
            print(learn_rate)