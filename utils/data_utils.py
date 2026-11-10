import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.spatial import Delaunay
import utils.spline as spline
from models.caustic_model import CausticNN

def read_data_from_txt(file_path):
    try:
        data_array = np.loadtxt(file_path, delimiter=' ')
        gpu_tensor = torch.as_tensor(data_array, dtype=torch.float32, device="cuda:0")
        return gpu_tensor
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"加载错误: {str(e)}")
        return None

def load_ot_data(point_data_path,  total_iter):
    origin_xy = read_data_from_txt(point_data_path + "in.txt")
    aim_xy = read_data_from_txt(point_data_path + f"{total_iter}" +"/out.txt")
    origin_xy = (origin_xy).requires_grad_()
    aim_xy = (aim_xy).requires_grad_()
    return origin_xy, aim_xy

def load_point_data(loader):
    origin_xy_data = torch.load(loader.data['origin_xy_data_path'],weights_only=True).to(torch.float32).to("cuda:0")
    aim_xy_data = torch.load(loader.data['aim_xy_data_path'],weights_only=True).to(torch.float32).to("cuda:0")
    triangles_tensor = torch.load(loader.data['triangle_tensor_path'],weights_only=True).to("cuda:0")
    return origin_xy_data, aim_xy_data, triangles_tensor

def load_point_data_id(root_dir, input_name, iter_num, data_num):
    aim_xy_data_path = root_dir + "/result_sp_data/"f"{input_name}/data_point/{iter_num}/aim_xy_data_{data_num}.pt"
    origin_xy_data_path = root_dir + "/result_sp_data/"f"{input_name}/data_point/{iter_num}/origin_xy_data_{data_num}.pt"
    triangle_tensor_path = root_dir + "/result_sp_data/"f"{input_name}/data_point/{iter_num}/triangle_tensor_{data_num}.pt"
    origin_xy_data = torch.load(origin_xy_data_path,weights_only=True).to(torch.float32).to("cuda:0")
    aim_xy_data = torch.load(aim_xy_data_path,weights_only=True).to(torch.float32).to("cuda:0")
    triangles_tensor = torch.load(triangle_tensor_path,weights_only=True).to("cuda:0")
    return origin_xy_data, aim_xy_data, triangles_tensor

def save_array_to_txt(corrected_image_array, txt_path):
    corrected_image_array = np.asarray(corrected_image_array, dtype=np.float32)
    if corrected_image_array.ndim != 2:
        raise ValueError("输入必须是二维矩阵")
    np.savetxt(txt_path, corrected_image_array, fmt="%.6f")

def load_pic_data(loader):
    image = Image.open(loader.data['image_path']).convert("L")
    image_array = np.array(image)
        
    pic_size = torch.tensor(
        image_array.shape,
        dtype=torch.float32
    ).to("cuda:0")

    tensor_image = torch.tensor(image_array, dtype=torch.float32).to("cuda:0")
    tensor_image = torch.flip(tensor_image, dims=[0]).t().contiguous()
    
    total_gray_value=tensor_image.sum()
    max_gray = tensor_image.max()
    return tensor_image,total_gray_value,pic_size,max_gray

def load_distance(loader):
    z_of_len = torch.tensor(loader.data["len_para"]["z_of_len"], device="cuda:0")
    z_of_receiver = torch.tensor(loader.data["len_para"]["z_of_receiver"], device="cuda:0")
    return z_of_len,z_of_receiver

def Delaunay_create(xy_data):
    xy_data_np = xy_data.cpu().detach().numpy()
    triangulation = Delaunay(xy_data_np)
    triangles = triangulation.simplices
    triangles_int_tensor = torch.tensor(triangles, dtype=torch.int).to("cuda:0")
    return triangles_int_tensor

def picture_print(loader, render_picture, picture_path, picture_name):
    os.makedirs(picture_path, exist_ok=True)

    txt_path = os.path.join(picture_path, picture_name + ".txt")
    png_path = os.path.join(picture_path, picture_name + ".png")

    txt_data = render_picture.detach().cpu().numpy()
    txt_data = txt_data.astype(np.int32)

    with open(txt_path, "w") as f:
        for row in txt_data:
            line = "".join(f"{num:4d}" for num in row)
            f.write(line + "\n")

    img = torch.clamp(render_picture, min=0, max=255).to(torch.uint8).detach().cpu()

    image_pil = Image.fromarray(img.numpy())
    image_pil.save(png_path)

def data_save(origin_xy_data,aim_xy_data,triangles_tensor,data_foldername,iter_xiushi):
    torch.save(origin_xy_data.detach().cpu(), data_foldername+"origin_xy_data_"+iter_xiushi+".pt")
    torch.save(aim_xy_data.detach().cpu(), data_foldername+"aim_xy_data_"+iter_xiushi+".pt")
    torch.save(triangles_tensor.detach().cpu(), data_foldername+"triangle_tensor_"+iter_xiushi+".pt")

def load_data(data_foldername,iter_xiushi):
    origin_xy_data = torch.load(data_foldername+"/origin_xy_data_"+iter_xiushi+".pt",weights_only=True).to(torch.float32).to("cuda:0")
    aim_xy_data = torch.load(data_foldername+"/aim_xy_data_"+iter_xiushi+".pt",weights_only=True).to(torch.float32).to("cuda:0")
    triangles_tensor = torch.load(data_foldername+"/triangle_tensor_"+iter_xiushi+".pt",weights_only=True).to("cuda:0")
    return origin_xy_data,aim_xy_data,triangles_tensor

def save_2D_mesh_to_obj(vertices, triangles, filename):
    zeros = torch.zeros((vertices.shape[0], 1), dtype=vertices.dtype).to("cuda:0")
    vertices_3d = torch.cat((vertices, zeros), dim=1)
    
    with open(filename, 'w') as f:
        for v in vertices_3d:
            f.write(f"v {v[0].item()} {v[1].item()} {v[2].item()}\n")
        for t in triangles:
            f.write(f"f {t[0].item()+1} {t[1].item()+1} {t[2].item()+1}\n")

def initialize_caustic_spline(loader, device_ids, if_requires_grad):
    spline_para = loader.data["spline_para"]
    sample_domain = loader.data["sample_point_para"]["sample_domain"]

    spline_func = spline.DifferentiableBspline(
        num_ctrl_u = spline_para["ctrl_points_num"][0],
        num_ctrl_v = spline_para["ctrl_points_num"][1], 
        degree = spline_para["spline_degree"], 
        x_range=(sample_domain[0], sample_domain[1]), 
        y_range=(sample_domain[2], sample_domain[3]), 
        padding = spline_para["padding"],
        avg_height = loader.data["sample_point_para"]["thickness"]
    ).to("cuda:0")

    spline_func = torch.nn.DataParallel(spline_func, device_ids=device_ids)

    if not if_requires_grad:
        for param in spline_func.parameters():
            param.requires_grad = False

    return spline_func

def initialize_caustic_thb_spline(loader, distance, device_ids, if_requires_grad=True):
    max_layer = loader.data["spline_para"]["max_layer"]
    spline_func = spline.DifferentiableTHBspline(distance, max_layer).to("cuda:0")
    spline_func = torch.nn.DataParallel(spline_func, device_ids=device_ids)
    if not if_requires_grad:
        for param in spline_func.parameters():
            param.requires_grad = False
    return spline_func

def subdivide_two_meshes(origin_xy, aim_xy, triangles):
    device = origin_xy.device
    dtype = origin_xy.dtype
    
    edges_raw = torch.cat([
        triangles[:, [0, 1]],
        triangles[:, [1, 2]],
        triangles[:, [2, 0]]
    ], dim=0)
    
    edges_sorted, _ = torch.sort(edges_raw, dim=1)
    
    unique_edges, inverse_indices = torch.unique(edges_sorted, dim=0, return_inverse=True)
    
    e_v0 = unique_edges[:, 0]
    e_v1 = unique_edges[:, 1]
    
    mid_origin = (origin_xy[e_v0] + origin_xy[e_v1]) * 0.5
    mid_aim = (aim_xy[e_v0] + aim_xy[e_v1]) * 0.5
    
    num_original_verts = origin_xy.shape[0]
    
    new_origin_xy = torch.cat([origin_xy, mid_origin], dim=0)
    new_aim_xy = torch.cat([aim_xy, mid_aim], dim=0)

    midpoint_indices = inverse_indices + num_original_verts
    
    num_faces = triangles.shape[0]
    m_indices = midpoint_indices.reshape(3, num_faces).t()
    
    v0 = triangles[:, 0]
    v1 = triangles[:, 1]
    v2 = triangles[:, 2]
    
    m01 = m_indices[:, 0]
    m12 = m_indices[:, 1]
    m20 = m_indices[:, 2]
    
    t1 = torch.stack([v0, m01, m20], dim=1)
    t2 = torch.stack([m01, v1, m12], dim=1)
    t3 = torch.stack([m20, m12, v2], dim=1)
    t4 = torch.stack([m01, m12, m20], dim=1)
    
    new_triangles = torch.cat([t1, t2, t3, t4], dim=0).int()
    
    return new_origin_xy, new_aim_xy, new_triangles

def adaptive_subdivide(origin_xy_data,triangles_tensor,n,area_threshold):
    device = origin_xy_data.device
    dtype = origin_xy_data.dtype
    M = triangles_tensor.shape[0]

    # === Compute triangle areas ===
    v0 = origin_xy_data[triangles_tensor[:, 0]]
    v1 = origin_xy_data[triangles_tensor[:, 1]]
    v2 = origin_xy_data[triangles_tensor[:, 2]]
    cross_val = (v1[:,0] - v0[:,0]) * (v2[:,1] - v0[:,1]) \
            - (v1[:,1] - v0[:,1]) * (v2[:,0] - v0[:,0])
    areas = 0.5 * torch.abs(cross_val)
    need_split = areas >= area_threshold  # [M]

    # === Construct unique edges ===
    edges_full = torch.cat([
        triangles_tensor[:, [0, 1]],
        triangles_tensor[:, [1, 2]],
        triangles_tensor[:, [2, 0]],
    ], dim=0)  # [3M, 2]

    edges_sorted = torch.sort(edges_full, dim=1).values
    unique_edges, edge_inverse = torch.unique(edges_sorted,
                                              dim=0, return_inverse=True)
    E = unique_edges.shape[0]

    edge_need_split = torch.zeros(E, dtype=torch.bool, device=device)
    for i in range(3):
        edge_need_split[edge_inverse[i*M:(i+1)*M]] |= need_split

    new_points = [origin_xy_data]
    
    if edge_need_split.any():
        e0 = origin_xy_data[unique_edges[:, 0]]
        e1 = origin_xy_data[unique_edges[:, 1]]
        t_points = torch.linspace(0, 1, n+1, dtype=dtype, device=device)[1:-1]
        if len(t_points) > 0:
            t = t_points.view(1, -1, 1)  # [1,n-1,1]
            subdiv_e = (1 - t) * e0[:, None, :] + t * e1[:, None, :]
            subdiv_e = subdiv_e[edge_need_split]  # only needed edges
            new_points.append(subdiv_e.reshape(-1, 2))  # flatten

    for tri_idx, tri in enumerate(triangles_tensor):
        if not need_split[tri_idx]:
            continue

        p0, p1, p2 = origin_xy_data[tri]
        interior = []
        for i in range(1, n):
            for j in range(1, n-i):
                k = n - i - j
                alpha = i / n
                beta  = j / n
                gamma = k / n
                interior.append(alpha * p0 + beta * p1 + gamma * p2)

        if interior:
            new_points.append(torch.stack(interior, dim=0))

    # === Merge all ===
    new_vertices = torch.cat(new_points, dim=0)
    return new_vertices

def load_spline_weights(model: nn.DataParallel, pt_path: str) -> nn.DataParallel:
    print(f"Loading weights from: {pt_path}")

    current_device = next(model.module.parameters()).device 
    print(f"Current model device: {current_device}")

    try:
        loaded_module = torch.load(pt_path, map_location=current_device)
    except Exception as e:
        print(f"Error loading model from {pt_path}: {e}")
        return model

    try:
        model.module.load_state_dict(loaded_module.state_dict())
        print("Successfully reloaded weights into distance.module")
    except RuntimeError as e:
        print(f"Error during state_dict loading (weights mismatch?): {e}")

    del loaded_module
    
    return model

def save_normal_grid_lens_to_obj(
    front_verts,
    back_verts,
    back_normal,
    grid_size,
    filename
):
    front_verts = np.asarray(front_verts)
    back_verts  = np.asarray(back_verts)
    back_normal = np.asarray(back_normal)

    gx, gy = grid_size   # gx 对应 x 方向采样数，gy 对应 y 方向采样数
    N = gx * gy

    assert front_verts.shape == (N, 3)
    assert back_verts.shape  == (N, 3)
    assert back_normal.shape == (N, 3)

    all_verts = np.vstack([back_verts, front_verts])

    front_normals = np.tile(np.array([[0.0, 0.0, -1.0]]), (N, 1))
    all_normals = np.vstack([back_normal, front_normals])

    # 每个点索引：row * gx + col
    rows, cols = np.mgrid[0:gy-1, 0:gx-1]
    base_idx = (rows * gx + cols).ravel()

    back_faces = np.stack([
        base_idx + 1,
        base_idx + 2,
        base_idx + gx + 2,
        base_idx + gx + 1
    ], axis=1)

    front_faces = np.stack([
        base_idx + N + 1,
        base_idx + N + gx + 1,
        base_idx + N + gx + 2,
        base_idx + N + 2
    ], axis=1)

    # 四条侧边
    edge_x = np.arange(gx - 1)
    edge_y = np.arange(gy - 1)

    # y 最小边
    s1 = np.stack([
        edge_x + N + 1,
        edge_x + N + 2,
        edge_x + 2,
        edge_x + 1
    ], axis=1)

    # y 最大边
    base = (gy - 1) * gx + edge_x
    s2 = np.stack([
        base + N + 1,
        base + 1,
        base + 2,
        base + N + 2
    ], axis=1)

    # x 最小边
    base = edge_y * gx
    s3 = np.stack([
        base + N + 1,
        base + 1,
        base + gx + 1,
        base + gx + N + 1
    ], axis=1)

    # x 最大边
    base = edge_y * gx + gx
    s4 = np.stack([
        base,
        base + N,
        base + gx + N,
        base + gx
    ], axis=1)

    all_faces = np.vstack([back_faces, front_faces])
    side_faces = np.vstack([s1, s2, s3, s4])

    with open(filename, 'w') as f:
        for v in all_verts:
            f.write(f"v {v[1]:.6f} {v[2]:.6f} {v[0]:.6f}\n")

        for n in all_normals:
            f.write(f"vn {n[1]:.6f} {n[2]:.6f} {n[0]:.6f}\n")

        for a, b, c, d in all_faces:
            f.write(f"f {a}//{a} {b}//{b} {c}//{c} {d}//{d}\n")

        for a, b, c, d in side_faces:
            f.write(f"f {a} {b} {c} {d}\n")

def initial_xy_point(loader):
    sample_type = loader.data['sample_point_para']['type']
    sample_point_num = loader.data['sample_point_para']['point_num']
    sample_domain = loader.data['sample_point_para']['sample_domain']
    x_scalar = sample_domain[1]-sample_domain[0]
    y_scalar = sample_domain[3]-sample_domain[2]

    if sample_type == "triangle":
        # 根据 sample_point_num 估算每行的点数
        sample_point_num = float(sample_point_num)
        x_sample_num = np.sqrt(np.sqrt(3)/2*sample_point_num*x_scalar/y_scalar)
        y_sample_num = x_sample_num * 2 / np.sqrt(3) * y_scalar/x_scalar
        x_sample_num = int(x_sample_num) + 1
        y_sample_num = int(y_sample_num) + 1
        if y_sample_num%2 == 0:
            y_sample_num = y_sample_num+1
        x_step = x_scalar/float(x_sample_num-1)
        y_step = y_scalar/float(y_sample_num-1)

        points = []
        
        for i in range(y_sample_num):
            y = (i) * y_step
            if i % 2 == 0:         
                for j in range(x_sample_num):
                    x = (j) * x_step
                    points.append([x, y])
            else:
                points.append([0.0, y])
                # 中间点（比偶数行少一个）
                for j in range(x_sample_num - 1):
                    x = j * x_step + x_step / 2.0
                    points.append([x, y])
                # 右边界点
                points.append([x_scalar, y])
        return np.array(points, dtype=float)
    else:
        raise ValueError(f"Unknown sample_type: {sample_type}.")

def initial_proj_point(loader,points):
    sample_domain = loader.data["sample_point_para"]["sample_domain"]
    proj_domain = loader.data["len_para"]["proj_domain"]

    sx_min, sx_max, sy_min, sy_max = sample_domain
    px_min, px_max, py_min, py_max = proj_domain

    sample_w = sx_max - sx_min
    sample_h = sy_max - sy_min
    proj_w = px_max - px_min
    proj_h = py_max - py_min

    scale = min(proj_w / sample_w, proj_h / sample_h)

    sample_cx = (sx_min + sx_max) / 2
    sample_cy = (sy_min + sy_max) / 2
    proj_cx = (px_min + px_max) / 2
    proj_cy = (py_min + py_max) / 2

    proj_points = np.empty_like(points, dtype=float)
    proj_points[:, 0] = (points[:, 0] - sample_cx) * scale + proj_cx
    proj_points[:, 1] = (points[:, 1] - sample_cy) * scale + proj_cy
    return proj_points

def find_boundary_points(loader,points):
    triangulation = Delaunay(points)
    triangles = triangulation.simplices
    triangles_tensor = torch.tensor(triangles, dtype=torch.int).to("cuda:0")

    sample_point_num = loader.data['sample_point_para']['point_num']
    sample_domain = loader.data['sample_point_para']['sample_domain']
    x_scalar = sample_domain[1]-sample_domain[0]
    y_scalar = sample_domain[3]-sample_domain[2]
    tol = 0.1 * np.sqrt((x_scalar*y_scalar)/sample_point_num)

    x = points[:, 0]
    y = points[:, 1]

    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)

    left_mask   = np.isclose(x, xmin, atol=tol)
    right_mask  = np.isclose(x, xmax, atol=tol)
    bottom_mask = np.isclose(y, ymin, atol=tol)
    top_mask    = np.isclose(y, ymax, atol=tol)

    edge_count = {}
    for tri in triangles:
        a, b, c = tri

        edges = [
            tuple(sorted((a, b))),
            tuple(sorted((b, c))),
            tuple(sorted((c, a))),
        ]

        for e in edges:
            edge_count[e] = edge_count.get(e, 0) + 1

    boundary_edges = [e for e, cnt in edge_count.items() if cnt == 1]

    boundary_vertex_mask = np.zeros(len(points), dtype=bool)
    for i, j in boundary_edges:
        boundary_vertex_mask[i] = True
        boundary_vertex_mask[j] = True

    left_mask   = left_mask   & boundary_vertex_mask
    right_mask  = right_mask  & boundary_vertex_mask
    bottom_mask = bottom_mask & boundary_vertex_mask
    top_mask    = top_mask    & boundary_vertex_mask

    left_idx = np.where(left_mask)[0]
    right_idx = np.where(right_mask)[0]
    bottom_idx = np.where(bottom_mask)[0]
    top_idx = np.where(top_mask)[0]

    return triangles_tensor, left_idx, right_idx, bottom_idx, top_idx

def find_boundary_vertex(points,triangles_tensor):
    triangles = triangles_tensor.detach().cpu().numpy()

    edge_count = {}
    for tri in triangles:
        a, b, c = tri
        edges = [
            tuple(sorted((a, b))),
            tuple(sorted((b, c))),
            tuple(sorted((c, a))),
        ]
        for e in edges:
            edge_count[e] = edge_count.get(e, 0) + 1

    boundary_edges = [e for e, cnt in edge_count.items() if cnt == 1]
    boundary_vertex_mask = np.zeros(len(points), dtype=bool)

    for i, j in boundary_edges:
        boundary_vertex_mask[i] = True
        boundary_vertex_mask[j] = True
    return boundary_vertex_mask

def pic_print_txt(loader):
    image = Image.open(loader.data['image_path']).convert("L")
    image_array = np.array(image)
    m, n = image_array.shape
    print("行数 m =", m)
    print("列数 n =", n)
    txt_path = loader.data['image_txt']
    save_array_to_txt(image_array, txt_path)

def initialize_caustic_nn(loader, device_ids, if_requires_grad):
    model_path = loader.data["model_path"]
    model_para = loader.data["model_para"]
    
    model = CausticNN(
        input_dim=model_para["input_dim"],
        unfreeze_scalar=model_para["unfreeze_scalar"],
        embedder_para=model_para["embedder_para"],
        output_dim=model_para["output_dim"],
        layer_dims=model_para["layer_dims"],
        skip_layers=model_para["skip_layers"],
        freeze_layers=model_para["freeze_layers"],
        scalar = loader.data["sample_point_para"]["scalar"],
        if_adafreq=model_para["if_adafreq"]
    ).to("cuda:0")
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    if model_para["if_initial"]:
        try:
            checkpoint = torch.load(model_path, weights_only=True)  # 安全模式
            model.load_state_dict(checkpoint)
        except RuntimeError as e:
            print(f"安全加载失败（可能包含自定义对象），尝试非安全模式: {e}")
            checkpoint = torch.load(model_path, weights_only=False)  # 回退到非安全模式（需确保文件可信）
            model.load_state_dict(checkpoint)

    for param in model.parameters():
        param.requires_grad = if_requires_grad
    for idx in model_para["freeze_layers"]:
        for param in model.module.layers[idx].parameters():
            param.requires_grad = False
    
    return model

def set_if_require_grad(loader,model,if_requires_grad):
    freeze_layers = loader.data["model_para"]["freeze_layers"]
    for param in model.parameters():
        param.requires_grad = if_requires_grad
    for idx in freeze_layers:
        for param in model.module.layers[idx].parameters():
            param.requires_grad = False

def print_model_structure(model):
    for name, module in model.named_modules():
        if name == '':
            continue
        print(f"Layer: {name}")
        print(f"Type: {type(module).__name__}")
        
        if isinstance(module, torch.nn.Linear):
            print(f"Input features: {module.in_features}")
            print(f"Output features: {module.out_features}")
        
        elif isinstance(module, (torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.Tanh)):
            print(f"Activation: {type(module).__name__}")
        
        print("-" * 50)
