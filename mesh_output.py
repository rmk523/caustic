import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from utils.ConfigLoader import sp_JSONDataProcessor
import utils.data_utils as data_utils
import utils.basic_function as basic_function
import utils.optimization as optimization
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device_ids = [0]
torch.set_num_threads(8)

if __name__=="__main__":
    script_abs_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_abs_path)
    
    json_path = os.path.join(script_dir,"configs/_rebuild_settings.json")

    parser = argparse.ArgumentParser()
    parser.add_argument('--pic_name', type=str, required=True, help='图片名称')
    parser.add_argument('--light_type', type=str, required=True, help='光照类型')
    parser.add_argument('--compute_type', type=str, required=True, help='计算方法')
    args = parser.parse_args()

    name = args.pic_name
    light_type = args.light_type
    compute_type = args.compute_type

    if compute_type == "ot":
        opti_type = "ot"
        data_path = "control_point/ot_opti_0/model_90000.pth"
    if compute_type == "rlot":
        opti_type = "rl"
        data_path = "control_point/ot_opti_1/model_30000.pth"
    if compute_type == "rlot_rot":
        opti_type = "rl"
        data_path = "control_point/op_opti_2/model_30000.pth"

    if compute_type == "ot_render":
        opti_type = "ot"
        data_path = "ray_tracing/control_point/ot_gauss/model_2400.pth"
    if compute_type == "rlot_render":
        opti_type = "rl"
        data_path = "ray_tracing/control_point/ot_gauss/model_2400.pth"
    if compute_type == "ot_rot_render":
        opti_type = "ot"
        data_path = "ray_tracing/control_point/gauss/model_2400.pth"
    if compute_type == "rlot_rot_render":
        opti_type = "rl"
        data_path = "ray_tracing/control_point/gauss/model_2400.pth"

    loader = sp_JSONDataProcessor(json_path)
    loader.data["light_para"]["light_type"] = f"{light_type}_light"
    
    distance = data_utils.initialize_caustic_spline(loader, device_ids, if_requires_grad=True)
    data_path = os.path.join(
        script_dir, "result_spline",
        f"{name}_tri_standard_sp_parallel_light_{opti_type}", data_path,
    )
    print(data_path)
    distance.module.load_spline(data_path, device="cuda:0")

    sample_domain = torch.tensor(loader.data["sample_point_para"]["sample_domain"], dtype=torch.float32).to("cuda:0")
    display_sample_num = [2401,2401]
    display_x = torch.linspace(sample_domain[0], sample_domain[1], display_sample_num[0]).repeat(display_sample_num[1], 1).to("cuda:0").reshape(-1)
    display_y = torch.linspace(sample_domain[2], sample_domain[3], display_sample_num[1]).repeat(display_sample_num[0], 1).to("cuda:0").t().reshape(-1)
    display_data = torch.stack((display_x, display_y), dim = 1).requires_grad_(True)
    display_z_of_len = torch.full((display_data.shape[0], 1), loader.data["len_para"]["z_of_len"]).to("cuda:0")
    display_xyz_data = torch.cat((display_data,display_z_of_len),dim=1)

    if loader.data['light_para']['light_type'] == "point_light":
        display_vertical_direct = torch.zeros((display_data.shape[0], 3), device=display_data.device)
        display_vertical_direct[:,2] = 1
        display_inside_light_direct, display_pq_grad = basic_function.compute_inside_refraction_jacobian(loader, display_data, display_vertical_direct)
        display_inside_light_direct = display_inside_light_direct.detach().to("cuda:0")
        display_pq_grad = display_pq_grad.detach().to("cuda:0")
        
    if loader.data['light_para']['light_type'] == "parallel_light":
        display_inside_light_direct = torch.zeros((display_data.shape[0], 3)).to("cuda:0")
        display_inside_light_direct[:,2] = 1
        n = display_inside_light_direct.shape[0]
        display_pq_grad = torch.zeros((n, 4)).to("cuda:0")

    display_z = distance(display_data)
    out_put = display_z.sum()
    grad = torch.autograd.grad(out_put, display_data,create_graph=False)[0]

    t1 = (1 + display_pq_grad[:,0].unsqueeze(1) * display_z)
    t2 = (display_pq_grad[:,2].unsqueeze(1) * display_z)
    t3 = (display_pq_grad[:,1].unsqueeze(1) * display_z)
    t4 = (1 + display_pq_grad[:,3].unsqueeze(1) * display_z)

    n_data = torch.cat([t2 * grad[:, 1:2] - t4 * grad[:, 0:1], t3 * grad[:, 0:1] - t1 * grad[:, 1:2],
                            (t1*t4-t3*t2) + (t4 * display_inside_light_direct[:,0:1] - t3 * display_inside_light_direct[:,1:2]) * grad[:,0:1]
                            + (t1 * display_inside_light_direct[:,1:2] - t2 * display_inside_light_direct[:,0:1])*grad[:,1:2]], dim=1)

    with torch.no_grad():
        display_z = distance(display_data)
        display_output_point = display_xyz_data + display_z * display_inside_light_direct

        back_vertices_np = torch.stack([
            display_output_point[:,0].float(),
            display_output_point[:,1].float(),
            display_output_point[:,2].float()
        ], dim=1).detach().cpu().numpy()

        back_normals_np = n_data.detach().cpu().numpy()
        back_normals_np = back_normals_np / np.linalg.norm(back_normals_np, axis=1, keepdims=True)

        if loader.data['light_para']['light_type'] == "point_light":
            front_vertices_tensor = display_xyz_data
        elif loader.data['light_para']['light_type'] == "parallel_light":
            front_vertices_tensor = display_xyz_data[:, :3].clone()
            front_vertices_tensor[:, 2] -= 1
        
        front_vertices_np = torch.stack([
            front_vertices_tensor[:,0].float(),
            front_vertices_tensor[:,1].float(),
            front_vertices_tensor[:,2].float()
        ], dim=1).detach().cpu().numpy()

        obj_dir = os.path.join(script_dir, "output_mesh", light_type, compute_type)
        os.makedirs(obj_dir, exist_ok=True)
        obj_path = os.path.join(obj_dir, f"{name}.obj")

        data_utils.save_normal_grid_lens_to_obj(
            front_verts=front_vertices_np,
            back_verts=back_vertices_np,
            back_normal = back_normals_np,
            grid_size=display_sample_num,
            filename=obj_path
        )
