import os
import sys
import torch
import numpy as np
from utils.ConfigLoader import sp_JSONDataProcessor
import utils.data_utils as data_utils
import utils.basic_function as basic_function
import utils.optimization as optimization
import shutil
import rl_render
import time
from datetime import datetime
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device_ids = [0]
torch.set_num_threads(8)

if __name__=="__main__":
    script_abs_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_abs_path)
    cpp_script_dir = script_dir
    cpp_build_dir = os.path.join(cpp_script_dir,"Optimal_Transport_rl","build")
    cpp_executable = os.path.join(cpp_build_dir, "caustic_GPU")
    
    json_path = os.path.join(script_dir,"configs/_py_sp_render_settings.json")
    loader = sp_JSONDataProcessor(json_path)

    parser = argparse.ArgumentParser()
    parser.add_argument('--pic_name', type=str, required=True, help='图片名称')
    parser.add_argument('--opti_type', type=str, choices=['ot', 'rl'], required=True, help='优化类型')
    args = parser.parse_args()

    pic_name = args.pic_name
    opti_type = args.opti_type

    loader.data["xiushi"] = f"{pic_name}_tri_standard_sp_parallel_light_{opti_type}"
    if opti_type == "ot":
        opti_time = 0
        iter_time = 90000
    if opti_type == "rl":
        opti_time = 1
        iter_time = 30000

    loader.data['image_path'] = loader.root_dir+f"/data/pic_data/{pic_name}.png"

    data_utils.pic_print_txt(loader)
    tensor_image,total_gray_value,pic_size,max_gray = data_utils.load_pic_data(loader)

    distance = data_utils.initialize_caustic_spline(loader, device_ids, if_requires_grad=True)

    # 读取半途模型
    data_path = os.path.join(
        loader.root_dir, "result_spline",
        f"{pic_name}_tri_standard_sp_parallel_light_{opti_type}",
        "control_point", f"ot_opti_{opti_time}", f"model_{iter_time}.pth",
    )
    distance.module.load_spline(data_path, device="cuda:0")

    loader.data["total_opti_para"]["if_triangle_refine"] = False

    regular_x_num = distance.module.num_u - distance.module.p
    regular_y_num = distance.module.num_v - distance.module.p
    print(regular_x_num,regular_y_num)

    render_folder = loader.root_dir+"/result_spline/"+loader.data["xiushi"]+"/ray_tracing/render/ot_gauss/"
    module_folder = loader.root_dir+"/result_spline/"+loader.data["xiushi"]+"/ray_tracing/control_point/ot_gauss/"
    optimization.spline_render(loader, distance, total_gray_value, tensor_image, pic_size, render_folder, module_folder)
    # optimization.spline_render_box(loader, distance, xy_tensor, triangles_tensor, total_gray_value, tensor_image, pic_size)
