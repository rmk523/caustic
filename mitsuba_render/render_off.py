import os
from os.path import realpath, join
import mitsuba as mi
import drjit as dr
import numpy as np
from PIL import Image
import argparse

# 环境设置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')  # 选择支持自动微分的变体

script_abs_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_abs_path)

parser = argparse.ArgumentParser()
parser.add_argument('--pic_name', type=str, required=True, help='图片名称')
parser.add_argument('--light_type', type=str, required=True, help='光照类型')
parser.add_argument('--compute_type', type=str, required=True, help='计算方法')
args = parser.parse_args()

# 2. 创建平行光源  37.57 18.72 3.64 9.36
fbl = 640
# pic_name = "ildTL"
# name = "ildTL"
# light_type = "parallel"
# compute_type = "ot_rot_render"

pic_name = args.pic_name
name = args.pic_name
light_type = args.light_type
compute_type = args.compute_type

if pic_name == "bird":
    parallel_light_intensity = 37.57/4
if pic_name == "butterfly":
    parallel_light_intensity = 18.72/4
if pic_name == "ildTL":
    parallel_light_intensity = 3.64/4
if pic_name == "ustc_centerL":
    parallel_light_intensity = 9.36/4

# 3. 创建投影平面
projection_plane = {
    'type': 'obj',
    'filename': script_dir + "/scenes/mesh/rectangle.obj",  # 平面文件路径
    'to_world': mi.ScalarTransform4f(),  # 位于 z=160，并设置为[-60, 100]*[0, 160]
    'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [1, 1, 1]}},
}

# 4. 定义材质
scene_materials = {
    'white-bsdf': {
        'type': 'diffuse',  # 设置为漫反射材质
        'reflectance': {'type': 'rgb', 'value': [1, 1, 1]},  # 设置反射率为白色 (RGB)
    },
}

# 5. 摄像头设置
sensor_to_world = mi.ScalarTransform4f().look_at(
    target=[20, 160, 80],
    origin=[20, 80, 80],
    up=[0, 0, 1]
)

sensor = {
    'type': 'perspective',
    'near_clip': 1,
    'far_clip': 500,
    'fov': 90,  # 设置适当的视场角度
    'to_world': sensor_to_world,
    'sampler': {'type': 'independent', 'sample_count': 40960},
    'film': {
        'type': 'hdrfilm',
        'width': fbl,
        'height': fbl,
        'pixel_format': 'luminance',  # 灰度渲染也使用rgb格式
        'rfilter': {'type': 'gaussian'},
    },
}

# 6. 场景组装
scene = mi.load_dict({
    'type': 'scene',
    'sensor': sensor,
    'integrator': {
        'type': 'ptracer',
        'samples_per_pass': 256,
        'max_depth': 4,
        'hide_emitters': False,
    },
    # 材质定义
    **scene_materials,  # 将材质定义添加到场景字典中

    'directional-light': {
        'type': 'directional',
        'direction': [0, 1, 0],  # y>0 方向
        'irradiance': {'type': 'rgb', 'value': [parallel_light_intensity]*3},
    },

    # 可见矩形 (不发光)
    'dangban1': {
        'type': 'obj',
        'filename': script_dir + '/scenes/mesh/dangban1.obj',
        'to_world': mi.ScalarTransform4f().translate([0, -100, 0]),
        'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [1,1,1]}},
    },
    'dangban2': {
        'type': 'obj',
        'filename': script_dir + '/scenes/mesh/dangban2.obj',
        'to_world': mi.ScalarTransform4f().translate([0, -100, 0]),
        'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [1,1,1]}},
    },
    'dangban3': {
        'type': 'obj',
        'filename': script_dir + '/scenes/mesh/dangban3.obj',
        'to_world': mi.ScalarTransform4f().translate([0, -100, 0]),
        'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [1,1,1]}},
    },
    # 投影平面
    'projection-plane': projection_plane,
    
    'glass-lens': {
        'type': 'obj',
        'filename': script_dir + "/../output_mesh/{}/{}/{}.obj".format(light_type, compute_type, name),
        'to_world': mi.ScalarTransform4f().translate([0, 0, 0]),
        'bsdf': {
            'type': 'dielectric',
            'int_ior': 1.5,    # 内部折射率（玻璃）
            'ext_ior': 1.0,    # 外部折射率（空气）
            'specular_reflectance': { 'type': 'spectrum', 'value': 0 }
        },
    },
})

# 7. 渲染设置（bw灰度渲染）
sensor_film = sensor['film']
sensor_film['pixel_format'] = 'gray'  # 设置为灰度渲染
sensor.update()

# 8. 渲染并保存图像 (保存为 PNG 格式)
image = mi.render(scene)
save_path_irra = script_dir + "/irradiance_output/{}/{}/{}.png".format(light_type, compute_type, name)
os.makedirs(os.path.dirname(save_path_irra), exist_ok=True)

img_np = np.array(image)*255
print(img_np.max(),img_np.min(),img_np.sum())
img_np = np.squeeze(img_np)
gray_direct = np.rint(img_np)
gray_direct = np.clip(gray_direct, 0, 255).astype(np.uint8)
Image.fromarray(gray_direct, mode="L").save(save_path_irra)
