import os
from os.path import realpath, join
import mitsuba as mi
import drjit as dr

# 环境设置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')  # 选择支持自动微分的变体

script_abs_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_abs_path)

# 1054 287 215(118.5) 580(325)
light_intensity = 325# 光照强度，用户可设置
fbl = 640
name = "ustc_centerL"
light_type = "point"
compute_type = "ot"

# 3. 创建投影平面
projection_plane = {
    'type': 'obj',
    'filename': script_dir + "/scenes/mesh/rectangle_point.obj",  # 平面文件路径
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
    target=[20, 160, 20],
    origin=[20, 140, 20],
    up=[0, 0, 1]
)

sensor = {
    'type': 'perspective',
    'near_clip': 1,
    'far_clip': 500,
    'fov': 90,  # 设置适当的视场角度
    'to_world': sensor_to_world,
    'sampler': {'type': 'independent', 'sample_count': 5120},
    'film': {
        'type': 'hdrfilm',
        'width': fbl,
        'height': fbl,
        'pixel_format': 'luminance',  # 灰度渲染也使用rgb格式
        'rfilter': {'type': 'box'},
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
    **scene_materials,

    'spot-light': {
        'type': 'spot',
        'to_world': mi.ScalarTransform4f()
            .translate([20, -20, 20])  # 光源位置
            .rotate([1, 0, 0], -90),   # 旋转使光源指向y正方向（局部z轴默认是光锥方向）
        'intensity': {'type': 'spectrum', 'value': light_intensity},#1600
        'cutoff_angle': 70.0,  # 光锥的半角（从中心到边缘的角度）
        'beam_width': 70.0,    # 内部完全照亮区域的半角（可选，用于柔化边缘）
    },

    # 投影平面
    'projection-plane': projection_plane,
    
    'glass-lens': {
        'type': 'obj',
        'filename': script_dir + f"/../output_mesh/{light_type}/{compute_type}/{name}.obj",
        'to_world': mi.ScalarTransform4f().translate([0, 0, 0]),
        'bsdf': {
            'type': 'dielectric',
            'int_ior': 1.5,    # 内部折射率（玻璃）
            'ext_ior': 1.0,    # 外部折射率（空气）
        },
    },
})

# 7. 渲染设置（bw灰度渲染）
sensor_film = sensor['film']
sensor_film['pixel_format'] = 'gray'  # 设置为灰度渲染
sensor.update()

# 8. 渲染并保存图像 (保存为 PNG 格式)
image = mi.render(scene)
save_path = script_dir + "/final_output/{}/{}/{}.png".format(light_type, compute_type, name)
os.makedirs(os.path.dirname(save_path), exist_ok=True)
mi.util.write_bitmap(save_path.format(light_type, compute_type, name), image)  # 修改为 .png 格式