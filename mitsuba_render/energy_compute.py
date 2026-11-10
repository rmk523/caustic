import os
from PIL import Image
import numpy as np

script_abs_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_abs_path)

def gamma_correction_and_energy_sum(img_path, gamma):
    # 1️⃣ 读取图像并转为灰度
    img = Image.open(img_path).convert("L")
    
    # 2️⃣ 转换为 numpy 数组（float64 防止溢出）
    img_array = np.asarray(img, dtype=np.float64)
    
    # 3️⃣ 进行伽马矫正
    img_corrected = np.power(img_array / 255.0, gamma) * 255.0
    
    # 4️⃣ 计算光能强度总和（灰度图像的所有像素值之和）
    energy_sum = np.sum(img_corrected)
    
    # 5️⃣ 保证像素值在合理范围内（0 到 255）
    img_corrected = np.clip(img_corrected, 0, 255)
    
    # 6️⃣ 转换回图片并保存（可选）
    img_corrected = Image.fromarray(img_corrected.astype(np.uint8))
    img_corrected.save("corrected_image.png")
    
    return energy_sum

# 使用示例
pic_name = "ildTL"
name = "ildTL"
light_type = "parallel"
compute_type = "rlot_rot_render"

img1 = script_dir + "/../data/pic_data/{}.png".format(pic_name)
img2 = script_dir + "/irradiance_output/{}/{}/{}.png".format(light_type, compute_type, name)

energy_sum = gamma_correction_and_energy_sum(img1, 1)
print(f"光能强度总和: {energy_sum}")

energy_sum = gamma_correction_and_energy_sum(img2, 1)
print(f"光能强度总和: {energy_sum}")
