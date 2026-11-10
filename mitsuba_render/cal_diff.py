import os
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
import argparse

script_abs_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_abs_path)

def image_gray_sum_and_rrmse(img1_path, img2_path):
    # 1. 读取并转为灰度
    img1 = Image.open(img1_path).convert("L")
    img2 = Image.open(img2_path).convert("L")
    img1 = img1.resize((640,640))
    img2 = img2.resize((640,640))
    # 2. 转 numpy
    I1 = np.asarray(img1, dtype=np.float64)
    I2 = np.asarray(img2, dtype=np.float64)

    assert I1.shape == I2.shape, "Images must have the same shape"

    I2 = I2

    # 3. 灰度和
    gray_sum_1 = I1.sum()
    gray_sum_2 = I2.sum()

    # 4. RRMSE / RRMSD
    numerator = np.sqrt(np.mean((I2 - I1) ** 2))
    denominator = np.sqrt(np.mean(I1 ** 2))

    rrmse = numerator / (denominator + 1e-12)

    return gray_sum_1, gray_sum_2, rrmse


def calculate_mae_and_normalized_mae(img1_path, img2_path):
    # 1. 读取并转为灰度
    img1 = Image.open(img1_path).convert("L")
    img2 = Image.open(img2_path).convert("L")
    img1 = img1.resize((640,640))
    img2 = img2.resize((640,640))
    # 2. 转 numpy
    I1 = np.asarray(img1, dtype=np.float64)
    I2 = np.asarray(img2, dtype=np.float64)

    I2 = I2

    assert I1.shape == I2.shape, "Images must have the same shape"

    # 3. 计算 MAE
    mae = np.mean(np.abs(I2 - I1))

    # 4. 归一化 MAE
    max_pixel_value = I2.max()
    normalized_mae = mae / (max_pixel_value + 1e-12)

    return mae, normalized_mae


def calculate_ssim_value(img1_path, img2_path):
    # 1. 读取并转为灰度
    img1 = Image.open(img1_path).convert("L")
    img2 = Image.open(img2_path).convert("L")
    img1 = img1.resize((640,640))
    img2 = img2.resize((640,640))
    # 2. 转 numpy
    I1 = np.asarray(img1, dtype=np.float64)
    I2 = np.asarray(img2, dtype=np.float64)

    I2 = I2

    assert I1.shape == I2.shape, "Images must have the same shape"

    # 3. 计算 SSIM
    # data_range 建议使用两图共同范围，避免缩放后超出 255
    data_range = max(I1.max(), I2.max()) - min(I1.min(), I2.min())
    if data_range == 0:
        data_range = 1.0

    ssim_value = ssim(I1, I2, data_range=data_range)

    return ssim_value


def save_abs_diff_image(img1_path, img2_path, out_path):
    img1 = Image.open(img1_path).convert("L")
    img2 = Image.open(img2_path).convert("L")
    img1 = img1.resize((640,640))
    img2 = img2.resize((640,640))
    I1 = np.asarray(img1, dtype=np.float64)
    I2 = np.asarray(img2, dtype=np.float64)

    assert I1.shape == I2.shape, "Images must have the same shape"

    diff = np.abs(I2 - I1)*3

    # 归一化到 0-255，方便观察
    # diff_norm = diff / (diff.max() + 1e-12) * 255.0
    diff_img = Image.fromarray(diff.astype(np.uint8))

    diff_img.save(out_path)

parser = argparse.ArgumentParser()
parser.add_argument('--pic_name', type=str, required=True, help='图片名称')
parser.add_argument('--light_type', type=str, required=True, help='光照类型')
parser.add_argument('--compute_type', type=str, required=True, help='计算方法')
args = parser.parse_args()

# pic_name = "ildTL"
# name = "ildTL"
# light_type = "parallel"
# compute_type = "ot_rot_render"

pic_name = args.pic_name
name = args.pic_name
light_type = args.light_type
compute_type = args.compute_type

img1 = script_dir + "/../data/pic_data/{}.png".format(pic_name)
img2 = script_dir + "/irradiance_output/{}/{}/{}.png".format(light_type, compute_type, name)
abs_img = script_dir + "/irradiance_output/{}/{}/{}_abs.png".format(light_type, compute_type, name)
txt_path = script_dir + "/irradiance_output/{}/{}/{}.txt".format(light_type, compute_type, name)

sum1, sum2, rrmse = image_gray_sum_and_rrmse(img1, img2)
mae, normalized_mae = calculate_mae_and_normalized_mae(img1, img2)
ssim_value = calculate_ssim_value(img1, img2)

save_abs_diff_image(img1, img2, abs_img)

output_text = (
    f"Gray sum (Image 1): {sum1:.6e}\n"
    f"Gray sum (Image 2): {sum2:.6e}\n"
    f"RRMSE / RRMSD      : {rrmse:.6e}\n"
    f"MAE                : {mae:.6e}\n"
    f"Normalized MAE     : {normalized_mae:.6e}\n"
    f"SSIM               : {ssim_value:.6e}"
)

print(output_text)

# 保存到 txt 文件
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(output_text)