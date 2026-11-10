from PIL import Image
import numpy as np
import os


def save_gamma_corrected_image(image_path, gamma=2.2, suffix="_gamma"):
    """
    读取图片，对其进行 gamma 矫正，并保存在原文件夹中。

    参数:
        image_path: 输入图片路径
        gamma: gamma 值，默认 2.2
        suffix: 输出文件名后缀，默认 "_gamma"

    输出:
        保存 gamma 矫正后的图片，并返回保存路径
    """

    # 读取图片
    img = Image.open(image_path)

    # 保持灰度图为 L，彩色图为 RGB
    if img.mode == "L":
        arr = np.array(img, dtype=np.float32)
    else:
        img = img.convert("RGB")
        arr = np.array(img, dtype=np.float32)

    # 归一化到 0-1
    arr_norm = arr / 255.0

    # gamma 矫正：linear -> display
    arr_gamma = np.power(np.clip(arr_norm, 0, 1), gamma)

    # 转回 0-255
    arr_out = np.rint(arr_gamma * 255.0)
    arr_out = np.clip(arr_out, 0, 255).astype(np.uint8)

    # 构造输出路径
    folder = os.path.dirname(image_path)
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)

    output_path = os.path.join(folder, name + suffix + ext)

    # 保存
    out_img = Image.fromarray(arr_out)
    out_img.save(output_path)

    return output_path


# 示例
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_path = os.path.join(
    ROOT, "mitsuba_render", "irradiance_output", "parallel",
    "rlot_rot_render", "ustc_centerL_gray.png",
)
output_path = save_gamma_corrected_image(input_path)

print("saved to:", output_path)