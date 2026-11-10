from PIL import Image
import numpy as np
import os


def png_gray_to_txt(png_path, txt_path):
    """
    读取 PNG 图片，将每个像素的灰度值保存到 txt。
    每个灰度值占 4 个字符宽度，不足补空格。
    
    例如：
       0  12 255
    """
    # 读取图片并转为 8-bit 灰度图
    img = Image.open(png_path).convert("L")

    # 转为 numpy 数组，shape = (H, W)
    gray = np.array(img, dtype=np.uint8)

    # 创建输出目录
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)

    # 写入 txt
    with open(txt_path, "w") as f:
        h, w = gray.shape
        for i in range(h):
            line = "".join(f"{int(gray[i, j]):4d}" for j in range(w))
            f.write(line + "\n")


# 示例
name = "ustc_centerL"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_dir = os.path.join(ROOT, "mitsuba_render", "irradiance_output", "parallel", "rlot_rot_render")
png_path = os.path.join(out_dir, f"{name}.png")
txt_path = os.path.join(out_dir, f"{name}_output_gray.txt")

png_gray_to_txt(png_path, txt_path)