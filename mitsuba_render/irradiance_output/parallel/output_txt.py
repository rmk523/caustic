from PIL import Image
import numpy as np
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
image_path = os.path.join(
    ROOT, "mitsuba_render", "irradiance_output", "parallel",
    "rlot_rot_render", "ildTL.png",
)

image = Image.open(image_path).convert("L")
gray = np.array(image)

output_path = os.path.splitext(image_path)[0] + "_pic.txt"
np.savetxt(output_path, gray, fmt="%4d", delimiter=" ")

print(f"图片尺寸：{gray.shape[1]} × {gray.shape[0]}")
print(f"TXT 已保存至：{output_path}")