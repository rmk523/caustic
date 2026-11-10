from PIL import Image
import numpy as np
import os


def save_image_compare_red_blue(
    image_path1,
    image_path2,
    n=3.0,
    output_png_path=None,
    output_txt_path=None
):
    img1 = Image.open(image_path1).convert("L")
    img2 = Image.open(image_path2).convert("L")

    arr1 = np.array(img1, dtype=np.float32)
    arr2 = np.array(img2, dtype=np.float32)

    if arr1.shape != arr2.shape:
        raise ValueError(f"图片尺寸不一致: {arr1.shape} vs {arr2.shape}")

    # 注意：这里是 image2 - image1
    # diff > 0 表示 image2 更亮，显示红色
    # diff < 0 表示 image2 更暗，显示蓝色
    diff = arr2 - arr1

    folder = os.path.dirname(image_path2)
    name1 = os.path.splitext(os.path.basename(image_path1))[0]
    name2 = os.path.splitext(os.path.basename(image_path2))[0]

    if output_png_path is None:
        output_png_path = os.path.join(folder, f"{name2}_compare_{name1}_red_blue_x{n}.png")

    if output_txt_path is None:
        output_txt_path = os.path.join(folder, f"{name2}_minus_{name1}_diff.txt")

    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)

    # 保存 TXT：原始差值，不乘 n
    with open(output_txt_path, "w") as f:
        h, w = diff.shape
        for i in range(h):
            line = " ".join(f"{diff[i, j]:8.3f}" for j in range(w))
            f.write(line + "\n")

    # 差值强度，乘 n 后限制到 [0, 255]
    mag = np.abs(diff) * n
    mag = np.clip(mag, 0, 255)

    # 白色底图
    rgb = np.ones((diff.shape[0], diff.shape[1], 3), dtype=np.float32) * 255

    # image2 > image1：红色
    red_mask = diff > 0
    rgb[red_mask, 0] = 255
    rgb[red_mask, 1] = 255 - mag[red_mask]
    rgb[red_mask, 2] = 255 - mag[red_mask]

    # image2 < image1：蓝色
    blue_mask = diff < 0
    rgb[blue_mask, 0] = 255 - mag[blue_mask]
    rgb[blue_mask, 1] = 255 - mag[blue_mask]
    rgb[blue_mask, 2] = 255

    rgb = np.rint(rgb)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    Image.fromarray(rgb, mode="RGB").save(output_png_path)

    print("diff = image2 - image1")
    print("diff max:", diff.max())
    print("diff min:", diff.min())
    print("diff sum:", diff.sum())
    print("diff mean:", diff.mean())
    print("png saved to:", output_png_path)
    print("txt saved to:", output_txt_path)

    return output_png_path, output_txt_path


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image1 = os.path.join(ROOT, "data", "pic_data", "bird.png")
image2 = os.path.join(
    ROOT, "result_sp_data", "bird_tri_standard_sp_parallel_light_rl",
    "render", "0", "0.png",
)

save_image_compare_red_blue(
    image1,
    image2,
    n=3
)