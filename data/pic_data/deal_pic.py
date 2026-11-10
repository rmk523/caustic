from PIL import Image
import os
import argparse


def center_crop_resize(input_path, output_path=None, crop_size=600, resize_size=640):
    img = Image.open(input_path)
    w, h = img.size

    if w < crop_size or h < crop_size:
        raise ValueError(f"图片尺寸太小，当前尺寸为 {w}x{h}，无法裁剪 {crop_size}x{crop_size}")

    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size

    img_crop = img.crop((left, top, right, bottom))
    img_resize = img_crop.resize((resize_size, resize_size), Image.Resampling.LANCZOS)

    if output_path is None:
        folder = os.path.dirname(input_path)
        name, ext = os.path.splitext(os.path.basename(input_path))
        output_path = os.path.join(folder, f"{name}_center600_resize640{ext}")

    img_resize.save(output_path)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(here, "ildTL.png")
    output_path = os.path.join(here, "ildTL_test.png")

    center_crop_resize(input_path, output_path)