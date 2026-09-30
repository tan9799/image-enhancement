import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# ============================================================
# 作业1：图像空域增强——胸片伪彩色处理（批量版）
# 不使用 OpenCV
# 使用：Pillow + NumPy + Matplotlib
# ============================================================

# 1. 文件路径配置
# 这里填入你要处理的两张胸片名字
image_list = ["chest_1.png", "chest_2.png"]
output_dir = "result"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. 建立伪彩色映射 LUT
def build_lut(anchors):
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        for j in range(len(anchors) - 1):
            x0, c0 = anchors[j]
            x1, c1 = anchors[j + 1]
            if x0 <= i <= x1:
                t = (i - x0) / (x1 - x0) if x1 > x0 else 0.0
                lut[i] = [
                    int(round(c0[0] + (c1[0] - c0[0]) * t)),
                    int(round(c0[1] + (c1[1] - c0[1]) * t)),
                    int(round(c0[2] + (c1[2] - c0[2]) * t)),
                ]
                break
    return lut

# 优化后的伪彩色锚点（蓝-绿-红-黄风格）
anchors = [
    (0,   (0,   0,   60)),    # 深蓝黑
    (40,  (0,   70,  180)),   # 蓝色
    (80,  (0,   200, 230)),   # 青色
    (120, (0,   220, 120)),   # 绿色
    (160, (180, 230, 0)),     # 黄绿色
    (190, (255, 180, 0)),     # 橙色
    (215, (255, 80,  0)),     # 红橙色
    (235, (200, 0,   0)),     # 深红色
    (255, (255, 230, 100)),   # 亮黄
]
lut = build_lut(anchors)

# 3. 循环处理每一张图片
for img_name in image_list:
    print(f"\n====================================")
    print(f"正在处理：{img_name}")
    print(f"====================================")
    
    try:
        image = Image.open(img_name)
    except FileNotFoundError:
        print(f"找不到图片 {img_name}，请检查文件名和路径！")
        continue

    # 转换为灰度图并转为 NumPy 数组
    gray_image = image.convert("L")
    gray = np.array(gray_image)

    # 百分位裁剪归一化 (1%~99%)
    low, high = np.percentile(gray, (1, 99))
    if high > low:
        gray_normalized = (gray.astype(np.float32) - low) / (high - low) * 255
        gray_normalized = np.clip(gray_normalized, 0, 255).astype(np.uint8)
    else:
        gray_normalized = gray.copy()

    # 伪彩色映射
    pseudo_color = lut[gray_normalized]

    # 截取文件名（去掉扩展名），用于结果命名，例如 chest_1
    base_name = os.path.splitext(img_name)[0]

    # 保存灰度图
    gray_output = os.path.join(output_dir, f"{base_name}_原始灰度.png")
    gray_image.save(gray_output)

    # 保存伪彩色图
    pseudo_image = Image.fromarray(pseudo_color, mode="RGB")
    pseudo_output = os.path.join(output_dir, f"{base_name}_伪彩色增强.png")
    pseudo_image.save(pseudo_output)

    # 绘制对比图并保存
    plt.figure(figsize=(18, 6))

    # 左：原始胸片
    plt.subplot(1, 3, 1)
    plt.imshow(gray_normalized, cmap="gray")
    plt.title(f"Original - {base_name}", fontsize=16, fontweight="bold")
    plt.axis("off")

    # 中：伪彩色胸片
    plt.subplot(1, 3, 2)
    plt.imshow(pseudo_color)
    plt.title("Pseudo-color Enhanced", fontsize=16, fontweight="bold")
    plt.axis("off")

    # 右：颜色映射图例
    plt.subplot(1, 3, 3)
    gradient = np.tile(np.arange(256, dtype=np.uint8), (40, 1))
    gradient_rgb = lut[gradient]
    plt.imshow(gradient_rgb, aspect="auto")
    plt.title("Color Mapping Legend", fontsize=16, fontweight="bold")
    plt.xlabel("Gray value")
    plt.yticks([])
    plt.xticks(
        [0, 40, 80, 120, 160, 190, 215, 235, 255],
        ["0", "40", "80", "120", "160", "190", "215", "235", "255"]
    )

    plt.tight_layout()
    comparison_output = os.path.join(output_dir, f"{base_name}_对比图.png")
    plt.savefig(comparison_output, dpi=300, bbox_inches="tight")
    plt.close()  # 关闭当前画布，防止内存溢出

    print(f"处理完成，已生成：")
    print(f"  - {gray_output}")
    print(f"  - {pseudo_output}")
    print(f"  - {comparison_output}")

print("\n====================================")
print("       所有图片处理完成！")
print("====================================")