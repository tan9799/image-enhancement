import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import os

# ============================================================
# 作业1：题目2 假彩色合成 —— 极致高级感优化版
# ============================================================

input_path = "coffee.png"
output_dir = "result_coffee"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

try:
    image = Image.open(input_path).convert("RGB")
except FileNotFoundError:
    print("找不到咖啡图片！请命名为 coffee.png 并与代码放在同一目录。")
    exit()

img = np.array(image).astype(np.float32)

# -------------------------
# 1. 通道混合矩阵（核心色彩调整）
# -------------------------
r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

# 优化：减少红色增益，适当保留蓝色
# 让背景从“橙黄”变为“香槟金”，桌面保留冷色暗部
new_r = np.clip(r * 1.04 + g * 0.06, 0, 255)
new_g = np.clip(g * 0.96 + r * 0.04, 0, 255)
# 蓝色稍微多保留一点 (0.90)，让暗部不死黑；减去极少红绿，防止偏紫
new_b = np.clip(b * 0.90 - r * 0.02, 0, 255)

false_color = np.stack([new_r, new_g, new_b], axis=2).astype(np.uint8)

# -------------------------
# 2. 伽马校正：提亮暗部，保留桌面纹理（新增）
# -------------------------
# 将像素值归一化到 [0, 1] 范围进行伽马运算
img_float = false_color.astype(np.float32) / 255.0
# gamma = 0.9 会轻微提亮阴影区域，同时不影响高光
gamma = 0.9
img_gamma = np.power(img_float, gamma)
false_color_gamma = (img_gamma * 255.0).astype(np.uint8)

# -------------------------
# 3. 后处理：提升高级质感
# -------------------------
pseudo_image = Image.fromarray(false_color_gamma)

# (1) 对比度提升 1.15 倍，弥补伽马校正带来的轻微发灰
pseudo_image = ImageEnhance.Contrast(pseudo_image).enhance(1.15)

# (2) 饱和度提升 1.10 倍，保持克制的高级感
pseudo_image = ImageEnhance.Color(pseudo_image).enhance(1.10)

# (3) 锐度提升 1.30 倍，凸显杯子釉面质感
pseudo_image = ImageEnhance.Sharpness(pseudo_image).enhance(1.30)

# -------------------------
# 4. 保存与展示
# -------------------------
output_path = os.path.join(output_dir, "coffee_final_high_end.png")
pseudo_image.save(output_path)

plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Coffee", fontsize=18, fontweight="bold")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(pseudo_image)
plt.title("Final High-End Coffee", fontsize=18, fontweight="bold")
plt.axis("off")

plt.tight_layout()
comparison_path = os.path.join(output_dir, "coffee_final_comparison.png")
plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
plt.show()

print("\n最终高级感优化完成！已生成：")
print("1.", output_path)
print("2.", comparison_path)