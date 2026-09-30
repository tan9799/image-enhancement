# 图像增强：假彩色合成与伪彩色映射

本项目为图像增强课程作业，基于 **Python + Pillow + NumPy + Matplotlib** 实现两类图像增强实验：

1. **咖啡商品图假彩色合成**：通过多通道线性混合、伽马校正与后处理，提升咖啡商品图的“香浓感、高级感、吸睛度”。
2. **胸片伪彩色批量处理**：通过百分位裁剪归一化与自定义 LUT 伪彩色映射，增强医学灰度图像中肺野、软组织和骨骼的视觉区分度。

项目全程**不使用 OpenCV**，仅使用 Pillow、NumPy 和 Matplotlib 完成图像读取、处理、保存与可视化。

---

## 一、环境依赖

- Python 3.8+
- NumPy
- Matplotlib
- Pillow

安装依赖：

```bash
pip install numpy matplotlib Pillow
