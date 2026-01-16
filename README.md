# YOLOv8 路面坑洼检测项目

本项目基于 YOLOv8 算法，针对路面坑洼检测任务进行了改进和优化。通过引入 CBAM (Convolutional Block Attention Module) 注意力机制，增强了模型对坑洼特征的提取能力，提高了检测精度。

## 🌟 项目创新点

本项目的主要创新在于**改进了 YOLOv8 的网络结构，集成了 CBAM 注意力模块**。

### CBAM (Convolutional Block Attention Module) 模块详解

CBAM 是一种轻量级的注意力模块，它可以集成到任何卷积神经网络架构中，通过在通道和空间两个维度上推断注意力图，从而增强特征的表达能力。在本项目中，我们将其添加到了 YOLOv8 的主干网络末端，以提升对坑洼特征的感知能力。

我们的 CBAM 实现包含两个连续的子模块：

#### 1. 通道注意力模块 (Channel Attention Module)
通道注意力机制关注 **"是什么" (What)**，旨在发现图像中哪个通道包含了有用的信息（例如坑洼的边缘纹理）。

*   **输入特征**: 对输入特征图 $F$ 分别进行**全局平均池化 (AvgPool)** 和**全局最大池化 (MaxPool)**。
*   **共享感知**: 将两个池化结果分别送入一个共享的多层感知机 (Shared MLP)。该 MLP 包含两层 1x1 卷积：
    1.  第一层将通道数压缩为原来的 $1/16$ (即 `ratio=16`)，减少参数量。
    2.  第二层将通道数恢复。
*   **特征融合**: 将 MLP 输出的两个特征向量相加，并通过 Sigmoid 激活函数生成通道注意力图 $M_c$。
*   **输出**: 将 $M_c$ 与原始特征图 $F$ 相乘，得到通道细化后的特征。

#### 2. 空间注意力模块 (Spatial Attention Module)
空间注意力机制关注**"在哪里" (Where)**，旨在定位包含坑洼的关键像素区域。

*   **输入特征**: 使用经过通道注意力细化后的特征图。
*   **通道压缩**: 在通道维度上分别进行**平均池化**和**最大池化**，得到两个二维特征图，并将它们拼接在一起（通道数为 2）。
*   **空间卷积**: 使用一个 $7 \times 7$ 的卷积层对拼接后的特征图进行卷积操作，融合空间信息。
*   **生成权重**: 通过 Sigmoid 激活函数生成空间注意力图 $M_s$。
*   **输出**: 将 $M_s$ 与输入特征相乘，得到最终的精炼特征。

---

在本项目中，我们修改了 YOLOv8 的 backbone/head 结构，在关键位置插入了 CBAM 模块，使其能够更好地适应路面坑洼这种形状不规则、背景复杂的检测目标。

---

### 🎓 创新点通俗解析 (新手友好版)

如果老师问你：“**你加的这个 CBAM 到底有什么用？**”，你可以这样回答：

把 AI 看作一个**正在找路面坑洼的工人**，原来的 YOLOv8 有时候会**走神**，或者**看错**。CBAM 就像是给这个工人戴上了一副**智能眼镜**，这副眼镜有两个核心功能：

1.  **滤镜功能 (通道注意力 - Channel Attention)**
    *   **问题**: 路面上有很多干扰，比如树影、水渍、井盖，它们看起来有时候很像坑洼。
    *   **解决**: 智能眼镜会自动过滤掉那些“看起来像但其实不是”的颜色和纹理，**只增强“真正的坑洼”所具有的特征**（比如边缘破碎感、特定的灰度变化）。它帮 AI 搞清楚**“重点看什么”**。

2.  **聚光灯功能 (空间注意力 - Spatial Attention)**
    *   **问题**: 照片很大，但坑洼可能只在角落里，或者路边的草丛里根本不可能有坑洼。
    *   **解决**: 智能眼镜会像聚光灯一样，把 AI 的视线**聚焦在路面区域**，忽略掉天空、草地等无关背景。它帮 AI 搞清楚**“重点看哪里”**。

**总结**:
通过这两步（先过滤特征，再聚焦位置），你的模型比原始模型**看得更准**（不容易把影子当坑洼），也**找得更快**（不浪费精力在无关背景上）。这就是为什么加上 CBAM 能提高检测精度的原因。

## 🚀 进阶优化：CBAM-Pro 多尺度注意力模型

为了进一步挖掘注意力机制的潜力，我们提出了 **CBAM-Pro** 改进方案，通过**多尺度部署**和**参数精调**来全方位提升检测性能。

### 1. 多尺度融合 (Multi-Scale Fusion)
*   **改进前**: 仅在 Backbone 主干网络的末端 (P5层) 添加 CBAM。
*   **改进后**: 构建了"全阶段注意力网络"，在检测头 (Head) 的 **P3 (小目标)**、**P4 (中目标)**、**P5 (大目标)** 三个输出层前全部加入 CBAM 模块。
*   **优势**: 路面坑洼尺寸差异巨大，多尺度部署确保了模型不会漏掉细小的坑洼，同时也能精准捕捉巨大的深坑。

### 2. 参数调优 (Micro-Tuning)
*   **改进前**: 压缩比 `ratio=16`。
*   **改进后**: 压缩比 `ratio=8`。
*   **优势**: 降低压缩比意味着保留了更多的通道特征信息。虽然参数量略微增加，但能让模型对坑洼的纹理细节（如边缘的碎石）更加敏感。

### 3. 如何使用 CBAM-Pro
我们提供了专门的配置文件和训练脚本来运行这个增强版模型：

*   **配置文件**: `yolov8n_cbam_pro.yaml`
*   **训练脚本**: `train_pothole_cbam_pro.py`

```bash
# 启动 CBAM-Pro 训练
python train_pothole_cbam_pro.py
```

训练结果将保存在 `runs/detect/exp_cbam_pro` 目录下，您可以将其与 `exp_cbam` 的结果进行对比，形成完美的**消融实验 (Ablation Study)** 报告。

##  实验环境与配置

本项目在以下高性能计算环境下进行了训练与测试，验证了改进算法的有效性。

*   **操作系统**: Windows 10 LTSC (CUDA 12.6)
*   **GPU**: NVIDIA A800-SXM4-40GB (1张, 40GB 显存)
*   **CPU**: 64 核高性能处理器
*   **内存**: 96GB
*   **深度学习框架**: PyTorch (配合 CUDA 12.6)
*   **关键训练参数**:
    *   Batch Size: 256
    *   Image Size: 640

## 📊 数据集介绍

本项目使用的数据集来源于 Roboflow Universe 的公开数据集，专门用于路面坑洼检测任务。

*   **数据集名称**: New Pothole Detection
*   **数据来源**: [Roboflow Universe](https://universe.roboflow.com/smartathon/new-pothole-detection)
*   **图片总数**: 9,240 张
*   **类别**: 单类别 (`Pothole`)
*   **预处理**:
    *   所有图片已统一调整大小为 **640x640**。
    *   应用了自动方向校正 (Auto-orientation)。
*   **数据划分**:
    *   数据集已按比例划分为训练集 (Train)、验证集 (Valid) 和测试集 (Test)，直接适配 YOLOv8 训练格式。

## 📂 目录结构

- `ultralytics/`: YOLOv8 核心源码 (已修改以支持 CBAM)
- `datasets/`: 数据集存放目录
- `train_pothole.py`: 基础版 CBAM 训练脚本
- `train_pothole_cbam_pro.py`: **[新增]** CBAM-Pro 进阶版训练脚本
- `test_result.py`: 测试与推理脚本
- `predict_video.py`: 实时视频/摄像头检测脚本
- `web_demo.py`: Web 可视化演示脚本
- `yolov8n_cbam.yaml`: 基础版 CBAM 模型配置文件
- `yolov8n_cbam_pro.yaml`: **[新增]** CBAM-Pro 进阶版模型配置文件
- `pothole_config.yaml`: 训练参数配置文件
- `README.md`: 项目说明文档

## 🚀 快速开始

### 1. 环境准备

确保已安装 Python 环境（建议 Python 3.8+）和 PyTorch。

```bash
pip install -r requirements.txt
```

*(如果根目录下没有 requirements.txt，请确保安装了 `ultralytics`, `torch`, `opencv-python`, `pandas`, `matplotlib` 等常用库)*

### 2. 数据准备

数据集位于 `datasets/New_pothole_detection.v2i.yolov8` 目录下，包含训练集 (train)、验证集 (valid) 和测试集 (test)。

### 3. 模型训练

#### 方案 A: 基础 CBAM 模型
使用 `train_pothole.py` 脚本开始训练。
```bash
python train_pothole.py
```

#### 方案 B: CBAM-Pro 进阶模型 (推荐)
使用 `train_pothole_cbam_pro.py` 脚本开始训练。
```bash
python train_pothole_cbam_pro.py
```

训练过程中，模型会自动下载预训练权重 `yolov8n.pt` (如果尚未下载)。训练结果（权重、日志）将保存在 `runs/detect/` 目录下。

### 4. 模型测试与推理

训练完成后，可以使用 `test_result.py` 进行测试。请确保脚本中的 `model_path` 指向你训练好的最佳权重文件 (例如 `runs/detect/Pothole_CBAM_Project/exp_cbam_pro/weights/best.pt`)。

```bash
python test_result.py
```

该脚本会加载模型，对测试集图片进行推理，并计算 mAP 等指标。

### 5. 实时视频检测与 Web 演示

我们提供了多种方式来体验模型效果。

#### 🖥️ Web 可视化演示
启动一个基于 Gradio 的网页应用，支持**图片检测**和**视频文件检测**。
```bash
python web_demo.py
```
启动后，在浏览器访问显示的地址（如 `http://127.0.0.1:7860`）。

#### 📹 实时摄像头/视频检测
使用 `predict_video.py` 脚本进行实时推理。

*   **使用摄像头 (默认)**:
    ```bash
    python predict_video.py
    ```
    按 `q` 键退出预览。

*   **检测本地视频文件**:
    ```bash
    python predict_video.py --source "path/to/your/video.mp4"
    ```

## 🛠️ 代码修改细节

为了支持 CBAM，我们对 `ultralytics/nn/modules/conv.py` 进行了修改，添加了 `CBAM`, `ChannelAttention`, `SpatialAttention` 类的定义，并在 `ultralytics/nn/tasks.py` 中确保了模块的正确注册和解析。

## 📝 备注

- 如果遇到显存不足的问题，请在训练脚本中减小 `batch` 大小。
- 训练轮数 `epochs` 可以在训练脚本中进行调整。
