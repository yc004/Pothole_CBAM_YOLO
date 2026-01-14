import warnings
import sys
import os

# 修复 Windows 下中文乱码问题
if sys.platform.startswith('win'):
    os.system('chcp 65001 >nul')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')
from ultralytics import YOLO

# ================= 坑洼检测项目配置 (基准模型) =================
# 1. 项目命名
PROJECT_NAME = "Pothole_Baseline_Project"

# 2. 配置文件路径
DATASET_YAML = "pothole_config.yaml"  # 数据集配置

# 3. 训练参数
EPOCHS = 100
BATCH_SIZE = 16  # 显存不够改小 (16 -> 8)
IMG_SIZE = 640
DEVICE = '0'


# ==================================================

def train_main():
    # 检查文件是否存在
    if not os.path.exists(DATASET_YAML):
        print(f"❌ 错误: 找不到 {DATASET_YAML}。请打开文件修改 'path' 为你的真实路径！")
        return

    print(f"🚀 开始训练路面坑洼检测模型 (基准模型): {PROJECT_NAME}")

    # 1. 加载模型 (Load Official Model)
    # 直接加载官方 yolov8n.pt，包含结构和预训练权重
    try:
        model = YOLO('yolov8n.pt')
        print("✅ 官方基准模型 yolov8n.pt 加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 3. 开始训练
    # 保持与 train_pothole.py 完全一致的训练参数
    results = model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=PROJECT_NAME,
        name='exp_baseline',  # 实验名称
        patience=20,
        save=True,
        exist_ok=True,
        optimizer='SGD',
        lr0=0.01,
        plots=True,
        workers=2  # Windows下减少workers以防报错
    )

    print(f"🎉 训练完成！结果保存在 {PROJECT_NAME}/exp_baseline 目录下")
    print("💡 提示: 请查看 results.png 查看 mAP 情况，并与 Pothole_CBAM_Project 进行对比")


if __name__ == '__main__':
    train_main()
