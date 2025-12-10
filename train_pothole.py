import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO
import os

# ================= 坑洼检测项目配置 =================
# 1. 项目命名 (训练结果会保存在 runs/detect/Pothole_CBAM_Project)
PROJECT_NAME = "Pothole_CBAM_Project"

# 2. 配置文件路径
DATASET_YAML = "pothole_config.yaml"  # 数据集配置
MODEL_CFG = "yolov8n_cbam.yaml"  # 你的创新点模型结构 (带CBAM)

# 3. 训练参数
EPOCHS = 100  # 建议 100 轮，观察 mAP 曲线
BATCH_SIZE = 16  # 显存不够改小 (8 或 4)
IMG_SIZE = 640
DEVICE = '0'  # 如果用 CPU 请改为 'cpu'


# ==================================================

def train_main():
    # 检查文件是否存在
    if not os.path.exists(DATASET_YAML):
        print(f"❌ 错误: 找不到 {DATASET_YAML}。请打开文件修改 'path' 为你的真实路径！")
        return
    if not os.path.exists(MODEL_CFG):
        print(f"❌ 错误: 找不到 {MODEL_CFG}。请确保你已经生成了带有 CBAM 的模型配置文件。")
        return

    print(f"🚀 开始训练路面坑洼检测模型: {PROJECT_NAME}")

    # 1. 加载模型配置 (Build from YAML)
    # 这会构建一个随机初始化的、带有 CBAM 模块的 YOLO 网络
    model = YOLO(MODEL_CFG)

    # 2. 加载预训练权重 (Transfer Learning)
    # 虽然结构变了，但我们加载官方 yolov8n.pt 的权重，能让模型收敛快很多
    # 不匹配的层（比如新增的 CBAM）会自动跳过，匹配的层会加载
    try:
        model.load('yolov8n.pt')
        print("✅ 预训练权重 yolov8n.pt 加载成功 (部分层)")
    except Exception as e:
        print("⚠️ 预训练权重加载提示 (正常):", e)

    # 3. 开始训练
    results = model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=PROJECT_NAME,
        name='exp_cbam',  # 实验名称
        patience=20,  # 早停
        save=True,
        exist_ok=True,
        optimizer='SGD',  # SGD 对小数据集通常更稳
        lr0=0.01,
        plots=True  # 自动画出混淆矩阵和 PR 曲线
    )

    print(f"🎉 训练完成！结果保存在 {PROJECT_NAME}/exp_cbam 目录下")
    print("💡 提示: 请查看 results.png 查看 mAP 提升情况")


if __name__ == '__main__':
    train_main()