import os
import sys
import warnings

# 修复 Windows 下中文乱码问题
if sys.platform.startswith("win"):
    # 尝试设置控制台编码为 UTF-8
    os.system("chcp 65001 >nul")
    # 强制 Python 标准输出使用 UTF-8
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")
import glob
import random

from ultralytics import YOLO

# ================= 配置区域 =================
# 1. 你的项目名称 (必须与 train_pothole.py 中的一致)
PROJECT_NAME = "Pothole_CBAM_Project"
EXP_NAME = "exp_cbam"  # 训练时定义的 name

# 2. 训练好的权重路径 (通常在 runs/detect/项目名/实验名/weights/best.pt)
# Windows 路径注意: 如果路径不对，请去文件夹里找一下 best.pt 在哪里
BEST_WEIGHTS = f"{PROJECT_NAME}/{EXP_NAME}/weights/best.pt"

# 3. 数据集配置 (用于评估指标)
DATASET_YAML = "pothole_config.yaml"

# 4. 测试图片目录 (用于画框演示)
# 这里默认使用验证集的一张图片，你也可以改为具体的图片路径
TEST_IMAGES_DIR = "D:/Desktop/教材/深度学习/大作业/yolov8/datasets/New_pothole_detection.v2i.yolov8/test/images"


# ===========================================


def validate_metrics():
    """1. 计算模型在验证集上的准确率指标.
    """
    print(f"\n📊 正在加载最佳权重: {BEST_WEIGHTS}")

    if not os.path.exists(BEST_WEIGHTS):
        print(f"❌ 错误：找不到权重文件 {BEST_WEIGHTS}")
        print("请检查 runs/detect/ 目录下生成的文件夹名称是否正确")
        return

    # 加载训练好的模型
    model = YOLO(BEST_WEIGHTS)

    print("开始在验证集上评估 mAP...")
    metrics = model.val(data=DATASET_YAML, split="val")

    print("\n" + "=" * 30)
    print("✅ 评估完成！关键指标如下 (请填入实验报告):")
    print(f"Precision (精确率): {metrics.results_dict['metrics/precision(B)']:.4f}")
    print(f"Recall    (召回率): {metrics.results_dict['metrics/recall(B)']:.4f}")
    print(f"mAP@50    (平均精度): {metrics.results_dict['metrics/mAP50(B)']:.4f}")
    print("=" * 30 + "\n")


def predict_visualization():
    """2. 随机选取图片进行推理，并保存结果.
    """
    print("🖼️ 开始进行可视化推理测试...")

    model = YOLO(BEST_WEIGHTS)

    # 获取测试集所有图片
    # 如果 datasets 路径不在当前目录，这里可能需要手动修改为你电脑上的绝对路径
    # 这里尝试去读取 dataset.yaml 里的 path，如果不方便，直接写死路径
    test_imgs = glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg")) + glob.glob(os.path.join(TEST_IMAGES_DIR, "*.png"))

    if len(test_imgs) == 0:
        print(f"⚠️ 警告：在 {TEST_IMAGES_DIR} 没找到图片，无法演示推理。")
        print("请手动修改代码中的 TEST_IMAGES_DIR 变量。")
        return

    # 随机选 3 张
    selected_imgs = random.sample(test_imgs, min(3, len(test_imgs)))

    # 推理并保存
    # save=True 会把结果保存在 runs/detect/predict/ 文件夹下
    results = model.predict(selected_imgs, save=True, conf=0.25, line_width=2)

    print("✅ 推理完成！")
    print(f"📂 请打开此文件夹查看效果图: {results[0].save_dir}")


if __name__ == "__main__":
    # 确保当前是在项目根目录运行
    print(f"当前工作目录: {os.getcwd()}")

    # 1. 跑分
    validate_metrics()

    # 2. 看图
    predict_visualization()
