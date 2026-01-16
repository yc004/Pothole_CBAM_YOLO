from ultralytics import YOLO
import os

# 定义模型路径列表
model_paths = [
    r"Pothole_Baseline_Project/exp_baseline/weights/best.pt",  # 基线版本
    r"Pothole_CBAM_Project/exp_cbam/weights/best.pt"           # 改进版本
]

for path in model_paths:
    if os.path.exists(path):
        print(f"\n正在处理模型: {path}")
        try:
            model = YOLO(path)
            model.export(
                format="onnx",
                imgsz=640,
                opset=12,
                simplify=True
            )
            print(f"成功导出: {path}")
        except Exception as e:
            print(f"导出失败 {path}: {e}")
    else:
        print(f"文件不存在: {path}")
