from ultralytics import YOLO

# 加载模型 (这里以 yolov8n.pt 为例，如果是你自己的模型，请替换为模型路径，例如 'runs/detect/train/weights/best.pt')
model = YOLO('Pothole_CBAM_Project\\exp_cbam\\weights\\best.pt')

# 打印模型信息
print("简单信息:")
model.info()

print("\n详细信息:")
model.info(detailed=True)
