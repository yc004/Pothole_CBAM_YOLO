import os
import sys
import time

import cv2

from ultralytics import YOLO

# 修复 Windows 下中文乱码问题
if sys.platform.startswith("win"):
    os.system("chcp 65001 >nul")


def process_video(source=0, weights="Pothole_CBAM_Project/exp_cbam/weights/best.pt", conf=0.25):
    """
    实时视频预测
    :param source: 视频源，0 表示摄像头，或者传入视频文件路径
    :param weights: 模型权重路径
    :param conf: 置信度阈值.
    """
    print(f"⏳ 正在加载模型: {weights} ...")
    try:
        model = YOLO(weights)
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 打开视频源
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ 无法打开视频源: {source}")
        return

    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"🎥 视频源信息: {width}x{height}, FPS: {fps}")
    print("👉 按 'q' 键退出预览")

    # 简单的 FPS 计算
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频结束或无法读取帧")
            break

        # 执行推理
        # stream=True 让推理更流畅，不积压内存
        results = model.predict(frame, conf=conf, verbose=False)
        result = results[0]

        # 绘制结果
        annotated_frame = result.plot()

        # 计算并显示 FPS
        curr_time = time.time()
        fps_curr = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time

        cv2.putText(annotated_frame, f"FPS: {fps_curr:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示画面
        cv2.imshow("Pothole Detection (Press 'q' to exit)", annotated_frame)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 视频流/摄像头实时检测")
    parser.add_argument("--source", type=str, default="0", help="视频源: '0' 代表摄像头，或输入视频文件路径")
    parser.add_argument(
        "--weights", type=str, default="Pothole_CBAM_Project/exp_cbam/weights/best.pt", help="模型权重路径"
    )
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")

    args = parser.parse_args()

    # 处理 source 参数，如果是数字字符串则转为 int
    source = args.source
    if source.isdigit():
        source = int(source)

    process_video(source, args.weights, args.conf)
