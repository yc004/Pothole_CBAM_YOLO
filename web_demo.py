import asyncio
import os
import sys

# 修复 Windows 下中文乱码问题
if sys.platform.startswith("win"):
    os.system("chcp 65001 >nul")
    # 修复 Windows 下 asyncio 报错 (WinError 10054)
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

import cv2
import gradio as gr

from ultralytics import YOLO

# ================= 配置 =================
# 权重路径 (请确保这个路径下有 best.pt)
MODEL_PATH = "Pothole_CBAM_Project/exp_cbam/weights/best.pt"

print(f"⏳ 正在加载模型: {MODEL_PATH} ...")
try:
    model = YOLO(MODEL_PATH)
    print("✅ 模型加载成功！")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    print("请检查路径是否正确，或者是否已经运行了 train_pothole.py 进行训练。")
    # 为了演示，如果加载失败，这里可能会报错退出
    sys.exit(1)


def detect_pothole(image):
    """
    执行路面坑洼检测
    :param image: 输入图片 (PIL.Image 或 numpy array)
    :return: 标注后的图片, 检测信息文本.
    """
    if image is None:
        return None, "请先上传图片"

    # 执行推理
    # conf: 置信度阈值
    results = model.predict(image, conf=0.25)

    # 获取第一张图的结果 (因为我们只输入了一张)
    result = results[0]

    # 绘制结果 (返回的是 BGR 格式的 numpy 数组)
    plot_img_bgr = result.plot()

    # 将 BGR 转为 RGB (Gradio 需要 RGB)
    plot_img_rgb = plot_img_bgr[..., ::-1]

    # 统计检测到的数量
    count = len(result.boxes)
    info = f"✅ 检测完成！\n🔍 发现 {count} 个坑洼目标。"

    return plot_img_rgb, info


import shutil
import subprocess

# ... (imports)


def detect_video(video_path):
    """处理视频文件."""
    if video_path is None:
        return None, "请上传视频"

    cap = cv2.VideoCapture(video_path)

    # 临时文件路径
    temp_raw = "temp_raw.mp4"
    output_path = "output_video.mp4"

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # OpenCV 写入临时文件
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_raw, fourcc, fps, (width, height))

    frame_count = 0
    total_detections = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.25, verbose=False)
        annotated_frame = results[0].plot()

        total_detections += len(results[0].boxes)
        out.write(annotated_frame)
        frame_count += 1

    cap.release()
    out.release()

    # 使用 ffmpeg 转码为 H.264 (浏览器兼容)
    # 检查 ffmpeg 是否可用
    if shutil.which("ffmpeg"):
        try:
            print("🔄 正在使用 FFmpeg 转码为 H.264 以支持浏览器播放...")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    temp_raw,
                    "-c:v",
                    "h264_mf",  # 使用 Windows 原生 MediaFoundation 编码器
                    "-b:v",
                    "5M",  # 设置 5Mbps 高码率
                    "-rate_control",
                    "cbr",  # 强制恒定码率控制，确保清晰度
                    "-f",
                    "mp4",
                    output_path,
                ],
                check=True,
                capture_output=True,
            )
            return output_path, f"✅ 视频处理完成！\n共处理 {frame_count} 帧，累计检测到 {total_detections} 次坑洼。"
        except Exception as e:
            print(f"⚠️ FFmpeg 转码失败: {e}。将返回原始视频。")
            return temp_raw, f"✅ 视频处理完成！(但转码失败，浏览器可能无法预览)\n共处理 {frame_count} 帧。"
    else:
        print("⚠️ 未找到 FFmpeg，将返回原始视频 (浏览器可能无法预览)。")
        return (
            temp_raw,
            f"✅ 视频处理完成！\n共处理 {frame_count} 帧。\n(注意：未安装 FFmpeg，视频可能无法直接预览，请下载后观看)",
        )


# ================= 构建界面 =================
with gr.Blocks(title="基于 YOLOv8-CBAM 的路面坑洼检测") as demo:
    gr.Markdown("# 🛣️ 路面坑洼检测系统 (YOLOv8 + CBAM)")

    with gr.Tabs():
        with gr.TabItem("📷 图片检测"):
            gr.Markdown("上传路面照片，系统将自动检测并标记出坑洼区域。")
            with gr.Row():
                with gr.Column():
                    input_img = gr.Image(type="pil", label="上传图片")
                    run_btn = gr.Button("开始检测", variant="primary")

                with gr.Column():
                    output_img = gr.Image(type="numpy", label="检测结果")
                    output_text = gr.Textbox(label="检测信息")

            run_btn.click(fn=detect_pothole, inputs=input_img, outputs=[output_img, output_text])

            gr.Examples(
                examples=[
                    "datasets/New_pothole_detection.v2i.yolov8/test/images/1_jpg.rf.a9cc87ae30331b83ba2e75fddcf1ebd5.jpg"
                ],
                inputs=input_img,
            )

        with gr.TabItem("🎥 视频检测"):
            gr.Markdown("上传路面视频，系统将生成检测后的视频文件。")
            with gr.Row():
                with gr.Column():
                    input_video = gr.Video(label="上传视频")
                    video_btn = gr.Button("开始处理视频", variant="primary")

                with gr.Column():
                    output_video = gr.Video(label="处理结果")
                    video_info = gr.Textbox(label="处理信息")

            video_btn.click(fn=detect_video, inputs=input_video, outputs=[output_video, video_info])

if __name__ == "__main__":
    print("🚀 启动 Web 服务...")
    # launch(inbrowser=True) 会自动打开浏览器
    demo.launch(inbrowser=True)
