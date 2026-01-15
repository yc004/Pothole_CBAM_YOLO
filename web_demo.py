import sys
import os
import asyncio

# 修复 Windows 下中文乱码问题
if sys.platform.startswith('win'):
    os.system('chcp 65001 >nul')
    # 修复 Windows 下 asyncio 报错 (WinError 10054)
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

import gradio as gr
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import subprocess
import shutil

# ================= 配置 =================
# 权重路径
MODEL_PATH_BASELINE = "Pothole_Baseline_Project/exp_baseline/weights/best.pt"
MODEL_PATH_CBAM = "Pothole_CBAM_Project/exp_cbam/weights/best.pt"

print(f"⏳ 正在加载模型...")
try:
    print(f"   - 加载基线模型: {MODEL_PATH_BASELINE}")
    model_baseline = YOLO(MODEL_PATH_BASELINE)
    print(f"   - 加载改进模型 (CBAM): {MODEL_PATH_CBAM}")
    model_cbam = YOLO(MODEL_PATH_CBAM)
    print("✅ 模型加载成功！")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    print("请检查路径是否正确，或者是否已经运行了 train_pothole.py 进行训练。")
    sys.exit(1)

def detect_pothole(image):
    """
    执行路面坑洼检测 (对比模式)
    :param image: 输入图片 (PIL.Image)
    :return: 基线结果图, 改进结果图, 检测信息文本
    """
    if image is None:
        return None, None, "请先上传图片"

    # 1. 基线模型推理
    results_baseline = model_baseline.predict(image, conf=0.25)
    res_base = results_baseline[0]
    plot_base_bgr = res_base.plot()
    plot_base_rgb = plot_base_bgr[..., ::-1] # BGR to RGB
    count_base = len(res_base.boxes)

    # 2. 改进模型推理
    results_cbam = model_cbam.predict(image, conf=0.25)
    res_cbam = results_cbam[0]
    plot_cbam_bgr = res_cbam.plot()
    plot_cbam_rgb = plot_cbam_bgr[..., ::-1] # BGR to RGB
    count_cbam = len(res_cbam.boxes)
    
    info = (f"✅ 检测完成！\n"
            f"🔹 基线模型检测到: {count_base} 个目标\n"
            f"🔸 改进模型检测到: {count_cbam} 个目标")
    
    return plot_base_rgb, plot_cbam_rgb, info

def detect_video(video_path):
    """
    处理视频文件 (对比模式 - 合并显示)
    """
    if video_path is None:
        return None, "请上传视频"
        
    cap = cv2.VideoCapture(video_path)
    
    # 临时文件路径
    temp_raw_combined = "temp_raw_combined.mp4"
    output_path_combined = "output_video_combined.mp4"
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 输出视频宽度为两倍 (左右并排)
    new_width = width * 2
    
    # OpenCV 写入临时文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_raw_combined, fourcc, fps, (new_width, height))
    
    frame_count = 0
    total_detections_base = 0
    total_detections_cbam = 0
    
    print("🔄 正在逐帧处理视频 (合并模式)...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 1. Baseline Inference
        results_base = model_baseline.predict(frame, conf=0.25, verbose=False)
        annotated_frame_base = results_base[0].plot()
        total_detections_base += len(results_base[0].boxes)
        
        # 添加标签
        cv2.putText(annotated_frame_base, "Baseline", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.2, (0, 0, 255), 3, cv2.LINE_AA)

        # 2. CBAM Inference
        results_cbam = model_cbam.predict(frame, conf=0.25, verbose=False)
        annotated_frame_cbam = results_cbam[0].plot()
        total_detections_cbam += len(results_cbam[0].boxes)
        
        # 添加标签
        cv2.putText(annotated_frame_cbam, "CBAM (Improved)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.2, (0, 255, 0), 3, cv2.LINE_AA)
        
        # 3. 合并画面
        combined_frame = np.hstack((annotated_frame_base, annotated_frame_cbam))
        out.write(combined_frame)
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"   已处理 {frame_count} 帧...", end="\r")
        
    cap.release()
    out.release()
    print(f"\n✅ 视频推理完成，共 {frame_count} 帧。")
    
    # 转码函数
    def transcode(input_path, output_path):
        if shutil.which("ffmpeg"):
            try:
                print(f"🔄 正在转码...")
                subprocess.run([
                    "ffmpeg", "-y", 
                    "-i", input_path, 
                    "-c:v", "h264_mf", # Windows 硬件加速
                    "-b:v", "5M",      
                    "-rate_control", "cbr", 
                    "-f", "mp4", 
                    output_path
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return output_path, True
            except Exception as e:
                print(f"⚠️ 转码失败: {e}")
                return input_path, False
        else:
            return input_path, False

    # 执行转码
    final_video, is_transcoded = transcode(temp_raw_combined, output_path_combined)
    
    msg_status = "转码成功" if is_transcoded else "未转码(可能无法预览)"
    if not shutil.which("ffmpeg"):
         msg_status = "未安装FFmpeg，无法预览，请下载观看"

    info = (f"✅ 视频处理完成！\n"
            f"共处理 {frame_count} 帧。\n"
            f"🔹 基线模型累计检测: {total_detections_base} 次\n"
            f"🔸 改进模型累计检测: {total_detections_cbam} 次\n"
            f"ℹ️ 状态: {msg_status}")

    return final_video, info

# ================= 构建界面 =================
with gr.Blocks(title="路面坑洼检测模型对比系统") as demo:
    gr.Markdown("# 🛣️ 路面坑洼检测系统 - 模型效果对比")
    gr.Markdown("本系统同时展示 **Baseline (基线模型)** 与 **CBAM (改进模型)** 的检测结果，以便直观对比性能差异。")
    
    with gr.Tabs():
        with gr.TabItem("📷 图片对比检测"):
            gr.Markdown("上传路面照片，系统将分别使用基线模型和改进模型进行检测。")
            with gr.Row():
                with gr.Column(scale=1):
                    input_img = gr.Image(type="pil", label="上传原始图片")
                    run_btn = gr.Button("开始对比检测", variant="primary")
                
                with gr.Column(scale=2):
                    with gr.Row():
                        output_base = gr.Image(type="numpy", label="基线模型 (Baseline) 结果")
                        output_cbam = gr.Image(type="numpy", label="改进模型 (CBAM) 结果")
                    output_text = gr.Textbox(label="检测统计信息")
                    
            run_btn.click(fn=detect_pothole, inputs=input_img, outputs=[output_base, output_cbam, output_text])
            
            gr.Examples(
                examples=["datasets/New_pothole_detection.v2i.yolov8/test/images/1_jpg.rf.a9cc87ae30331b83ba2e75fddcf1ebd5.jpg"],
                inputs=input_img
            )
            
        with gr.TabItem("🎥 视频对比检测"):
            gr.Markdown("上传路面视频，系统将生成 **Baseline (左)** 和 **CBAM (右)** 的并排对比视频，方便逐帧比对效果。")
            with gr.Row():
                with gr.Column(scale=1):
                    input_video = gr.Video(label="上传视频")
                    video_btn = gr.Button("开始对比处理", variant="primary")
                
                with gr.Column(scale=2):
                    output_video_combined = gr.Video(label="对比结果 (左: Baseline | 右: CBAM)")
                    video_info = gr.Textbox(label="处理信息")
            
            video_btn.click(fn=detect_video, inputs=input_video, outputs=[output_video_combined, video_info])

if __name__ == "__main__":
    print("🚀 启动 Web 服务...")
    # launch(inbrowser=True) 会自动打开浏览器
    demo.launch(inbrowser=True)
