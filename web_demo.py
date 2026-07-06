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

import shutil
import subprocess

import cv2
import gradio as gr
import numpy as np
import torch

from ultralytics import YOLO


# ================= 热力图工具类 =================
class ActivationHook:
    def __init__(self):
        self.activation = None

    def hook_fn(self, module, input, output):
        self.activation = output.detach()


def generate_heatmap(activation, img_size):
    # activation: (1, C, H, W)
    if activation is None:
        print("⚠️ Heatmap Error: Activation is None")
        return None

    if activation.numel() == 0:
        print("⚠️ Heatmap Error: Activation is empty")
        return None

    # Debug activation stats
    # print(f"DEBUG: Activation shape: {activation.shape}, Range: [{activation.min():.4f}, {activation.max():.4f}]")

    # 1. Pre-process: Clamp negative values to 0 (SiLU/ReLU outputs)
    # This prevents negative activations from canceling out positive ones during averaging
    activation = activation.clamp(min=0)

    # 2. Aggregation: Use Mean of activations (or could use Max)
    heatmap = torch.mean(activation, dim=1).squeeze()

    # 3. Move to CPU
    heatmap = heatmap.cpu().numpy()

    # 4. Normalization
    max_val = np.max(heatmap)
    min_val = np.min(heatmap)

    # Check if we have a valid range
    if max_val <= 0:
        # This implies no positive activation at all in the entire layer
        # print(f"⚠️ Heatmap Warning: No positive activation (Max={max_val}).")
        return None

    if max_val - min_val == 0:
        heatmap = np.zeros_like(heatmap)
    else:
        heatmap = (heatmap - min_val) / (max_val - min_val)

    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.resize(heatmap, img_size)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap_color


def get_heatmap_for_model(model, pil_img):
    # Ensure consistent input format (Numpy BGR)
    img_rgb = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_h, img_w = img_rgb.shape[:2]

    hook = ActivationHook()
    target_layer = None
    layer_name = "Unknown"

    # --- Robust Layer Finding (Index Based) ---
    try:
        if hasattr(model.model, "model"):
            model_layers = model.model.model

            # Check for CBAM model heuristic
            is_cbam_model = False
            for m in model_layers:
                if "CBAM" in m.__class__.__name__:
                    is_cbam_model = True
                    break

            if is_cbam_model:
                # CBAM Model: Layer 9 is CBAM. We want to hook CBAM layer.
                if len(model_layers) > 9:
                    target_layer = model_layers[9]
                    layer_name = f"Layer 9 ({target_layer.__class__.__name__})"
                elif len(model_layers) > 10:
                    # Fallback to SPPF if Layer 9 is somehow not right (unlikely based on yaml)
                    target_layer = model_layers[10]
                    layer_name = f"Layer 10 ({target_layer.__class__.__name__})"
            else:
                # Baseline Model: Layer 9 is SPPF.
                if len(model_layers) > 9:
                    target_layer = model_layers[9]
                    layer_name = f"Layer 9 ({target_layer.__class__.__name__})"

    except Exception as e:
        print(f"Error accessing model layers: {e}")

    # Fallback: Search by name
    if target_layer is None:
        for m in model.model.modules():
            if m.__class__.__name__ == "SPPF":
                target_layer = m
                layer_name = "SPPF (Module Search)"
                break

    # Refine target for CBAM: If it has spatial_attention, hook that for a cleaner heatmap
    if target_layer and hasattr(target_layer, "spatial_attention"):
        target_layer = target_layer.spatial_attention
        layer_name += " (SpatialAttention)"

    heatmap_overlay = img_rgb  # Default to original image

    if target_layer:
        # print(f"DEBUG: Hooking {layer_name}")
        handle = target_layer.register_forward_hook(hook.hook_fn)

        try:
            # Use BGR numpy array for inference
            model.predict(img_bgr, verbose=False, conf=0.25)
        except Exception as e:
            print(f"Inference failed: {e}")

        handle.remove()

        if hook.activation is not None:
            heatmap_color = generate_heatmap(hook.activation, (img_w, img_h))
            if heatmap_color is not None:
                # heatmap_color is BGR. Convert to RGB for display
                heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                heatmap_overlay = cv2.addWeighted(img_rgb, 0.6, heatmap_rgb, 0.4, 0)
            else:
                print(f"DEBUG: Heatmap generation returned None for {layer_name}")
        else:
            print(f"DEBUG: No activation captured for {layer_name}")
    else:
        print("DEBUG: No target layer found for heatmap")

    return heatmap_overlay


# ================= 配置 =================
# 权重路径
MODEL_PATH_BASELINE = "Pothole_Baseline_Project/exp_baseline/weights/best.pt"
MODEL_PATH_CBAM = "Pothole_CBAM_Project/exp_cbam/weights/best.pt"

print("⏳ 正在加载模型...")
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
    :return: 基线结果图, 改进结果图, 基线热力图, 改进热力图, 检测信息文本.
    """
    if image is None:
        return None, None, None, None, "请先上传图片"

    # 1. 基线模型推理
    results_baseline = model_baseline.predict(image, conf=0.25)
    res_base = results_baseline[0]
    plot_base_bgr = res_base.plot()
    plot_base_rgb = plot_base_bgr[..., ::-1]  # BGR to RGB
    count_base = len(res_base.boxes)

    # 生成基线热力图
    heatmap_base = get_heatmap_for_model(model_baseline, image)

    # 2. 改进模型推理
    results_cbam = model_cbam.predict(image, conf=0.25)
    res_cbam = results_cbam[0]
    plot_cbam_bgr = res_cbam.plot()
    plot_cbam_rgb = plot_cbam_bgr[..., ::-1]  # BGR to RGB
    count_cbam = len(res_cbam.boxes)

    # 生成改进热力图
    heatmap_cbam = get_heatmap_for_model(model_cbam, image)

    info = f"✅ 检测完成！\n🔹 基线模型检测到: {count_base} 个目标\n🔸 改进模型检测到: {count_cbam} 个目标"

    return plot_base_rgb, plot_cbam_rgb, heatmap_base, heatmap_cbam, info


def detect_video(video_path):
    """处理视频文件 (对比模式 - 合并显示 - 包含热力图)."""
    if video_path is None:
        return None, "请上传视频"

    cap = cv2.VideoCapture(video_path)

    # 临时文件路径
    temp_raw_combined = "temp_raw_combined.mp4"
    output_path_combined = "output_video_combined.mp4"

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 输出视频布局: 2x2 网格
    # Top Left: Baseline Detection | Top Right: CBAM Detection
    # Bot Left: Baseline Heatmap   | Bot Right: CBAM Heatmap
    new_width = width * 2
    new_height = height * 2

    # OpenCV 写入临时文件
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_raw_combined, fourcc, fps, (new_width, new_height))

    frame_count = 0
    total_detections_base = 0
    total_detections_cbam = 0

    # 准备 Hook
    hook_base = ActivationHook()
    hook_cbam = ActivationHook()

    layer_base = None
    layer_cbam = None

    # Try to hook Layer 9 (SPPF for Base, CBAM for CBAM-Model)
    target_index = 9

    try:
        if hasattr(model_baseline.model, "model"):
            # Baseline: Layer 9 is SPPF
            if len(model_baseline.model.model) > target_index:
                layer_base = model_baseline.model.model[target_index]

        if hasattr(model_cbam.model, "model"):
            # CBAM Model: Layer 9 is CBAM
            if len(model_cbam.model.model) > target_index:
                layer_cbam = model_cbam.model.model[target_index]
    except:
        pass

    # Fallback search if index failed
    if layer_base is None:
        for m in model_baseline.model.modules():
            if m.__class__.__name__ == "SPPF":  # Fallback to SPPF
                layer_base = m
                break

    if layer_cbam is None:
        for m in model_cbam.model.modules():
            if m.__class__.__name__ == "SPPF":
                layer_cbam = m
                break

    # 注册 Hook
    handle_base = None
    if layer_base:
        if hasattr(layer_base, "spatial_attention"):
            layer_base = layer_base.spatial_attention
        handle_base = layer_base.register_forward_hook(hook_base.hook_fn)

    handle_cbam = None
    if layer_cbam:
        if hasattr(layer_cbam, "spatial_attention"):
            layer_cbam = layer_cbam.spatial_attention
        handle_cbam = layer_cbam.register_forward_hook(hook_cbam.hook_fn)

    print("🔄 正在逐帧处理视频 (合并模式 + 热力图)...")
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Baseline Inference
            # frame is BGR
            results_base = model_baseline.predict(frame, conf=0.25, verbose=False)
            annotated_frame_base = results_base[0].plot()
            total_detections_base += len(results_base[0].boxes)

            # Generate Baseline Heatmap
            heatmap_vis_base = frame.copy()  # fallback
            if hook_base.activation is not None:
                heatmap = generate_heatmap(hook_base.activation, (width, height))
                if heatmap is not None:
                    # heatmap is BGR, frame is BGR
                    heatmap_vis_base = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)

            # 添加标签
            cv2.putText(
                annotated_frame_base,
                "Baseline Det",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                heatmap_vis_base,
                "Baseline Feature",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

            # 2. CBAM Inference
            results_cbam = model_cbam.predict(frame, conf=0.25, verbose=False)
            annotated_frame_cbam = results_cbam[0].plot()
            total_detections_cbam += len(results_cbam[0].boxes)

            # Generate CBAM Heatmap
            heatmap_vis_cbam = frame.copy()  # fallback
            if hook_cbam.activation is not None:
                heatmap = generate_heatmap(hook_cbam.activation, (width, height))
                if heatmap is not None:
                    heatmap_vis_cbam = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)

            # 添加标签
            cv2.putText(
                annotated_frame_cbam, "CBAM Det", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA
            )
            cv2.putText(
                heatmap_vis_cbam, "CBAM Attention", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA
            )

            # 3. 合并画面 (2x2 Grid)
            # Top Row
            top_row = np.hstack((annotated_frame_base, annotated_frame_cbam))
            # Bottom Row
            bot_row = np.hstack((heatmap_vis_base, heatmap_vis_cbam))
            # Full Grid
            combined_frame = np.vstack((top_row, bot_row))

            out.write(combined_frame)

            frame_count += 1
            if frame_count % 10 == 0:
                print(f"   已处理 {frame_count} 帧...", end="\r")
    finally:
        # 清理 Hook
        if handle_base:
            handle_base.remove()
        if handle_cbam:
            handle_cbam.remove()

    cap.release()
    out.release()
    print(f"\n✅ 视频推理完成，共 {frame_count} 帧。")

    # 转码函数
    def transcode(input_path, output_path):
        if shutil.which("ffmpeg"):
            try:
                print("🔄 正在转码...")
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        input_path,
                        "-c:v",
                        "h264_mf",  # Windows 硬件加速
                        "-b:v",
                        "5M",
                        "-rate_control",
                        "cbr",
                        "-f",
                        "mp4",
                        output_path,
                    ],
                    check=True,
                    capture_output=True,
                )
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

    info = (
        f"✅ 视频处理完成！\n"
        f"共处理 {frame_count} 帧。\n"
        f"🔹 基线模型累计检测: {total_detections_base} 次\n"
        f"🔸 改进模型累计检测: {total_detections_cbam} 次\n"
        f"ℹ️ 状态: {msg_status}"
    )

    return final_video, info


import time

# 全局变量用于计算 FPS
prev_time = 0


def detect_webcam(image):
    """
    摄像头实时检测 (双模型对比)
    :param image: 摄像头采集的当前帧 (RGB numpy array).
    """
    global prev_time

    if image is None:
        return None, "等待摄像头输入..."

    # 记录开始时间
    start_time = time.time()

    # 1. 基线模型推理
    results_base = model_baseline.predict(image, conf=0.25, verbose=False)
    res_base = results_base[0]
    plot_base_bgr = res_base.plot()
    count_base = len(res_base.boxes)

    # 2. 改进模型推理
    results_cbam = model_cbam.predict(image, conf=0.25, verbose=False)
    res_cbam = results_cbam[0]
    plot_cbam_bgr = res_cbam.plot()
    count_cbam = len(res_cbam.boxes)

    # 3. 合并画面 (左右并排)
    combined_bgr = np.hstack((plot_base_bgr, plot_cbam_bgr))
    combined_rgb = combined_bgr[..., ::-1]

    # 4. 计算 FPS
    curr_time = time.time()
    # 计算瞬时 FPS (基于本次处理时间)
    process_time = curr_time - start_time
    fps = 1 / process_time if process_time > 0 else 0

    # 也可以使用平滑 FPS (基于帧间隔)
    # fps_smooth = 1 / (curr_time - prev_time) if prev_time > 0 else 0
    prev_time = curr_time

    # 5. 生成统计信息
    info = (
        f"⏱️ 实时 FPS: {fps:.2f}\n"
        f"⚡ 处理耗时: {process_time * 1000:.1f} ms\n"
        f"----------------------\n"
        f"🔹 Baseline 检测目标: {count_base}\n"
        f"🔸 CBAM (改进) 检测目标: {count_cbam}"
    )

    return combined_rgb, info


# ================= 构建界面 =================
with gr.Blocks(title="路面坑洼检测模型对比系统") as demo:
    gr.Markdown("# 🛣️ 路面坑洼检测系统 - 模型效果对比")
    gr.Markdown("本系统同时展示 **Baseline (基线模型)** 与 **CBAM (改进模型)** 的检测结果，以便直观对比性能差异。")

    with gr.Tabs():
        with gr.TabItem("📷 图片对比检测"):
            gr.Markdown("上传路面照片，系统将分别使用基线模型和改进模型进行检测，并展示注意力热力图。")
            with gr.Row():
                with gr.Column(scale=1):
                    input_img = gr.Image(type="pil", label="上传原始图片")
                    run_btn = gr.Button("开始对比检测", variant="primary")

                with gr.Column(scale=2):
                    gr.Markdown("### 🔍 检测结果")
                    with gr.Row():
                        output_base = gr.Image(type="numpy", label="基线模型 (Baseline) 结果")
                        output_cbam = gr.Image(type="numpy", label="改进模型 (CBAM) 结果")

                    gr.Markdown("### 🔥 热力图对比 (Heatmap Comparison)")
                    gr.Markdown(
                        "注：基线模型展示的是 **SPPF层特征激活图 (Feature Activation)**，反映高响应区域；改进模型展示的是 **CBAM层空间注意力图 (Spatial Attention)**，反映模型主动关注的区域。"
                    )
                    with gr.Row():
                        heatmap_base = gr.Image(type="numpy", label="基线模型特征响应 (SPPF Activation)")
                        heatmap_cbam = gr.Image(type="numpy", label="改进模型注意力 (CBAM Attention)")

                    output_text = gr.Textbox(label="检测统计信息")

            run_btn.click(
                fn=detect_pothole,
                inputs=input_img,
                outputs=[output_base, output_cbam, heatmap_base, heatmap_cbam, output_text],
            )

            gr.Examples(
                examples=[
                    "datasets/New_pothole_detection.v2i.yolov8/test/images/1_jpg.rf.a9cc87ae30331b83ba2e75fddcf1ebd5.jpg"
                ],
                inputs=input_img,
            )

        with gr.TabItem("🎥 视频对比检测"):
            gr.Markdown(
                "上传路面视频，系统将生成 **Baseline (左)** 和 **CBAM (右)** 的并排对比视频，下方附带**热力图**，方便逐帧比对效果。"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    input_video = gr.Video(label="上传视频")
                    video_btn = gr.Button("开始对比处理", variant="primary")

                with gr.Column(scale=2):
                    # 给 Video 组件添加 elem_id，方便 JS 定位
                    output_video_combined = gr.Video(label="对比结果 (上:检测框 | 下:热力图)", elem_id="video_output")

                    # 添加播放速度控制滑块
                    speed_slider = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        step=0.1,
                        value=1.0,
                        label="播放速度 (0.1x - 2.0x)",
                        elem_id="speed_slider",
                    )

                    video_info = gr.Textbox(label="处理信息")

            # 按钮点击事件（后端处理）
            video_btn.click(fn=detect_video, inputs=input_video, outputs=[output_video_combined, video_info])

            # 滑块改变事件（前端 JS 处理）
            # 使用 JavaScript 直接控制 video 元素的 playbackRate
            speed_slider.change(
                fn=None,
                inputs=speed_slider,
                outputs=None,
                js="(speed) => { const video = document.querySelector('#video_output video'); if (video) { video.playbackRate = speed; } }",
            )

        with gr.TabItem("🔴 实时预测"):
            gr.Markdown("使用摄像头进行实时路面坑洼检测。**左侧：Baseline (基线模型) | 右侧：CBAM (改进模型)**")
            gr.Markdown("⚠️ 注意：同时运行两个模型可能会导致帧率较低，具体取决于硬件性能。")
            with gr.Row():
                with gr.Column():
                    # Gradio 4.x 兼容性修改:
                    # 1. source="webcam" -> sources=["webcam"]
                    # 2. 移除 streaming=True (通过 .stream() 事件处理)
                    input_webcam = gr.Image(sources=["webcam"], label="摄像头输入", type="numpy")
                with gr.Column():
                    output_webcam = gr.Image(label="实时检测结果 (对比)")
                    webcam_info = gr.Textbox(label="实时数据监控", lines=5)

            # 使用 stream 事件实现实时流处理 (如果是旧版 Gradio，可能需要用 change)
            # 为了兼容性，尝试检测 stream 属性，或者直接使用 change (在 streaming=True 时通常也有效)
            # 这里使用 stream 以获得更好的性能 (如果支持)
            try:
                input_webcam.stream(
                    fn=detect_webcam, inputs=input_webcam, outputs=[output_webcam, webcam_info], show_progress="hidden"
                )
            except AttributeError:
                # Fallback for older Gradio versions
                input_webcam.change(
                    fn=detect_webcam, inputs=input_webcam, outputs=[output_webcam, webcam_info], show_progress="hidden"
                )

if __name__ == "__main__":
    print("🚀 启动 Web 服务...")
    # launch(inbrowser=True) 会自动打开浏览器
    demo.launch(inbrowser=True)
