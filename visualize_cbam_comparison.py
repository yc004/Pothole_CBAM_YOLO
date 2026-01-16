import torch
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

# Define the target image path (picked one from the list)
IMAGE_PATH = r"d:\Desktop\教材\深度学习\大作业\yolov8\datasets\New_pothole_detection.v2i.yolov8\test\images\po_389_jpg.rf.91adc5ab8d236c3ce354aaf38a9efeb1.jpg"
BASELINE_WEIGHTS = r"Pothole_Baseline_Project/exp_baseline/weights/best.pt"
CBAM_WEIGHTS = r"Pothole_CBAM_Project/exp_cbam/weights/best.pt"
OUTPUT_PATH = "cbam_attention_comparison.jpg"

class ActivationHook:
    def __init__(self):
        self.activation = None

    def hook_fn(self, module, input, output):
        self.activation = output.detach()

def generate_heatmap(activation, img_size):
    # activation: (1, C, H, W)
    # Simple Mean Activation Mapping
    heatmap = torch.mean(activation, dim=1).squeeze().cpu().numpy()
    
    # Normalize to [0, 255]
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Resize to original image size
    heatmap = cv2.resize(heatmap, img_size)
    
    # Apply ColorMap
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap_color

def overlay_heatmap(img, heatmap, alpha=0.5):
    return cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image not found at {IMAGE_PATH}")
        return

    # Load Image
    img = cv2.imread(IMAGE_PATH)
    img_h, img_w = img.shape[:2]
    print(f"Processing image: {IMAGE_PATH} ({img_w}x{img_h})")

    # --- Process Baseline ---
    print("Loading Baseline Model...")
    model_baseline = YOLO(BASELINE_WEIGHTS)
    hook_baseline = ActivationHook()
    
    # Baseline SPPF is typically Layer 9
    # We target the SPPF module directly to be safe
    target_layer_baseline = None
    for m in model_baseline.model.modules():
        if m.__class__.__name__ == 'SPPF':
            target_layer_baseline = m
            break
            
    if target_layer_baseline:
        handle_baseline = target_layer_baseline.register_forward_hook(hook_baseline.hook_fn)
        model_baseline(IMAGE_PATH, verbose=False)
        handle_baseline.remove()
        
        heatmap_baseline = generate_heatmap(hook_baseline.activation, (img_w, img_h))
        vis_baseline = overlay_heatmap(img, heatmap_baseline)
    else:
        print("Warning: SPPF layer not found in Baseline model.")
        vis_baseline = img.copy()

    # --- Process CBAM ---
    print("Loading CBAM Model...")
    model_cbam = YOLO(CBAM_WEIGHTS)
    hook_cbam = ActivationHook()
    
    # CBAM Model has CBAM at Layer 9 and SPPF at Layer 10
    # We want to see the effect AFTER CBAM and SPPF (input to head)
    target_layer_cbam = None
    for m in model_cbam.model.modules():
        if m.__class__.__name__ == 'SPPF':
            target_layer_cbam = m
            break
            
    if target_layer_cbam:
        handle_cbam = target_layer_cbam.register_forward_hook(hook_cbam.hook_fn)
        model_cbam(IMAGE_PATH, verbose=False)
        handle_cbam.remove()
        
        heatmap_cbam = generate_heatmap(hook_cbam.activation, (img_w, img_h))
        vis_cbam = overlay_heatmap(img, heatmap_cbam)
    else:
        print("Warning: SPPF layer not found in CBAM model.")
        vis_cbam = img.copy()

    # --- Combine Results ---
    # Layout: [Original] [Baseline Heatmap] [CBAM Heatmap]
    
    # Add titles
    def add_title(img, text):
        img_copy = img.copy()
        cv2.putText(img_copy, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return img_copy

    res_orig = add_title(img, "Original")
    res_base = add_title(vis_baseline, "Baseline (SPPF)")
    res_cbam = add_title(vis_cbam, "CBAM Model (SPPF)")

    final_result = np.hstack([res_orig, res_base, res_cbam])
    
    cv2.imwrite(OUTPUT_PATH, final_result)
    print(f"Comparison saved to: {os.path.abspath(OUTPUT_PATH)}")
    
    # Optional: Show if environment supports it (might fail in headless)
    # cv2.imshow("Comparison", final_result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
