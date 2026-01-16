import torch
from ultralytics import YOLO
import numpy as np

def debug_model(weights_path, model_name):
    print(f"\n=== Debugging {model_name} ({weights_path}) ===")
    try:
        model = YOLO(weights_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 1. Print Structure
    print("--- Model Structure (Top Level) ---")
    if hasattr(model.model, 'model'):
        for i, m in enumerate(model.model.model):
            print(f"Layer {i}: {m.__class__.__name__}")
    else:
        print("Could not access model.model.model")
        return

    # 2. Test Inference and Hook
    print("\n--- Testing Activation Hooks ---")
    
    # Create dummy input (1, 3, 640, 640)
    dummy_img = torch.zeros((1, 3, 640, 640)).to(next(model.parameters()).device)
    
    activations = {}
    
    def get_hook(name):
        def hook(model, input, output):
            if isinstance(output, torch.Tensor):
                activations[name] = {
                    "shape": output.shape,
                    "max": output.max().item(),
                    "min": output.min().item(),
                    "mean": output.mean().item()
                }
            elif isinstance(output, (list, tuple)):
                 # Detect layer usually returns list
                 pass
        return hook

    hooks = []
    
    # Register hooks for Layer 8, 9, 10
    layers_to_test = [8, 9, 10]
    if len(model.model.model) > 10:
        layers_to_test.append(11)
        
    for i in layers_to_test:
        if i < len(model.model.model):
            layer = model.model.model[i]
            h = layer.register_forward_hook(get_hook(f"Layer {i} ({layer.__class__.__name__})"))
            hooks.append(h)
            
    # Run inference
    model.model(dummy_img)
    
    # Remove hooks
    for h in hooks:
        h.remove()
        
    # Print results
    for name, stats in activations.items():
        print(f"{name}: Shape={stats['shape']}, Max={stats['max']:.4f}, Mean={stats['mean']:.4f}")
        if stats['max'] == 0:
            print(f"   ⚠️ WARNING: ZERO ACTIVATION in {name}")

if __name__ == "__main__":
    debug_model("Pothole_Baseline_Project/exp_baseline/weights/best.pt", "Baseline")
    debug_model("Pothole_CBAM_Project/exp_cbam/weights/best.pt", "CBAM")
