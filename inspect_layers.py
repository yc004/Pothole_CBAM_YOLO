from ultralytics import YOLO

def inspect_model(weights_path):
    print(f"Loading {weights_path}...")
    model = YOLO(weights_path)
    print(model.model)

if __name__ == "__main__":
    print("--- Baseline ---")
    inspect_model("Pothole_Baseline_Project/exp_baseline/weights/best.pt")
    print("\n--- CBAM ---")
    inspect_model("Pothole_CBAM_Project/exp_cbam/weights/best.pt")
