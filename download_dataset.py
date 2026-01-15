from roboflow import Roboflow

# 请替换为您的 Roboflow API Key
# 您可以在 Roboflow 项目页面的 "Download this Dataset" 中找到完整的代码
API_KEY = "rE73tQrrn9WrkZqX284V"
WORKSPACE = "pothole"
PROJECT = "new-pothole-detection-hiafr"
VERSION = 1

def download_dataset():
    if API_KEY == "YOUR_API_KEY":
        print("❌ 请先在 download_dataset.py 文件中填写您的 Roboflow API Key")
        print("   或者直接将本地电脑的 datasets 文件夹上传到服务器")
        return

    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    dataset = project.version(VERSION).download("yolov8")
    
    # 移动数据集到 datasets 目录
    import shutil
    import os
    
    target_dir = "datasets/New_pothole_detection.v2i.yolov8"
    if os.path.exists(target_dir):
        print(f"⚠️ 目标目录 {target_dir} 已存在")
    else:
        # Roboflow 下载通常会创建一个以项目名命名的文件夹
        # 这里假设下载后的文件夹名与 dataset.location 一致
        print(f"✅ 数据集已下载到: {dataset.location}")
        # 根据实际情况可能需要移动或重命名，这里仅打印提示
        print(f"请确保数据集位于: {os.path.abspath(target_dir)}")

if __name__ == "__main__":
    try:
        import roboflow
        download_dataset()
    except ImportError:
        print("❌ 请先安装 roboflow: pip install roboflow")
    except Exception as e:
        print(f"❌ 下载出错: {e}")
