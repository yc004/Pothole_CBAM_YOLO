import os
import shutil
import zipfile

from roboflow import Roboflow

# 请替换为您的 Roboflow API Key
API_KEY = "rE73tQrrn9WrkZqX284V"
WORKSPACE = "pothole"
PROJECT = "new-pothole-detection-q1hk4"
VERSION = 1


def download_dataset():
    if API_KEY == "YOUR_API_KEY":
        print("❌ 请先在 download_dataset.py 文件中填写您的 Roboflow API Key")
        print("   或者直接将本地电脑的 datasets 文件夹上传到服务器")
        return

    # 1. 确保 datasets 目录存在
    datasets_root = "datasets"
    if not os.path.exists(datasets_root):
        os.makedirs(datasets_root)
        print(f"✅ 创建目录: {datasets_root}")

    # 2. 定义目标路径
    target_dir_name = "New_pothole_detection.v2i.yolov8"
    target_dir = os.path.join(datasets_root, target_dir_name)
    target_dir_abs = os.path.abspath(target_dir)

    # 如果目标目录已存在，先清理（可选，为了确保是新的）
    # 这里我们选择如果存在就跳过下载，或者提示用户
    if os.path.exists(target_dir) and os.listdir(target_dir):
        print(f"⚠️ 目标目录 {target_dir} 已存在且不为空。跳过下载。")
        print("   如果需要重新下载，请先删除该目录。")
        return

    print("🚀 开始下载数据集...")
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)

    # 下载到默认位置
    # roboflow 默认下载到当前目录下的 {ProjectName}-{Version} 文件夹
    # 指定 model_format="yolov8"
    dataset = project.version(VERSION).download("yolov8")

    downloaded_location = dataset.location
    print(f"✅ 原始下载路径: {downloaded_location}")

    # 3. 处理下载后的文件
    print(f"📂 处理下载内容: {downloaded_location}")

    # 情况 A: 下载的是一个 zip 文件 (通过 zipfile 模块检测，不完全依赖后缀)
    if os.path.isfile(downloaded_location) and zipfile.is_zipfile(downloaded_location):
        print(f"📦 检测到压缩包，开始自动解压到 {target_dir}...")
        try:
            with zipfile.ZipFile(downloaded_location, "r") as zip_ref:
                zip_ref.extractall(target_dir)
            print("✅ 解压完成")
            # 删除 zip 包
            os.remove(downloaded_location)
            print("🗑️ 已删除原始压缩包")
        except Exception as e:
            print(f"❌ 解压失败: {e}")
            return

    # 情况 B: 下载的是一个文件夹 (Roboflow SDK 可能已经自动解压)
    elif os.path.isdir(downloaded_location):
        print("📂 下载内容已是文件夹 (Roboflow SDK 已自动解压)")

        # 检查是否就是目标文件夹
        if os.path.abspath(downloaded_location) == target_dir_abs:
            print("✅ 数据集已在正确位置")
        else:
            print(f"🚚 移动数据集到 {target_dir}...")
            # 如果目标文件夹不存在，直接移动
            if not os.path.exists(target_dir):
                shutil.move(downloaded_location, target_dir)
            else:
                # 如果存在（可能是空的），先删除再移动
                # 注意：这里假设目标目录是空的或者可以被覆盖
                try:
                    shutil.rmtree(target_dir)  # 强制删除旧目录
                    shutil.move(downloaded_location, target_dir)
                except Exception as e:
                    print(f"⚠️ 移动失败，请手动检查: {e}")
                    return
            print("✅ 移动完成")

    else:
        print(f"⚠️ 未知的文件类型或路径: {downloaded_location}")
        print("   请手动检查该路径下的内容。")
        return

    print(f"🎉 数据集准备就绪！位置: {target_dir}")
    print("💡 提示: 请确保 pothole_config.yaml 中的 'path' 指向此目录")


if __name__ == "__main__":
    try:
        download_dataset()
    except ImportError:
        print("❌ 请先安装 roboflow: pip install roboflow")
    except Exception as e:
        print(f"❌ 下载出错: {e}")
        import traceback

        traceback.print_exc()
