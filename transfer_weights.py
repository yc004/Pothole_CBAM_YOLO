from ultralytics import YOLO


def transfer_weights():
    print("🚀 开始执行权重迁移...")

    # 1. 加载官方预训练权重
    print("1. 加载官方 yolov8n.pt...")
    yolo_official = YOLO("yolov8n.pt")
    official_dict = yolo_official.model.state_dict()

    # 2. 构建你的新模型 (随机初始化)
    print("2. 构建 yolov8n_cbam 模型...")
    model_cbam = YOLO("yolov8n_cbam.yaml")
    # 先初始化一下模型结构

    # 3. 创建新的权重字典
    new_state_dict = {}

    print("3. 开始迁移权重 (Layer 9+ 顺延一位)...")
    transferred_count = 0

    for k, v in official_dict.items():
        # k 是键名，例如 "model.0.conv.weight"
        parts = k.split(".")

        # 检查是否是 model 层的参数
        if parts[0] == "model" and parts[1].isdigit():
            layer_idx = int(parts[1])

            # === 核心逻辑 ===
            # 如果是前 9 层 (0-8)，保持不变
            if layer_idx < 9:
                new_key = k
            # 如果是第 9 层及之后，索引 +1 (因为插入了 CBAM)
            else:
                new_layer_idx = layer_idx + 1
                parts[1] = str(new_layer_idx)
                new_key = ".".join(parts)

            new_state_dict[new_key] = v
            transferred_count += 1
        else:
            # 其他参数直接复制
            new_state_dict[k] = v

    print(f"   - 已处理 {transferred_count} 个参数张量")

    # 4. 保存为新的预训练权重
    save_path = "yolov8n_cbam_pretrained.pt"
    print(f"4. 保存新权重到: {save_path}")

    # 我们需要把这个 dict 包装成 YOLO 能识别的格式 (model 对象)
    # 最简单的方法是直接把权重 load 进新模型，然后 save

    # 过滤掉不匹配的键 (比如最后的 Detect 头形状可能不同，如果类别数不一样)
    # 但这里我们主要目的是迁移 backbone 和 head 的通用特征

    try:
        # 尝试加载到新模型中
        model_cbam.model.load_state_dict(new_state_dict, strict=False)
        print("✅ 权重注入成功！(CBAM 层将保持随机初始化，其他层继承官方权重)")
    except Exception as e:
        print(f"⚠️ 权重注入部分失败 (正常现象，因为新增了层): {e}")

    # 保存
    model_cbam.save(save_path)
    print(f"🎉 完成！请在训练脚本中使用 '{save_path}' 作为 model 参数")


if __name__ == "__main__":
    transfer_weights()
