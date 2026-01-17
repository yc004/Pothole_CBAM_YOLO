import torch

def transfer_weights_pro():
    # 1. 加载官方预训练权重
    try:
        # 尝试加载本地文件，如果没有则需要用户手动下载
        if not os.path.exists('yolov8n.pt'):
             # 这里不自动下载，假设用户已有或者能自动下载
             pass
        base_ckpt = torch.load('yolov8n.pt', map_location='cpu')
        base_model = base_ckpt['model']
        print("✅ 成功加载官方权重 yolov8n.pt")
    except FileNotFoundError:
        print("❌ 错误: 当前目录下找不到 yolov8n.pt，请先下载官方权重！")
        return

    # 2. 定义层索引映射关系 (Base -> Pro)
    # 原始 YOLOv8n 的层结构 (简化版):
    # Backbone: 0-9 (9是SPPF)
    # Head: 10(Up), 11(Cat), 12(C2f), 13(Up), 14(Cat), 15(C2f), 16(Conv), 17(Cat), 18(C2f), 19(Conv), 20(Cat), 21(C2f), 22(Detect)
    
    # Pro 版的层结构变动:
    # 插入点 1 (Layer 9): CBAM -> 原 Layer 9 (SPPF) 变为 10. (后续 +1)
    # 插入点 2 (Layer 17): CBAM (Head P3) -> 原 Layer 16 (Conv) 后的结构顺延.
    # 插入点 3 (Layer 21): CBAM (Head P4)
    # 插入点 4 (Layer 25): CBAM (Head P5)
    
    # 映射字典: {原层索引: 新层索引}
    # 注意: 只映射有参数的层 (Conv, C2f, Detect 等)
    
    # 阶段 1: Backbone (0-8) -> 保持不变 (0-8)
    # 阶段 2: SPPF (原9) -> 变为 10 (因为插入了 Layer 9 CBAM)
    # 阶段 3: Head 上采样部分 (原10-15) -> 变为 11-16 (偏移 +1)
    # 阶段 4: Head P3 融合后 (原16 Conv) -> 变为 18 (因为 Layer 17 插入了 CBAM, 且之前已偏移 +1, 这里是新插入前的层? 不对，看yaml)
    
    # 让我们重新梳理 YAML 结构:
    # Base Layer | Pro Layer | 说明
    # ----------------------------
    # 0-8        | 0-8       | 对应
    # -          | 9         | [插入] CBAM
    # 9 (SPPF)   | 10        | 偏移 +1
    # 10 (Up)    | 11        | 偏移 +1
    # 11 (Cat)   | 12        | 偏移 +1
    # 12 (C2f)   | 13        | 偏移 +1
    # 13 (Up)    | 14        | 偏移 +1
    # 14 (Cat)   | 15        | 偏移 +1
    # 15 (C2f)   | 16        | 偏移 +1
    # -          | 17        | [插入] CBAM (P3)
    # 16 (Conv)  | 18        | 偏移 +2
    # 17 (Cat)   | 19        | 偏移 +2
    # 18 (C2f)   | 20        | 偏移 +2
    # -          | 21        | [插入] CBAM (P4)
    # 19 (Conv)  | 22        | 偏移 +3
    # 20 (Cat)   | 23        | 偏移 +3
    # 21 (C2f)   | 24        | 偏移 +3
    # -          | 25        | [插入] CBAM (P5)
    # 22 (Detect)| 26        | 偏移 +4 (注意: Detect 层的输入来源也变了，但权重本身是可以复用的)

    layer_map = {}
    
    # 0-8: 直通
    for i in range(9):
        layer_map[i] = i
        
    # 9-15: +1
    for i in range(9, 16):
        layer_map[i] = i + 1
        
    # 16-18: +2
    for i in range(16, 19):
        layer_map[i] = i + 2
        
    # 19-21: +3
    for i in range(19, 22):
        layer_map[i] = i + 3
        
    # 22 (Detect): +4
    layer_map[22] = 26

    # 3. 构建新权重字典
    new_state_dict = {}
    matched_count = 0
    total_count = 0
    
    print("\n🔄 开始迁移权重...")
    
    for key, value in base_model.state_dict().items():
        total_count += 1
        parts = key.split('.')
        
        # 提取层索引 (model.X.xxx)
        if len(parts) > 1 and parts[0] == 'model' and parts[1].isdigit():
            layer_idx = int(parts[1])
            suffix = '.'.join(parts[2:])
            
            if layer_idx in layer_map:
                new_idx = layer_map[layer_idx]
                new_key = f"model.{new_idx}.{suffix}"
                new_state_dict[new_key] = value
                matched_count += 1
                # print(f"  映射: {key} -> {new_key}")
            else:
                # 理论上不应该有漏掉的层，除非是 anchors 等不需要迁移的buffer
                pass
        else:
            # 非层权重的部分 (如 meta info)，直接保留
            new_state_dict[key] = value

    # 4. 保存新权重
    new_ckpt = base_ckpt.copy()
    new_ckpt['model'].load_state_dict(new_state_dict, strict=False)
    
    save_path = 'yolov8n_cbam_pro_pretrained.pt'
    torch.save(new_ckpt, save_path)
    
    print(f"\n✅ 权重迁移完成！")
    print(f"   原模型层数: {len(base_model.state_dict())}")
    print(f"   成功迁移层数: {matched_count}")
    print(f"   新权重已保存至: {save_path}")
    print("   (注意: 新插入的 CBAM 层和 Detect 头的输入部分将使用随机初始化)")

if __name__ == '__main__':
    import sys
    import os
    transfer_weights_pro()
