import os
import torch
import yaml
import random
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

# 环境配置（GPU加速优化）
os.environ['YOLO_ASSETS_URL'] = 'https://mirror.tuna.tsinghua.edu.cn/github-release/ultralytics/assets/LatestRelease/'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 指定使用第一个GPU

# 固定随机种子
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def create_dataset_yaml():
    project_dir = r"D:\BaiduNetdiskDownload\YOLO5_2\YOLO5\yolov5-master"
    train_path = os.path.join(project_dir, "YOLO_dataset_5_classes", "images", "train")
    val_path = os.path.join(project_dir, "YOLO_dataset_5_classes", "images", "valid")
    class_names = ['no_fracture', 'mild_fracture', 'moderate_fracture', 'severe_fracture', 'critical_fracture']
    
    yaml_content = {
        'train': train_path,
        'val': val_path,
        'nc': 5,
        'names': class_names
    }
    yaml_path = "rib_dataset_5_classes.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)
    return yaml_path

def get_next_run_name(base_dir):
    if not os.path.exists(base_dir):
        return "yolov9c_5_classes_train1"
    
    max_num = 0
    for name in os.listdir(base_dir):
        if name.startswith("yolov9c_5_classes_train") and name[22:].isdigit():
            num = int(name[22:])
            if num > max_num:
                max_num = num
    return f"yolov9c_5_classes_train{max_num + 1}"

def train_yolov9c_5_classes():
    print("="*80)
    print("YOLOv9c + 5类别（含no_fracture） + CT图像增强策略")
    print("="*80)
    
    # 创建数据集配置
    data_yaml = create_dataset_yaml()
    print(f"数据集配置: {data_yaml}")
    
    # 输出目录
    output_base = r"D:\BaiduNetdiskDownload\YOLO5_2\YOLO5\yolov5-master\runs\detect"
    run_name = get_next_run_name(output_base)
    print(f"输出目录: {os.path.join(output_base, run_name)}")
    
    # 加载模型（强制使用GPU）
    print("\n🚀 加载YOLOv9c模型（GPU）...")
    if torch.cuda.is_available():
        model = YOLO('yolov9c.pt').to('cuda')
        device = "0"
        print("✅ 成功使用GPU (CUDA)")
    else:
        model = YOLO('yolov9c.pt')
        device = "cpu"
        print("⚠️  GPU不可用，使用CPU")
    
    # CT图像增强配置
    print("\n📋 CT图像增强策略:")
    print("  hsv_h: 0.0        (关闭色调)")
    print("  hsv_s: 0.0        (关闭饱和度)")
    print("  hsv_v: 0.2        (轻微亮度)")
    print("  degrees: 1.0      (极微旋转)")
    print("  translate: 0.01    (小幅平移)")
    print("  scale: 0.15        (小幅缩放)")
    print("  shear: 0.0         (关闭剪切)")
    print("  flipud: 0.0        (关闭上下翻转)")
    print("  fliplr: 0.0        (关闭左右翻转)")
    print("  mosaic: 0.5        (适度mosaic)")
    print("  mixup: 0.0         (关闭mixup)")
    print("  copy_paste: 0.1    (小幅copy_paste)")
    
    # 开始训练
    print("\n🏃 开始训练（1个epoch）...")
    results = model.train(
        data=data_yaml,
        epochs=5,
        batch=4,
        imgsz=896,
        device=device,
        project=output_base,
        name=run_name,
        exist_ok=True,
        save=True,
        save_period=1,
        val=True,
        workers=0,
        cache=False,
        amp=True,
        optimizer='SGD',
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        # CT图像增强参数
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.2,
        degrees=1.0,
        translate=0.01,
        scale=0.15,
        shear=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.1,
        patience=10,
        weight_decay=0.0005,
        conf=0.25,
        iou=0.45
    )
    
    # 训练完成后，随机选1张验证图检测
    print("\n" + "="*80)
    print("📊 训练后检测效果展示：")
    print("="*80)
    
    # 加载数据集配置
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
    
    # 筛选验证集图片
    val_images = []
    for f in os.listdir(data_cfg['val']):
        if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            val_images.append(os.path.join(data_cfg['val'], f))
    
    if val_images:
        test_img = random.choice(val_images)
        print(f"🔍 检测图片路径: {test_img}")
        
        detect_results = model.predict(
            source=test_img,
            imgsz=896,
            device=device,
            conf=0.2,
            iou=0.4,
            save=True,
            save_txt=True,
            project=output_base,
            name=f"{run_name}_detect",
            exist_ok=True
        )
        
        for r in detect_results:
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes
                print(f"✅ 检测到骨折目标数: {len(boxes)}")
                for i, box in enumerate(boxes):
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = data_cfg['names'][cls] if cls < len(data_cfg['names']) else 'unknown'
                    print(f"  目标{i+1}: {class_name} (置信度: {conf:.2f})")
            else:
                print("⚠️  未检测到骨折目标")
    else:
        print("⚠️  验证集无有效图片")
    
    # 打印权重保存路径
    weights_path = os.path.join(output_base, run_name, "weights", "last.pt")
    if os.path.exists(weights_path):
        print(f"\n📌 训练权重保存路径: {weights_path}")
    else:
        print(f"\n⚠️  权重未找到，请检查输出目录")
    
    print("\n✅ 训练完成！")
    return model, results

if __name__ == '__main__':
    try:
        model, results = train_yolov9c_5_classes()
    except Exception as e:
        print(f"\n❌ 运行出错: {str(e)}")
        import traceback
        traceback.print_exc()
