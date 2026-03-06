import os
import torch
import yaml
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

# 环境配置
os.environ['YOLO_ASSETS_URL'] = 'https://mirror.tuna.tsinghua.edu.cn/github-release/ultralytics/assets/LatestRelease/'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def create_dataset_yaml():
    train_path = r"D:\YOLO_dataset\images\train"
    val_path = r"D:\YOLO_dataset\images\valid"
    class_names = ['mild_fracture', 'moderate_fracture', 'severe_fracture', 'critical_fracture']
    
    yaml_content = {
        'train': train_path,
        'val': val_path,
        'nc': 4,
        'names': class_names
    }
    yaml_path = "rib_dataset.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)
    return yaml_path

def validate_model():
    print("="*80)
    print("继续训练模型验证 - continue_train4/last.pt")
    print("="*80)
    
    # 权重路径
    weights_path = r"D:\BaiduNetdiskDownload\YOLO5_2\YOLO5\yolov5-master\runs\detect\continue_train4\weights\last.pt"
    
    # 检查权重文件
    if not os.path.exists(weights_path):
        print(f"❌ 权重文件不存在: {weights_path}")
        return
    
    print(f"✅ 使用权重: {weights_path}")
    
    # 创建数据集配置
    data_yaml = create_dataset_yaml()
    
    # 加载模型
    print("\n🚀 加载模型...")
    model = YOLO(weights_path)
    
    # 选择设备
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
    
    # 执行验证
    print("\n📊 开始验证...")
    results = model.val(
        data=data_yaml,
        batch=4,
        imgsz=640,
        device=device,
        project="runs",
        name="validate_continue_train4",
        exist_ok=True,
        verbose=True,
        conf=0.001,  # 低置信度阈值以获得完整的PR曲线
        iou=0.6,
        max_det=300,
        half=True,
        plots=True,
        rect=False,
        workers=0,
    )
    
    # 打印验证结果
    print("\n" + "="*80)
    print("验证结果汇总")
    print("="*80)
    
    # 精确率 (Precision)
    precision = None
    try:
        precision = float(getattr(results.box, 'mp', None) or results.box.p.mean())
    except Exception:
        pass
    
    # 召回率 (Recall)
    recall = None
    try:
        recall = float(getattr(results.box, 'mr', None) or results.box.r.mean())
    except Exception:
        pass
    
    # mAP指标
    map50 = float(results.box.map50)
    map = float(results.box.map)
    
    # Loss指标
    loss_total = None
    loss_components = {}
    if hasattr(results, 'loss'):
        lobj = results.loss
        if hasattr(lobj, 'total') and getattr(lobj, 'total') is not None:
            try:
                loss_total = float(lobj.total)
            except Exception:
                loss_total = None
        for k in ['box', 'cls', 'dfl']:
            if hasattr(lobj, k):
                try:
                    val = getattr(lobj, k)
                    if isinstance(val, (int, float)):
                        loss_components[k] = float(val)
                except Exception:
                    pass
        if loss_total is None and loss_components:
            loss_total = sum(loss_components.values())
    
    # 打印所有指标
    print(f"{'指标':<30} {'值':<15}")
    print("-"*45)
    
    if precision is not None:
        print(f"{'精确率 (Precision)':<30} {precision:.6f}")
    else:
        print(f"{'精确率 (Precision)':<30} N/A")
    
    if recall is not None:
        print(f"{'召回率 (Recall)':<30} {recall:.6f}")
    else:
        print(f"{'召回率 (Recall)':<30} N/A")
    
    print(f"{'mAP@0.5':<30} {map50:.6f}")
    print(f"{'mAP@0.5:0.95':<30} {map:.6f}")
    
    print("-"*45)
    for k, v in loss_components.items():
        print(f"{('Loss_' + k):<30} {v:.6f}")
    if loss_total is not None:
        print(f"{'Loss_Total':<30} {loss_total:.6f}")
    
    print("="*80)
    
    # 保存结果到文件
    output_file = "validation_continue_train4.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("验证结果汇总 (continue_train4/last.pt)\n")
        f.write("="*80 + "\n")
        if precision is not None:
            f.write(f"精确率 (Precision): {precision:.6f}\n")
        if recall is not None:
            f.write(f"召回率 (Recall): {recall:.6f}\n")
        f.write(f"mAP@0.5: {map50:.6f}\n")
        f.write(f"mAP@0.5:0.95: {map:.6f}\n")
        for k, v in loss_components.items():
            f.write(f"Loss_{k}: {v:.6f}\n")
        if loss_total is not None:
            f.write(f"Loss_Total: {loss_total:.6f}\n")
        f.write("="*80 + "\n")
    
    print(f"\n✅ 验证结果已保存到: {os.path.abspath(output_file)}")
    print(f"📊 可视化图表保存在: runs/detect/validate_continue_train4/")
    
    return {
        'precision': precision,
        'recall': recall,
        'map50': map50,
        'map': map,
        'loss': loss_total,
        'loss_components': loss_components
    }

if __name__ == '__main__':
    try:
        metrics = validate_model()
        print("\n✅ 验证完成！")
    except Exception as e:
        print(f"\n❌ 验证出错: {str(e)}")
        import traceback
        traceback.print_exc()
