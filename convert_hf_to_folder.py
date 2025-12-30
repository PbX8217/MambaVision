import os
import argparse
from datasets import load_dataset, Image
from tqdm import tqdm
import shutil

def convert_hf_imagenet(src_dir, dst_dir):
    print(f"Loading dataset from {src_dir}...")
    # 首先加载一次以获取类别名称
    try:
        # 尝试作为本地脚本或文件夹加载
        ds_info = load_dataset(src_dir, split="train", streaming=True)
        features = ds_info.features
        if 'label' in features and hasattr(features['label'], 'names'):
            classes = features['label'].names
            print(f"Found {len(classes)} classes.")
        else:
            print("Warning: 'label' feature not found or has no names. Using indices.")
            classes = None
    except Exception as e:
        print(f"Error inspecting dataset: {e}")
        return

    # 重新加载数据集，这次设置 decode=False 以获取原始字节，避免重新编码
    print("Loading dataset with decode=False for fast export...")
    dataset = load_dataset(src_dir)
    dataset = dataset.cast_column("image", Image(decode=False))

    splits_mapping = {
        'train': 'train',
        'validation': 'val',
        'test': 'test'
    }

    for split_name in dataset.keys():
        if split_name not in splits_mapping:
            print(f"Skipping unknown split: {split_name}")
            continue
        
        target_split = splits_mapping[split_name]
        print(f"Processing {split_name} -> {target_split}...")
        
        ds = dataset[split_name]
        output_root = os.path.join(dst_dir, target_split)
        
        for i, item in tqdm(enumerate(ds), total=len(ds)):
            img_info = item['image']
            label_idx = item['label']
            
            # 获取类别名称作为文件夹名
            if classes:
                class_name = classes[label_idx]
                # 处理可能包含特殊字符的类别名，通常 ImageNet 是 nXXXXXXX
                # 如果是 "goldfish, Carassius auratus"，取第一部分或整个作为文件夹名均可
                # 标准 ImageNet 文件夹通常是 nXXXXXXX
                # 如果 classes 里面是 nXXXXXXX 最好。如果不是，可能需要额外的映射。
                # 这里假设 classes 就是我们想要的文件夹名。
                class_name = class_name.split(',')[0].strip()
            else:
                class_name = str(label_idx)
            
            class_dir = os.path.join(output_root, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # 确定文件名和扩展名
            # img_info['path'] 可能包含原始文件名
            filename = img_info.get('path')
            if not filename:
                filename = f"{i}.jpg"
            else:
                filename = os.path.basename(filename)
            
            # 确保扩展名正确 (ImageNet 主要是 JPEG)
            if 'bytes' in img_info and img_info['bytes']:
                image_bytes = img_info['bytes']
                target_path = os.path.join(class_dir, filename)
                
                # 直接写入字节
                with open(target_path, 'wb') as f:
                    f.write(image_bytes)
            else:
                # 理论上 decode=False 应该总是有 bytes，除非是本地文件路径模式
                # 如果只有 path，就复制文件
                src_path = img_info.get('path')
                if src_path and os.path.exists(src_path):
                    target_path = os.path.join(class_dir, filename)
                    shutil.copy2(src_path, target_path)
                else:
                    print(f"Warning: No bytes or valid path for item {i}")

    print(f"Conversion complete. Data saved to {dst_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Hugging Face ImageNet parquet to folder structure")
    parser.add_argument("--src", type=str, required=True, help="Path to the HF dataset directory (containing parquet files)")
    parser.add_argument("--dst", type=str, required=True, help="Path to output directory")
    args = parser.parse_args()
    
    convert_hf_imagenet(args.src, args.dst)
