import os
import sys
import subprocess
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

class BagAnnotationPipeline:
    def __init__(self, config):
        self.config = config
        self.extract_script = Path("extract_images_from_bag.py")
        self.annotate_script = Path("Automated_labeling.py")
        
        self.stats = {
            "extraction": {"success": False, "output_dir": None, "time": 0},
            "annotation": {"success": False, "output_dir": None, "time": 0}
        }
    
    def run_extraction(self):
        """运行提取"""
        print("\n" + "=" * 70)
        print("📸 步骤 1: 从 ROS2 Bag 提取图片")
        print("=" * 70)
        
        # 直接运行提取脚本（使用其内置配置）
        print("执行提取脚本...")
        start_time = time.time()
        
        result = subprocess.run([sys.executable, str(self.extract_script)], capture_output=False)
        
        end_time = time.time()
        self.stats["extraction"]["time"] = end_time - start_time
        
        if result.returncode == 0:
            print(f"\n✅ 图片提取成功！耗时: {self.stats['extraction']['time']:.2f} 秒")
            self.stats["extraction"]["success"] = True
            # 提取脚本实际输出的目录
            self.stats["extraction"]["output_dir"] = "/home/why/rezone_space/bags/output_images_rosbag2_2026_03_10-10_46_58_停车场"
            return True
        else:
            print(f"\n❌ 图片提取失败")
            return False
    
    def run_annotation(self):
        """运行标注"""
        print("\n" + "=" * 70)
        print("🎨 步骤 2: YOLOv26 + SAM2 自动标注")
        print("=" * 70)
        
        if not self.stats["extraction"]["success"]:
            print("⚠️ 图片提取未成功，跳过标注")
            return False
        
        # 查找实际图片目录
        extract_dir = Path(self.stats["extraction"]["output_dir"])
        
        # 查找包含图片的子目录
        images_dir = extract_dir
        for subdir in extract_dir.iterdir():
            if subdir.is_dir() and (list(subdir.glob("*.jpg")) or list(subdir.glob("*.png"))):
                images_dir = subdir
                break
        
        print(f"图片目录: {images_dir}")
        
        # 检查图片
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        if not image_files:
            print(f"❌ 没有找到图片")
            return False
        
        print(f"找到 {len(image_files)} 张图片")
        
        # 输出目录
        output_dir = Path(self.config["output_base"]) / "annotated_images"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 构建命令
        cmd = [
            sys.executable, str(self.annotate_script),
            "--input", str(images_dir),
            "--output", str(output_dir),
            "--conf", str(self.config["conf"]),
            "--device", self.config["device"]
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        start_time = time.time()
        
        result = subprocess.run(cmd, capture_output=False)
        
        end_time = time.time()
        self.stats["annotation"]["time"] = end_time - start_time
        
        if result.returncode == 0:
            print(f"\n✅ 标注完成！耗时: {self.stats['annotation']['time']:.2f} 秒")
            print(f"输出目录: {output_dir}")
            self.stats["annotation"]["success"] = True
            self.stats["annotation"]["output_dir"] = str(output_dir)
            return True
        else:
            print(f"\n❌ 标注失败")
            return False
    
    def run(self):
        """运行流水线"""
        print("=" * 80)
        print("🚀 ROS2 Bag 提取 + YOLOv26+SAM2 标注流水线")
        print("=" * 80)
        print(f"📁 Bag 路径: {self.config['bag_path']}")
        print(f"📸 图像话题: {self.config['topics']}")
        print(f"🎯 置信度阈值: {self.config['conf']}")
        print("=" * 80)
        
        extraction_ok = self.run_extraction()
        annotation_ok = self.run_annotation()
        
        print("\n" + "=" * 80)
        if extraction_ok and annotation_ok:
            print("✅ 流水线执行成功！")
            print(f"   提取图片: {self.stats['extraction']['output_dir']}")
            print(f"   标注结果: {self.stats['annotation']['output_dir']}")
        else:
            print("⚠️ 流水线部分失败")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag_path', type=str, 
                        default="/home/why/rezone_space/bags/rosbag2_2026_03_10-10_46_58_停车场")
    parser.add_argument('--topics', nargs='+', default=["/metoak/right_image"])
    parser.add_argument('--step', type=float, default=3.0)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--output_base', type=str, default="/home/why/rezone_space/out")
    
    args = parser.parse_args()
    
    config = {
        "bag_path": args.bag_path,
        "topics": args.topics,
        "step": args.step,
        "conf": args.conf,
        "device": args.device,
        "output_base": args.output_base
    }
    
    pipeline = BagAnnotationPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
