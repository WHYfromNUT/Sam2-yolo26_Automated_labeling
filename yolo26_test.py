import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


class YOLOv26Tester:
    def __init__(self, model_path="yolo26n.pt", device=None):
        """初始化 YOLOv26 模型"""
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"🔧 使用设备: {self.device}")

        # 加载 YOLOv26 模型
        print(f"📦 加载 YOLOv26 模型: {model_path}")

        # 检查模型文件是否存在，如果不存在则自动下载
        if not Path(model_path).exists():
            print(f"⚠️  模型文件不存在，正在下载...")

        self.model = YOLO(model_path)
        self.model.to(self.device)

        # 获取模型信息
        self.model_info = self.get_model_info()

        print(f"✅ YOLOv26 模型加载成功！")
        print(f"📊 模型信息: {self.model_info}")

    def get_model_info(self):
        """获取模型信息"""
        info = {}

        # 获取类别信息
        if hasattr(self.model, 'names'):
            info['num_classes'] = len(self.model.names)
            info['classes'] = list(self.model.names.values())
        else:
            info['num_classes'] = 0
            info['classes'] = []

        # 获取模型类型
        if hasattr(self.model, 'model'):
            info['model_type'] = type(self.model.model).__name__

        return info

    def detect_single_image(self, image_path, conf_threshold=0.25, iou_threshold=0.45,
                            max_det=300, output_path=None):
        """检测单张图像"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 无法读取图像: {image_path}")
            return None

        h, w = image.shape[:2]

        print(f"\n{'=' * 60}")
        print(f"📸 测试图像: {Path(image_path).name}")
        print(f"📐 图像尺寸: {h} x {w}")
        print(f"🎯 检测参数: conf={conf_threshold}, iou={iou_threshold}")
        print(f"{'=' * 60}")

        # 执行推理
        print(f"⏱️  开始检测...")
        start_time = time.time()

        results = self.model.predict(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            max_det=max_det,
            verbose=False,
            device=self.device
        )[0]

        inference_time = time.time() - start_time

        print(f"✨ 检测完成！")
        print(f"⏱️  推理时间: {inference_time:.3f} 秒")

        # 获取检测结果
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)

            print(f"\n✅ 检测到 {len(boxes)} 个目标:")
            print(f"{'序号':<4} {'类别':<20} {'置信度':<8} {'边界框':<35}")
            print("-" * 75)

            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                class_name = results.names[cls_id] if cls_id in results.names else f"class_{cls_id}"
                x1, y1, x2, y2 = [int(v) for v in box]
                print(f"{i + 1:<4} {class_name:<20} {conf:.3f}     [{x1}, {y1}, {x2}, {y2}]")

            # 保存可视化结果
            if output_path:
                self.visualize_results(image, results, output_path)

            return results, inference_time
        else:
            print(f"\n❌ 未检测到任何目标")
            return None, inference_time

    def detect_with_video(self, video_path, conf_threshold=0.25, output_path=None):
        """检测视频"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频: {video_path}")
            return

        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\n{'=' * 60}")
        print(f"🎬 视频: {Path(video_path).name}")
        print(f"📐 尺寸: {width}x{height}, FPS: {fps}, 总帧数: {total_frames}")
        print(f"{'=' * 60}")

        # 设置输出
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        total_time = 0
        detections_per_frame = []

        print(f"⏱️  开始处理视频...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 检测
            start_time = time.time()
            results = self.model.predict(frame, conf=conf_threshold, verbose=False)[0]
            inference_time = time.time() - start_time
            total_time += inference_time

            # 统计检测数量
            num_detections = len(results.boxes) if results.boxes is not None else 0
            detections_per_frame.append(num_detections)

            # 可视化
            if output_path:
                annotated_frame = results.plot()
                out.write(annotated_frame)

            # 进度显示
            if frame_count % 30 == 0:
                print(f"  处理进度: {frame_count}/{total_frames} 帧, "
                      f"检测到 {num_detections} 个目标, "
                      f"平均帧率: {frame_count / total_time:.1f} FPS")

        cap.release()
        if output_path:
            out.release()

        # 打印统计信息
        print(f"\n✨ 视频处理完成！")
        print(f"📊 统计信息:")
        print(f"   - 总帧数: {frame_count}")
        print(f"   - 总耗时: {total_time:.2f} 秒")
        print(f"   - 平均帧率: {frame_count / total_time:.1f} FPS")
        print(f"   - 平均每帧检测: {np.mean(detections_per_frame):.1f} 个目标")
        print(f"   - 最多检测: {max(detections_per_frame)} 个目标")

        if output_path:
            print(f"   💾 结果已保存: {output_path}")

    def detect_batch(self, input_dir, output_dir=None, conf_threshold=0.25):
        """批量检测图像"""
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"❌ 目录不存在: {input_dir}")
            return

        # 获取所有图像
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]

        if not image_files:
            print(f"❌ 目录中没有图像文件: {input_dir}")
            return

        print(f"\n{'=' * 60}")
        print(f"📂 批量检测 {len(image_files)} 张图像")
        print(f"{'=' * 60}")

        # 设置输出目录
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        results_summary = []
        total_time = 0

        for i, img_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] 处理: {img_file.name}")

            # 检测
            results, infer_time = self.detect_single_image(
                str(img_file),
                conf_threshold=conf_threshold,
                output_path=str(output_path / f"result_{img_file.name}") if output_dir else None
            )

            total_time += infer_time

            # 统计
            num_detections = len(results.boxes) if results and results.boxes is not None else 0
            results_summary.append({
                'image': img_file.name,
                'detections': num_detections,
                'time': infer_time
            })

        # 打印总结
        print(f"\n{'=' * 60}")
        print(f"📊 批量检测总结")
        print(f"{'=' * 60}")
        print(f"总图像数: {len(image_files)}")
        print(f"总检测目标数: {sum(r['detections'] for r in results_summary)}")
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"平均每张: {total_time / len(image_files):.2f} 秒")

        return results_summary

    def visualize_results(self, image, results, output_path):
        """可视化检测结果"""
        # 使用 YOLO 自带的可视化
        annotated = results.plot()

        # 保存
        cv2.imwrite(output_path, annotated)
        print(f"   💾 结果已保存: {output_path}")

    def analyze_model(self):
        """分析模型结构和性能"""
        print("\n" + "=" * 60)
        print("🔍 YOLOv26 模型分析")
        print("=" * 60)

        # 模型结构信息
        print(f"📊 模型类型: {self.model_info.get('model_type', 'Unknown')}")
        print(f"📊 类别数量: {self.model_info.get('num_classes', 0)}")

        if self.model_info.get('num_classes', 0) > 0:
            print(f"📊 前10个类别: {self.model_info['classes'][:10]}")

        # 模型参数
        if hasattr(self.model, 'model'):
            total_params = sum(p.numel() for p in self.model.model.parameters())
            print(f"📊 总参数数量: {total_params:,}")

        # 模型层级
        if hasattr(self.model, 'model'):
            print(f"📊 模型层数: {len(list(self.model.model.modules()))}")

    def benchmark(self, image_path, num_runs=10):
        """性能基准测试"""
        print("\n" + "=" * 60)
        print(f"⚡ 性能基准测试 (运行 {num_runs} 次)")
        print("=" * 60)

        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 无法读取图像: {image_path}")
            return

        times = []

        # 预热
        for _ in range(3):
            _ = self.model.predict(image, verbose=False)

        # 正式测试
        for i in range(num_runs):
            start_time = time.time()
            _ = self.model.predict(image, verbose=False)
            inference_time = time.time() - start_time
            times.append(inference_time)

        times = np.array(times)

        print(f"📊 测试结果:")
        print(f"   - 平均时间: {np.mean(times) * 1000:.2f} ms")
        print(f"   - 标准差: {np.std(times) * 1000:.2f} ms")
        print(f"   - 最小值: {np.min(times) * 1000:.2f} ms")
        print(f"   - 最大值: {np.max(times) * 1000:.2f} ms")
        print(f"   - 平均 FPS: {1 / np.mean(times):.1f}")

        return times


# 主测试函数
def main():
    print("=" * 70)
    print("🚀 YOLOv26 测试脚本")
    print("=" * 70)

    # 可用的 YOLOv26 模型
    models = {
        "n": "yolo26n.pt",  # Nano - 最快，最轻量
        "s": "yolo26s.pt",  # Small
        "m": "yolo26m.pt",  # Medium
        "l": "yolo26l.pt",  # Large
        "x": "weights/pretrained/yolo26x.pt",  # X-Large - 最准确，最慢
    }

    # 选择模型（默认使用 nano）
    model_size = "x"  # 可改为 s, m, l, x
    model_path = models[model_size]

    print(f"📦 使用模型: {model_path}")

    # 初始化测试器
    tester = YOLOv26Tester(model_path=model_path)

    # 分析模型
    tester.analyze_model()

    # 测试单张图像
    test_image = "labelinput/1/00904.jpg"
    if Path(test_image).exists():
        print("\n" + "=" * 70)
        print("测试1: 单张图像检测")
        print("=" * 70)

        # 不同置信度阈值测试
        for conf in [0.25, 0.5, 0.7]:
            print(f"\n--- 置信度阈值: {conf} ---")
            results, _ = tester.detect_single_image(
                test_image,
                conf_threshold=conf,
                output_path=f"yolo26_result_conf_{int(conf * 100)}.jpg"
            )

        # 性能测试
        print("\n" + "=" * 70)
        print("测试2: 性能基准测试")
        print("=" * 70)
        tester.benchmark(test_image, num_runs=10)

    else:
        print(f"\n⚠️  测试图像不存在: {test_image}")
        print("请确保 image.png 在当前目录")

    # 可选：批量检测
    # if Path("labelinput").exists():
    #     print("\n" + "="*70)
    #     print("测试3: 批量图像检测")
    #     print("="*70)
    #     tester.detect_batch("labelinput", "yolo26_output", conf_threshold=0.25)

    # 可选：视频检测
    # if Path("test_video.mp4").exists():
    #     print("\n" + "="*70)
    #     print("测试4: 视频检测")
    #     print("="*70)
    #     tester.detect_with_video("test_video.mp4", conf_threshold=0.25, output_path="yolo26_video_result.mp4")


if __name__ == "__main__":
    main()


