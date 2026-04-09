import cv2
import torch
import numpy as np
from ultralytics import YOLO
import supervision as sv
from pathlib import Path
import time
from omegaconf import OmegaConf


class YOLOWorldTester:
    def __init__(self, config_path="annotator.yaml"):
        """从配置文件初始化 YOLO-World"""
        # 加载配置
        self.cfg = OmegaConf.load(config_path)

        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 使用设备: {self.device}")

        # 加载 YOLO-World 模型
        print(f"📦 加载 YOLO-World 模型: {self.cfg.yolo_weights}")
        self.model = YOLO(self.cfg.yolo_weights)
        self.model.to(self.device)

        # 提取所有提示词
        self.all_prompts = []
        self.prompt_to_class = {}
        for class_item in self.cfg.classes:
            for prompt in class_item.prompts:
                self.all_prompts.append(prompt)
                self.prompt_to_class[prompt] = class_item.name

        print(f"✅ YOLO-World 模型加载成功！")
        print(f"📝 共 {len(self.all_prompts)} 个提示词，涵盖 {len(self.cfg.classes)} 个类别")

        # 检测参数
        self.conf_threshold = self.cfg.conf.detect

    def fix_results(self, results):
        """修复 YOLO 结果的类型问题"""
        if results is None:
            return results

        # 修复 names 字典的键类型
        if hasattr(results, 'names') and results.names:
            results.names = {int(k): str(v) for k, v in results.names.items()}

        return results

    def convert_to_detections(self, results):
        """将 YOLO 结果转换为 supervision Detections"""
        if results is None or results.boxes is None:
            return sv.Detections.empty(), results

        try:
            # 修复 names
            results = self.fix_results(results)

            # 获取数据
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy()

            # 转换为 Python int 类型
            class_ids = class_ids.astype(np.int32)

            # 创建 Detections
            detections = sv.Detections(
                xyxy=boxes,
                confidence=confidences,
                class_id=class_ids
            )

            return detections, results

        except Exception as e:
            print(f"⚠️  转换检测结果时出错: {e}")
            return sv.Detections.empty(), results

    def test_single_image(self, image_path, specific_prompts=None):
        """测试单张图像"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 无法读取图像: {image_path}")
            return None, None, 0

        print(f"\n{'=' * 60}")
        print(f"📸 测试图像: {Path(image_path).name}")
        print(f"📐 图像尺寸: {image.shape}")
        print(f"{'=' * 60}")

        # 选择要检测的提示词
        if specific_prompts:
            prompts = specific_prompts
        else:
            prompts = self.all_prompts

        print(f"🔍 检测提示词 ({len(prompts)}个)")
        if len(prompts) <= 10:
            print(f"   {prompts}")
        else:
            print(f"   {prompts[:5]}... (共{len(prompts)}个)")

        # 设置文本提示并检测
        print(f"\n⏱️  开始检测...")
        start_time = time.time()

        try:
            # 设置类别提示词
            self.model.set_classes(prompts)

            # 执行预测
            results = self.model.predict(
                image,
                conf=self.conf_threshold,
                verbose=False,
                device=self.device
            )[0]

            inference_time = time.time() - start_time

            # 转换为 Detections
            detections, results = self.convert_to_detections(results)

        except Exception as e:
            print(f"❌ 检测失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None, 0

        # 显示结果
        print(f"\n✨ 检测完成！")
        print(f"⏱️  推理时间: {inference_time:.3f} 秒")

        if len(detections) > 0:
            print(f"\n✅ 检测到 {len(detections)} 个目标:")
            print(f"{'序号':<4} {'类别':<15} {'置信度':<8} {'边界框':<30}")
            print("-" * 70)

            # 获取类别名称映射
            class_names_map = {}
            if hasattr(results, 'names') and results.names:
                class_names_map = {int(k): str(v) for k, v in results.names.items()}

            for i, (xyxy, conf, class_id) in enumerate(
                    zip(detections.xyxy, detections.confidence, detections.class_id)):
                # 获取类别名称
                class_id_int = int(class_id)
                if class_id_int in class_names_map:
                    class_name = class_names_map[class_id_int]
                else:
                    class_name = f"class_{class_id_int}"

                original_class = self.prompt_to_class.get(class_name, class_name)
                x1, y1, x2, y2 = xyxy
                print(f"{i + 1:<4} {original_class:<15} {conf:.3f}     [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
        else:
            print(f"\n❌ 未检测到任何目标")
            print("💡 提示: 可以尝试降低置信度阈值或检查图像内容")

        return detections, results, inference_time

    def visualize_results(self, image_path, detections, results, output_path=None):
        """可视化检测结果 - 使用 OpenCV 直接绘制（避免 supervision 兼容性问题）"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print("⚠️  无法读取图像")
            return None

        if detections is None or len(detections) == 0:
            print("⚠️  没有检测结果可显示")
            return None

        # 复制图像用于绘制
        annotated = image.copy()

        # 获取类别名称映射
        class_names_map = {}
        if results and hasattr(results, 'names') and results.names:
            class_names_map = {int(k): str(v) for k, v in results.names.items()}

        # 定义颜色列表（BGR格式）
        colors = [
            (0, 255, 0),  # 绿色
            (0, 0, 255),  # 红色
            (255, 0, 0),  # 蓝色
            (0, 255, 255),  # 黄色
            (255, 0, 255),  # 品红
            (255, 255, 0),  # 青色
        ]

        # 为每个检测绘制边界框和标签
        for i, (xyxy, conf, class_id) in enumerate(zip(detections.xyxy, detections.confidence, detections.class_id)):
            # 获取边界框坐标
            x1, y1, x2, y2 = [int(v) for v in xyxy]

            # 选择颜色（根据类别ID循环使用）
            color = colors[class_id % len(colors)]

            # 绘制边界框
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # 获取类别名称
            class_id_int = int(class_id)
            if class_id_int in class_names_map:
                class_name = class_names_map[class_id_int]
                # 尝试映射到原始类别名称
                original_class = self.prompt_to_class.get(class_name, class_name)
            else:
                original_class = f"class_{class_id_int}"

            # 创建标签文本
            label = f"{original_class} {conf:.2f}"

            # 计算文本大小
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # 绘制标签背景
            cv2.rectangle(annotated,
                          (x1, y1 - text_height - 5),
                          (x1 + text_width, y1),
                          color, -1)

            # 绘制标签文本
            cv2.putText(annotated, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 添加统计信息
        info_text = f"Detections: {len(detections)}"
        cv2.putText(annotated, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 保存或显示结果
        if output_path:
            cv2.imwrite(output_path, annotated)
            print(f"   💾 结果已保存: {output_path}")
        else:
            # 显示图像
            cv2.imshow("YOLO-World Detection", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return annotated

    def test_multiple_images(self, input_dir, output_dir=None):
        """测试多张图像"""
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
        print(f"📂 批量测试 {len(image_files)} 张图像")
        print(f"{'=' * 60}")

        # 设置输出目录
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        all_results = []
        total_time = 0

        for i, img_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] 处理: {img_file.name}")

            detections, results, infer_time = self.test_single_image(str(img_file))
            total_time += infer_time

            if detections is not None and len(detections) > 0 and output_dir:
                # 保存可视化结果
                output_file = output_path / f"yolo_{img_file.stem}_result.jpg"
                self.visualize_results(str(img_file), detections, results, str(output_file))

            all_results.append({
                'image': img_file.name,
                'detections': len(detections) if detections else 0,
                'time': infer_time
            })

        # 打印总结
        print(f"\n{'=' * 60}")
        print(f"📊 批量测试总结")
        print(f"{'=' * 60}")
        print(f"总图像数: {len(image_files)}")
        print(f"总检测目标数: {sum(r['detections'] for r in all_results)}")
        print(f"总推理时间: {total_time:.3f} 秒")
        print(f"平均每张: {total_time / len(image_files):.3f} 秒")

        return all_results

    def analyze_model_info(self):
        """分析模型信息"""
        print("\n" + "=" * 60)
        print("📊 模型信息分析")
        print("=" * 60)

        # 检查模型是否有 names 属性
        if hasattr(self.model, 'names'):
            print(f"模型类别数量: {len(self.model.names)}")
            print(f"模型类别示例: {list(self.model.names.values())[:10]}")
        else:
            print("模型没有预设类别名称（YOLO-World 动态类别）")

        # 检查模型结构
        print(f"模型类型: {type(self.model)}")

        # 检查是否支持 set_classes
        if hasattr(self.model, 'set_classes'):
            print("✅ 模型支持 set_classes 方法（YOLO-World）")
        else:
            print("⚠️  模型不支持 set_classes 方法")


# 测试代码
if __name__ == "__main__":
    # 初始化测试器
    tester = YOLOWorldTester("config/annotator.yaml")

    # 分析模型信息
    tester.analyze_model_info()

    # 测试单张图像
    image_path = "labelinput/1/00904.jpg"
    if Path(image_path).exists():
        # 测试所有类别
        print("\n" + "=" * 60)
        print("测试1: 检测所有类别")
        print("=" * 60)
        detections, results, _ = tester.test_single_image(image_path)

        # 可视化
        if detections and len(detections) > 0:
            tester.visualize_results(image_path, detections, results, "yolo_result_all.jpg")
            print("\n✅ 可视化结果已保存为: yolo_result_all.jpg")
        else:
            print("未检测到目标，尝试降低置信度阈值...")
            # 临时降低阈值再试一次
            original_threshold = tester.conf_threshold
            tester.conf_threshold = 0.05
            detections, results, _ = tester.test_single_image(image_path)
            tester.conf_threshold = original_threshold
            if detections and len(detections) > 0:
                tester.visualize_results(image_path, detections, results, "yolo_result_all_lowconf.jpg")
                print("✅ 可视化结果已保存为: yolo_result_all_lowconf.jpg")

        # 测试特定类别
        print("\n" + "=" * 60)
        print("测试2: 检测特定类别 (road, car, person, traffic lights)")
        print("=" * 60)
        specific_prompts = ["road", "car", "person", "traffic lights"]
        specific_detections, results_specific, _ = tester.test_single_image(
            image_path,
            specific_prompts=specific_prompts
        )

        if specific_detections and len(specific_detections) > 0:
            tester.visualize_results(image_path, specific_detections, results_specific, "yolo_result_specific.jpg")
            print("\n✅ 特定类别检测可视化已保存为: yolo_result_specific.jpg")
        else:
            print("未检测到特定类别目标")
    else:
        print(f"⚠️  图像文件不存在: {image_path}")
        print("请确保 image.png 在当前目录")