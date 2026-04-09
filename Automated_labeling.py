import cv2
import torch
import numpy as np
import supervision as sv
import json
import shutil
import time
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from omegaconf import OmegaConf
from hydra.utils import instantiate
from sam2.sam2_image_predictor import SAM2ImagePredictor
from typing import List, Dict, Tuple, Optional
import logging
import re

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =========================================================
# 工具函数：解析 YAML 类别配置
# =========================================================
def parse_classes(class_cfg):
    """
    解析类别配置，将多个 prompts 统一映射到同一个 name

    Args:
        class_cfg: 类别配置列表
        [
            {"name": "person", "prompts": ["person", "human", "man", "woman"]},
            {"name": "vehicle", "prompts": ["car", "bus", "truck", "vehicle"]}
        ]

    Returns:
        class_names: 类别名称列表，索引为类别ID
        prompts: 所有提示词列表
        prompt2cid: 提示词到类别ID的映射字典
        name_to_cid: 类别名称到类别ID的映射字典
    """
    class_names = []
    prompts = []
    prompt2cid = {}
    name_to_cid = {}

    for cid, item in enumerate(class_cfg):
        name = item["name"]
        class_names.append(name)
        name_to_cid[name] = cid

        for p in item["prompts"]:
            p_str = str(p).strip().lower()
            prompts.append(p_str)
            prompt2cid[p_str] = cid

    logger.info(f"解析完成: {len(class_names)} 个类别, {len(prompts)} 个提示词")
    logger.info(f"类别映射: {list(zip(class_names, range(len(class_names))))}")

    return class_names, prompts, prompt2cid, name_to_cid


class YOLOv26SAM2Annotator:
    """YOLOv26 + SAM2 自动标注器 - 支持类别统一映射"""

    def __init__(self, yolo_weights, sam2_checkpoint, sam2_config_path, device=None):
        """
        初始化标注器

        Args:
            yolo_weights: YOLOv26 模型权重路径
            sam2_checkpoint: SAM2 模型检查点路径
            sam2_config_path: SAM2 配置文件路径
            device: 运行设备 ('cuda' or 'cpu')
        """
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"运行设备: {self.device}")

        # 1. 加载 YOLOv26
        logger.info(f"加载 YOLOv26 模型: {yolo_weights}")
        self.yolo = YOLO(yolo_weights)
        self.yolo.to(self.device)

        # 获取模型预定义的类别名称
        self.model_class_names = self.yolo.names if hasattr(self.yolo, 'names') else {}
        logger.info(f"✅ YOLOv26 加载成功，模型预定义 {len(self.model_class_names)} 个类别")
        logger.info(f"模型类别示例: {list(self.model_class_names.values())[:10]}")

        # 2. 加载 SAM2
        logger.info("加载 SAM2 模型...")
        full_config = OmegaConf.load(sam2_config_path)
        sam_model = instantiate(full_config.model)
        checkpoint = torch.load(sam2_checkpoint, map_location="cpu", weights_only=True)
        sam_model.load_state_dict(checkpoint["model"], strict=False)
        sam_model.to(self.device).eval()
        self.sam2_predictor = SAM2ImagePredictor(sam_model)
        logger.info("✅ SAM2 加载成功")

        # 统计信息
        self.stats = {
            'total_images': 0,
            'total_objects': 0,
            'total_time': 0,
            'class_counts': {},
            'mapping_stats': {}  # 统计映射情况
        }

    def _map_yolo_to_custom_classes(self, detections, custom_class_names,
                                    custom_prompts, prompt2cid, name_to_cid):
        """
        将 YOLOv26 检测到的类别映射到自定义类别

        YOLOv26 只能检测其训练时定义的类别，需要通过多级匹配映射到自定义类别

        Args:
            detections: YOLO 检测结果
            custom_class_names: 自定义类别名称列表
            custom_prompts: 自定义提示词列表
            prompt2cid: 提示词到类别ID的映射
            name_to_cid: 类别名称到类别ID的映射

        Returns:
            映射后的 detections 和命中的提示词
        """
        if len(detections) == 0:
            return detections, []

        mapped_cids = []
        hit_prompts = []
        original_class_names = []

        # 构建匹配规则（不区分大小写）
        # 1. 直接类别名称映射
        name_to_cid_lower = {k.lower(): v for k, v in name_to_cid.items()}

        # 2. 提示词到类别的映射（已小写）
        prompt2cid_lower = {k.lower(): v for k, v in prompt2cid.items()}

        # 3. 关键词映射（用于模糊匹配）
        keyword_mapping = {
            'person': ['person', 'human', 'man', 'woman', 'people', 'pedestrian'],
            'vehicle': ['car', 'bus', 'truck', 'van', 'suv', 'vehicle',
                         'coach', 'shuttle', 'pickup', 'lorry','public transit','automobile'],
            'bicycle': ['bicycle', 'bike', 'cycle'],
            'motorcycle': ['motorcycle'],
            'E-bike': ['electric vehicle', 'E-bike', 'green vehicle', 'battery-electric vehicle'],
            'traffic lights': ['traffic light', 'traffic lights', 'stoplight', 'signal'],
            'lane': ['lane', 'driveway', 'carriageway', 'road lane'],
            'sidewalk': ['sidewalk', 'footway', 'pavement', 'footpath'],
            'crosswalk': ['crosswalk', 'zebra crossing', 'pelican crossing'],
            'blind path': ['blind path', 'tactile paving', 'detecting tiles','braille path'],
            'ramp': ['ramp'],
            'barrier': ['barrier','curb-boundary', 'vase','fire hydrant','flower bed', 'planter', 'ornamentation', 'potted plant'],
            'trash-bin': ['trash-bin'],
            'road': ['road', 'street', 'highway', 'path', 'route', 'way'],
            'stairs': ['stairs', 'staircase', 'steps'],
            'chair': ['chair', 'seat', 'bench'],
            'table': ['table', 'desk'],
            'door': ['door', 'gate', 'entrance']
        }

        for class_id in detections.class_id:
            # 获取 YOLO 检测到的原始类别名称
            original_name = self.model_class_names.get(int(class_id), f"class_{class_id}")
            original_name_lower = original_name.lower()
            original_class_names.append(original_name)

            matched_cid = None
            matched_prompt = None

            # === 方法1: 直接匹配类别名称 ===
            if original_name_lower in name_to_cid_lower:
                matched_cid = name_to_cid_lower[original_name_lower]
                matched_prompt = original_name
                logger.debug(f"直接匹配: {original_name} -> {custom_class_names[matched_cid]}")

            # === 方法2: 通过提示词匹配 ===
            if matched_cid is None:
                for prompt, cid in prompt2cid_lower.items():
                    if prompt in original_name_lower or original_name_lower in prompt:
                        matched_cid = cid
                        matched_prompt = prompt
                        logger.debug(
                            f"提示词匹配: {original_name} -> {custom_class_names[matched_cid]} (prompt: {prompt})")
                        break

            # === 方法3: 关键词映射 ===
            if matched_cid is None:
                for target_name, keywords in keyword_mapping.items():
                    for keyword in keywords:
                        if keyword in original_name_lower:
                            if target_name in name_to_cid_lower:
                                matched_cid = name_to_cid_lower[target_name]
                                matched_prompt = keyword
                                logger.debug(
                                    f"关键词匹配: {original_name} -> {custom_class_names[matched_cid]} (keyword: {keyword})")
                                break
                    if matched_cid is not None:
                        break

            # === 方法4: 模糊匹配（使用字符串相似度）===
            if matched_cid is None:
                best_match = None
                best_score = 0
                for custom_name in custom_class_names:
                    # 计算简单相似度
                    if custom_name.lower() in original_name_lower:
                        score = len(custom_name)
                        if score > best_score:
                            best_score = score
                            best_match = custom_name
                    elif original_name_lower in custom_name.lower():
                        score = len(original_name)
                        if score > best_score:
                            best_score = score
                            best_match = custom_name

                if best_match:
                    matched_cid = name_to_cid_lower[best_match.lower()]
                    matched_prompt = best_match
                    logger.debug(f"模糊匹配: {original_name} -> {custom_class_names[matched_cid]}")

            if matched_cid is not None:
                mapped_cids.append(matched_cid)
                hit_prompts.append(matched_prompt)
                # 统计映射
                key = f"{original_name} -> {custom_class_names[matched_cid]}"
                self.stats['mapping_stats'][key] = self.stats['mapping_stats'].get(key, 0) + 1
            else:
                # 未匹配，跳过这个检测
                mapped_cids.append(-1)
                hit_prompts.append("")
                logger.debug(f"未匹配: {original_name}")

        # 过滤掉未匹配的检测
        valid_indices = [i for i, cid in enumerate(mapped_cids) if cid != -1]
        if valid_indices:
            detections = detections[valid_indices]
            mapped_cids = [mapped_cids[i] for i in valid_indices]
            hit_prompts = [hit_prompts[i] for i in valid_indices]
            detections.class_id = np.array(mapped_cids)
        else:
            detections = sv.Detections.empty()
            hit_prompts = []

        return detections, hit_prompts

    def process(self, input_dir, output_dir, classes_cfg, conf_cfg, switches):
        """
        处理图像目录
        """
        start_time = time.time()

        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # === 1. 创建输出目录 ===
        dirs = self._create_output_dirs(output_path, switches)

        # === 2. 解析类别配置 ===
        custom_class_names, custom_prompts, prompt2cid, name_to_cid = parse_classes(classes_cfg)
        logger.info(f"自定义类别: {custom_class_names}")
        logger.info(f"提示词总数: {len(custom_prompts)}")

        # 获取置信度阈值
        detect_conf = conf_cfg.get("detect", 0.25)
        save_conf = conf_cfg.get("save", detect_conf)
        iou_threshold = conf_cfg.get("iou", 0.45)

        logger.info(f"检测参数: conf={detect_conf}, iou={iou_threshold}, save_conf={save_conf}")

        # === 3. 获取图像列表 ===
        image_files = self._get_image_files(input_path)
        if not image_files:
            logger.error(f"目录中没有图像文件: {input_dir}")
            return

        logger.info(f"找到 {len(image_files)} 张图像，开始处理...")
        self.stats['total_images'] = len(image_files)

        # === 4. 处理每张图像 ===
        ls_tasks = []
        failed_images = []

        for img_file in tqdm(image_files, desc="Processing"):
            try:
                result = self._process_single_image(
                    img_file, dirs, custom_class_names, custom_prompts,
                    prompt2cid, name_to_cid,
                    detect_conf, save_conf, iou_threshold, switches
                )
                if result:
                    image_result, detections, hit_prompts = result
                    if switches.get("save_studio", True) and len(detections) > 0:
                        h, w = image_result.shape[:2]
                        ls_tasks.append(self.format_ls_task(
                            detections, img_file.name, custom_class_names, (h, w)
                        ))

                    # 更新统计
                    self._update_stats(detections, custom_class_names)

            except Exception as e:
                logger.error(f"处理图像 {img_file.name} 失败: {e}")
                import traceback
                traceback.print_exc()
                failed_images.append(img_file.name)
                continue

        # === 5. 导出结果 ===
        if switches.get("save_studio", True) and ls_tasks:
            self._save_label_studio_tasks(ls_tasks, output_path)

        # === 6. 输出统计信息 ===
        self._print_stats(start_time, failed_images)

        logger.info(f"处理完成！结果保存在: {output_path}")

    def _process_single_image(self, img_file, dirs, custom_class_names, custom_prompts,
                              prompt2cid, name_to_cid, detect_conf, save_conf, iou_threshold, switches):
        """处理单张图像"""
        # 读取图像
        image_bgr = cv2.imread(str(img_file))
        if image_bgr is None:
            logger.warning(f"无法读取图像: {img_file.name}")
            return None

        h, w = image_bgr.shape[:2]

        # 拷贝图片
        if "images" in dirs:
            shutil.copy2(img_file, dirs["images"] / img_file.name)
        if "labelme" in dirs:
            shutil.copy2(img_file, dirs["labelme"] / img_file.name)

        # YOLOv26 检测
        results = self.yolo.predict(
            image_bgr,
            conf=detect_conf,
            iou=iou_threshold,
            verbose=False,
            device=self.device
        )[0]

        # 修复 names 键类型
        if hasattr(results, 'names') and results.names:
            results.names = {int(k): str(v) for k, v in results.names.items()}

        # 转换为 Detections
        detections = sv.Detections.from_ultralytics(results)

        if len(detections) == 0:
            self._save_empty_annotations(dirs, img_file, (h, w))
            return None

        # 映射类别
        detections, hit_prompts = self._map_yolo_to_custom_classes(
            detections, custom_class_names, custom_prompts, prompt2cid, name_to_cid
        )

        if len(detections) == 0:
            self._save_empty_annotations(dirs, img_file, (h, w))
            return None

        # 过滤低置信度检测
        keep = detections.confidence >= save_conf
        detections = detections[keep]
        hit_prompts = [p for i, p in enumerate(hit_prompts) if keep[i]]

        # SAM2 分割
        need_mask = self._need_mask(switches)
        if len(detections) > 0 and need_mask:
            detections = self._run_sam2_segmentation(image_bgr, detections)

            if "preview" in dirs:
                self.save_preview(image_bgr, detections, dirs["preview"] / img_file.name, custom_class_names)

        # 保存标注
        self._save_annotations(detections, hit_prompts, dirs, img_file, custom_class_names, (h, w))

        return image_bgr, detections, hit_prompts

    def _need_mask(self, switches):
        """判断是否需要运行 SAM2"""
        return (switches.get("save_labelme", True) or
                switches.get("save_custom", True) or
                switches.get("save_preview", True))

    def _run_sam2_segmentation(self, image_bgr, detections):
        """运行 SAM2 分割"""
        try:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            self.sam2_predictor.set_image(image_rgb)

            boxes = detections.xyxy
            masks = []

            # 分批处理
            batch_size = 10
            for i in range(0, len(boxes), batch_size):
                batch_boxes = boxes[i:i + batch_size]
                batch_masks, _, _ = self.sam2_predictor.predict(
                    box=batch_boxes,
                    multimask_output=False
                )
                if batch_masks.ndim == 4:
                    batch_masks = batch_masks.squeeze(1)
                masks.extend(batch_masks)

            detections.mask = np.array(masks).astype(bool)

        except Exception as e:
            logger.warning(f"SAM2 分割失败: {e}")
            detections.mask = None

        return detections

    def _save_annotations(self, detections, hit_prompts, dirs, img_file, class_names, img_shape):
        """保存各类标注"""
        h, w = img_shape

        if "yolo" in dirs:
            self.save_yolo_txt(detections, (h, w), dirs["yolo"] / f"{img_file.stem}.txt")

        if "labelme" in dirs:
            self.save_labelme_json(
                detections, img_file.name, class_names,
                dirs["labelme"] / f"{img_file.stem}.json", (h, w)
            )

        if "custom" in dirs:
            self.save_custom_json(
                detections, hit_prompts, img_file.name, class_names,
                dirs["custom"] / f"{img_file.stem}.json", (h, w)
            )

    def _save_empty_annotations(self, dirs, img_file, img_shape):
        """保存空标注"""
        h, w = img_shape
        empty_detections = sv.Detections.empty()

        if "yolo" in dirs:
            self.save_yolo_txt(empty_detections, (h, w), dirs["yolo"] / f"{img_file.stem}.txt")

        if "labelme" in dirs:
            self.save_labelme_json(
                empty_detections, img_file.name, [],
                dirs["labelme"] / f"{img_file.stem}.json", (h, w)
            )

    def _create_output_dirs(self, output_path, switches):
        """创建输出目录"""
        dirs = {}

        if switches.get("copy_images", True):
            dirs["images"] = output_path / "images"
        if switches.get("save_preview", True):
            dirs["preview"] = output_path / "previews"
        if switches.get("save_yolo", True):
            dirs["yolo"] = output_path / "labels_yolo"
        if switches.get("save_labelme", True):
            dirs["labelme"] = output_path / "labels_json"
        if switches.get("save_custom", True):
            dirs["custom"] = output_path / "labels_custom_json"

        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        return dirs

    def _get_image_files(self, input_path):
        """获取所有图像文件"""
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        return sorted(image_files)

    def _update_stats(self, detections, class_names):
        """更新统计信息"""
        self.stats['total_objects'] += len(detections)
        for class_id in detections.class_id:
            class_name = class_names[int(class_id)]
            self.stats['class_counts'][class_name] = self.stats['class_counts'].get(class_name, 0) + 1

    def _print_stats(self, start_time, failed_images):
        """打印统计信息"""
        total_time = time.time() - start_time
        self.stats['total_time'] = total_time

        logger.info("\n" + "=" * 60)
        logger.info("📊 处理统计")
        logger.info("=" * 60)
        logger.info(f"总耗时: {total_time:.2f} 秒")
        logger.info(f"总图像数: {self.stats['total_images']}")
        logger.info(f"总检测物体: {self.stats['total_objects']}")
        if self.stats['total_images'] > 0:
            logger.info(f"平均每张: {self.stats['total_objects'] / self.stats['total_images']:.1f} 个")

        if self.stats['class_counts']:
            logger.info("\n类别统计:")
            for class_name, count in sorted(self.stats['class_counts'].items(),
                                            key=lambda x: x[1], reverse=True):
                logger.info(f"  {class_name}: {count}")

        if self.stats['mapping_stats']:
            logger.info("\n类别映射统计:")
            for mapping, count in sorted(self.stats['mapping_stats'].items(),
                                         key=lambda x: x[1], reverse=True)[:20]:
                logger.info(f"  {mapping}: {count}")

        if failed_images:
            logger.warning(f"\n失败图像 ({len(failed_images)}):")
            for img in failed_images[:10]:
                logger.warning(f"  {img}")

    def _save_label_studio_tasks(self, ls_tasks, output_path):
        """保存 Label Studio 任务"""
        ls_path = output_path / "label_studio_import.json"
        with open(ls_path, "w") as f:
            json.dump(ls_tasks, f, indent=2)
        logger.info(f"Label Studio 任务已保存: {ls_path}")

    # =========================================================
    # 标注保存方法
    # =========================================================

    @staticmethod
    def save_yolo_txt(detections, img_shape, save_path):
        """保存 YOLO 格式标注"""
        h, w = img_shape
        lines = []
        if len(detections) > 0:
            for xyxy, cid in zip(detections.xyxy, detections.class_id):
                x1, y1, x2, y2 = xyxy
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                lines.append(f"{int(cid)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        save_path.write_text("\n".join(lines))

    @staticmethod
    def save_labelme_json(detections, image_name, class_names, save_path, img_shape):
        """保存 LabelMe JSON 格式"""
        shapes = []
        if len(detections) > 0 and detections.mask is not None:
            for i in range(len(detections.mask)):
                mask = detections.mask[i].astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
                for contour in contours:
                    if len(contour) < 3:
                        continue
                    points = contour.reshape(-1, 2).tolist()
                    shapes.append({
                        "label": class_names[int(detections.class_id[i])],
                        "points": points,
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {}
                    })

        data = {
            "version": "5.0.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": image_name,
            "imageData": None,
            "imageHeight": img_shape[0],
            "imageWidth": img_shape[1]
        }
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def save_custom_json(detections, hit_prompts, image_name, class_names, save_path, img_shape):
        """保存自定义 JSON 格式"""
        anns = []
        if len(detections) > 0:
            for i in range(len(detections)):
                mask_data = []
                if detections.mask is not None and i < len(detections.mask):
                    mask_data = detections.mask[i].astype(np.uint8).tolist()

                anns.append({
                    "class_id": int(detections.class_id[i]),
                    "class_name": class_names[int(detections.class_id[i])],
                    "prompt": hit_prompts[i] if i < len(hit_prompts) else "",
                    "confidence": float(detections.confidence[i]),
                    "bbox_xyxy": detections.xyxy[i].tolist(),
                    "mask": mask_data
                })

        out_dict = {
            "image": image_name,
            "imageHeight": img_shape[0],
            "imageWidth": img_shape[1],
            "annotations": anns
        }
        with open(save_path, "w") as f:
            json.dump(out_dict, f, indent=2)

    @staticmethod
    def format_ls_task(detections, image_name, class_names, img_shape):
        """格式化 Label Studio 任务"""
        h, w = img_shape
        results = []
        for i in range(len(detections.xyxy)):
            x, y, x2, y2 = detections.xyxy[i]
            results.append({
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "rectanglelabels": [class_names[int(detections.class_id[i])]],
                    "x": float(x / w * 100),
                    "y": float(y / h * 100),
                    "width": float((x2 - x) / w * 100),
                    "height": float((y2 - y) / h * 100),
                    "rotation": 0
                },
                "score": float(detections.confidence[i])
            })
        return {
            "data": {"image": f"/data/local-files/?d=images/{image_name}"},
            "predictions": [{"result": results}]
        }

    @staticmethod
    def save_preview(image, detections, save_path, class_names):
        """保存可视化预览"""
        mask_annotator = sv.MaskAnnotator()
        label_annotator = sv.LabelAnnotator()
        annotated = mask_annotator.annotate(scene=image.copy(), detections=detections)
        labels = [f"{class_names[int(cid)]} {c:.2f}"
                  for cid, c in zip(detections.class_id, detections.confidence)]
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
        cv2.imwrite(str(save_path), annotated)

    @staticmethod
    def generate_summary_report(output_dir, stats):
        """生成总结报告"""
        report_path = Path(output_dir) / "annotation_report.txt"
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("YOLOv26 + SAM2 自动标注报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"总图像数: {stats.get('total_images', 0)}\n")
            f.write(f"总检测物体: {stats.get('total_objects', 0)}\n")
            f.write(f"总耗时: {stats.get('total_time', 0):.2f} 秒\n\n")

            f.write("类别统计:\n")
            for class_name, count in stats.get('class_counts', {}).items():
                f.write(f"  {class_name}: {count}\n")

            f.write("\n类别映射统计:\n")
            for mapping, count in stats.get('mapping_stats', {}).items():
                f.write(f"  {mapping}: {count}\n")

        logger.info(f"报告已保存: {report_path}")


# =========================================================
# 命令行入口
# =========================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='YOLOv26 + SAM2 自动标注工具')
    parser.add_argument('--config', '-c', type=str, default="config/annotator.yaml",
                        help='配置文件路径')
    parser.add_argument('--input', '-i', type=str, default='labelinput/camera_color_image_raw_compressed',help='输入图像目录（覆盖配置文件）')
    parser.add_argument('--output', '-o', type=str, default='labeloutput/camera_color_image_raw_compressed',help='输出目录（覆盖配置文件）')
    parser.add_argument('--conf', type=float, default='0.2',help='置信度阈值（覆盖配置文件）')
    parser.add_argument('--device', type=str, default='cpu',choices=['cuda', 'cpu'], help='运行设备')

    args = parser.parse_args()

    # 加载配置
    cfg = OmegaConf.load(args.config)

    # 覆盖配置
    if args.input:
        cfg.input_dir = args.input
    if args.output:
        cfg.output_dir = args.output
    if args.conf:
        cfg.conf.detect = args.conf
        cfg.conf.save = args.conf

    # 初始化标注器
    device = args.device if args.device else None
    annotator = YOLOv26SAM2Annotator(
        cfg.yolo_weights,
        cfg.sam2_checkpoint,
        cfg.sam2_yaml,
        device=device
    )

    # 获取开关
    switches = cfg.get("output_switches", {})

    # 运行处理
    annotator.process(
        cfg.input_dir,
        cfg.output_dir,
        cfg.classes,
        cfg.conf,
        switches
    )

    # 生成报告
    annotator.generate_summary_report(cfg.output_dir, annotator.stats)