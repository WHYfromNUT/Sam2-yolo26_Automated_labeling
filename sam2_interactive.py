import cv2
import torch
import numpy as np
from pathlib import Path
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from omegaconf import OmegaConf
from hydra.utils import instantiate
from sam2.sam2_image_predictor import SAM2ImagePredictor


class SAM2InteractiveGUI:
    def __init__(self, config_path="annotator.yaml"):
        """初始化 SAM2 交互式分割器（matplotlib GUI）"""
        # 加载配置
        self.cfg = OmegaConf.load(config_path)

        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 使用设备: {self.device}")

        # 加载 SAM2 模型
        print(f"📦 加载 SAM2 模型...")
        print(f"   检查点: {self.cfg.sam2_checkpoint}")

        start_time = time.time()

        # 加载配置
        full_config = OmegaConf.load(self.cfg.sam2_yaml)
        sam_model = instantiate(full_config.model)

        # 加载权重
        checkpoint = torch.load(self.cfg.sam2_checkpoint, map_location="cpu")
        sam_model.load_state_dict(checkpoint["model"], strict=False)
        sam_model.to(self.device).eval()

        # 创建预测器
        self.predictor = SAM2ImagePredictor(sam_model)

        load_time = time.time() - start_time
        print(f"✅ SAM2 模型加载成功！耗时: {load_time:.2f} 秒")

        # 交互状态
        self.image = None
        self.image_rgb = None
        self.points = []  # 存储点击的点
        self.labels = []  # 存储点的标签（1=前景，0=背景）
        self.current_mask = None
        self.box_start = None
        self.box_mode = False
        self.fig = None
        self.ax = None

    def load_image(self, image_path):
        """加载图像"""
        self.image = cv2.imread(image_path)
        if self.image is None:
            print(f"❌ 无法读取图像: {image_path}")
            return False

        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(self.image_rgb)
        self.clear_points()

        h, w = self.image.shape[:2]
        print(f"✅ 图像加载成功: {Path(image_path).name}, 尺寸: {h} x {w}")
        return True

    def clear_points(self):
        """清除所有点"""
        self.points = []
        self.labels = []
        self.current_mask = None
        self.box_mode = False
        self.box_start = None

    def segment_with_points(self):
        """使用点提示进行分割"""
        if not self.points:
            print("⚠️  请先点击图像添加提示点")
            return None

        points = np.array(self.points)
        labels = np.array(self.labels)

        print(f"🔍 执行分割，使用 {len(points)} 个点...")
        start_time = time.time()

        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True
        )

        inference_time = time.time() - start_time
        print(f"✨ 分割完成！耗时: {inference_time:.3f} 秒")
        print(f"📊 生成 {len(masks)} 个掩码，得分: {scores}")

        # 选择得分最高的掩码
        best_idx = np.argmax(scores)
        self.current_mask = masks[best_idx].astype(bool)  # 确保是布尔类型

        print(f"✅ 选择掩码 {best_idx + 1}, 得分: {scores[best_idx]:.3f}")
        print(f"📊 掩码面积: {np.sum(self.current_mask)} 像素")

        return self.current_mask

    def segment_with_box(self, box):
        """使用边界框进行分割"""
        box = np.array(box)

        print(f"🔍 使用边界框分割: {box}")
        start_time = time.time()

        masks, scores, logits = self.predictor.predict(
            box=box,
            multimask_output=False
        )

        inference_time = time.time() - start_time
        print(f"✨ 分割完成！耗时: {inference_time:.3f} 秒")
        print(f"📊 得分: {scores[0]:.3f}")

        self.current_mask = masks[0].astype(bool)
        return self.current_mask

    def save_results(self, output_prefix="sam2_result"):
        """保存分割结果"""
        if self.current_mask is None:
            print("⚠️  没有分割结果可保存")
            return

        # 保存掩码
        mask_path = f"{output_prefix}_mask.png"
        mask_img = (self.current_mask * 255).astype(np.uint8)
        cv2.imwrite(mask_path, mask_img)
        print(f"💾 掩码已保存: {mask_path}")

        # 保存叠加图像
        if self.image is not None:
            overlay = self.image.copy()
            overlay[self.current_mask] = [0, 255, 0]
            overlay_path = f"{output_prefix}_overlay.jpg"
            cv2.imwrite(overlay_path, overlay)
            print(f"💾 叠加图像已保存: {overlay_path}")

        # 保存为 JSON
        import json
        info = {
            "image": self.image_path if hasattr(self, 'image_path') else "unknown",
            "mask_shape": list(self.current_mask.shape),
            "mask_area": int(np.sum(self.current_mask)),
            "num_points": len(self.points),
            "points": self.points,
            "labels": self.labels
        }
        json_path = f"{output_prefix}_info.json"
        with open(json_path, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"💾 信息已保存: {json_path}")

    def on_click(self, event):
        """鼠标点击事件处理"""
        if event.inaxes != self.ax:
            return

        x, y = int(event.xdata), int(event.ydata)

        if self.box_mode:
            # 框选模式
            if event.button == 1:  # 左键
                if self.box_start is None:
                    # 开始框选
                    self.box_start = (x, y)
                    print(f"📦 框选起点: ({x}, {y})")
                else:
                    # 结束框选
                    x1, y1 = self.box_start
                    x2, y2 = x, y
                    box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                    print(f"📦 框选区域: ({box[0]}, {box[1]}) -> ({box[2]}, {box[3]})")
                    self.segment_with_box(box)
                    self.box_start = None
                    self.box_mode = False
                    self.update_display()
        else:
            # 点模式
            if event.button == 1:  # 左键 - 前景点
                self.points.append([x, y])
                self.labels.append(1)
                print(f"➕ 添加前景点: ({x}, {y})")
                self.update_display()
            elif event.button == 3:  # 右键 - 背景点
                self.points.append([x, y])
                self.labels.append(0)
                print(f"➖ 添加背景点: ({x}, {y})")
                self.update_display()

    def on_key(self, event):
        """键盘事件处理"""
        if event.key == ' ':
            # 空格键执行分割
            print("\n" + "=" * 50)
            self.segment_with_points()
            self.update_display()
        elif event.key == 'u':
            # U 键撤销
            if self.points:
                self.points.pop()
                self.labels.pop()
                print(f"↩️  撤销最后一个点，剩余 {len(self.points)} 个点")
                if self.points:
                    self.segment_with_points()
                else:
                    self.current_mask = None
                self.update_display()
        elif event.key == 'c':
            # C 键清除
            self.clear_points()
            print("🗑️  清除所有点")
            self.update_display()
        elif event.key == 'b':
            # B 键切换框选模式
            self.box_mode = not self.box_mode
            self.box_start = None
            print(f"📦 框选模式: {'开启' if self.box_mode else '关闭'}")
            self.update_display()
        elif event.key == 's':
            # S 键保存
            self.save_results("sam2_result")
        elif event.key == 'q':
            # Q 键退出
            print("\n退出程序...")
            plt.close()

    def update_display(self):
        """更新显示"""
        if self.ax is None:
            return

        # 清除当前图像
        self.ax.clear()

        # 显示原始图像
        self.ax.imshow(self.image_rgb)

        # 绘制提示点
        for point, label in zip(self.points, self.labels):
            color = 'green' if label == 1 else 'red'
            marker = 'o' if label == 1 else 'x'
            self.ax.plot(point[0], point[1], marker, color=color, markersize=12,
                         markeredgewidth=2, markerfacecolor=color if label == 1 else 'none')

        # 绘制掩码
        if self.current_mask is not None and self.current_mask.any():
            # 创建掩码叠加（确保掩码是布尔类型）
            mask_bool = self.current_mask.astype(bool)

            # 创建绿色半透明叠加
            overlay = np.zeros((*self.image_rgb.shape[:2], 4))
            overlay[mask_bool] = [0, 1, 0, 0.3]  # 绿色半透明
            self.ax.imshow(overlay)

            # 绘制掩码轮廓
            contours = self.get_mask_contours(mask_bool)
            for contour in contours:
                contour_points = np.array(contour).reshape(-1, 2)
                if len(contour_points) > 2:
                    self.ax.plot(contour_points[:, 0], contour_points[:, 1],
                                 'g-', linewidth=2)

        # 绘制框选
        if self.box_mode and self.box_start:
            x, y = self.box_start
            rect = Rectangle((x, y), 0, 0, linewidth=2, edgecolor='blue', facecolor='none')
            self.ax.add_patch(rect)

        # 设置标题
        title = f"SAM2 Interactive Segmentation"
        if self.points:
            title += f"\nPoints: {len(self.points)}"
        if self.current_mask is not None and self.current_mask.any():
            area = np.sum(self.current_mask)
            title += f" | Mask Area: {area}"
        self.ax.set_title(title, fontsize=12)
        self.ax.axis('off')

        # 刷新显示
        self.fig.canvas.draw()

    def get_mask_contours(self, mask):
        """获取掩码的轮廓"""
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def run_interactive(self, image_path):
        """运行交互式分割"""
        if not self.load_image(image_path):
            return

        # 创建图形
        self.fig, self.ax = plt.subplots(1, 1, figsize=(14, 10))

        # 显示图像
        self.ax.imshow(self.image_rgb)
        self.ax.axis('off')

        # 设置标题
        title = (
            "SAM2 Interactive Segmentation\n"
            "Left Click: Foreground | Right Click: Background | "
            "Space: Segment | U: Undo | C: Clear | B: Box Mode | S: Save | Q: Quit"
        )
        self.ax.set_title(title, fontsize=10, pad=10)

        # 连接事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # 显示操作说明
        print("\n" + "=" * 70)
        print("🎨 SAM2 交互式分割 (Matplotlib GUI)")
        print("=" * 70)
        print("操作说明:")
        print("  🖱️  左键点击: 添加前景点（要分割的物体）")
        print("  🖱️  右键点击: 添加背景点（不是物体的部分）")
        print("  ⌨️  空格键: 执行分割")
        print("  ⌨️  U 键: 撤销最后一个点")
        print("  ⌨️  C 键: 清除所有点")
        print("  ⌨️  B 键: 切换到框选模式")
        print("  ⌨️  S 键: 保存分割结果")
        print("  ⌨️  Q 键: 退出")
        print("=" * 70)
        print("\n💡 提示: 点击图像上的点，然后按空格键查看分割效果")
        print("💡 可以添加多个点来提高分割精度")
        print("💡 按 B 键切换到框选模式，画框分割")

        plt.tight_layout()
        plt.show()


# 主程序
if __name__ == "__main__":
    print("=" * 70)
    print("SAM2 交互式分割测试")
    print("=" * 70)

    # 初始化 GUI
    gui = SAM2InteractiveGUI("config/annotator.yaml")

    # 运行交互
    image_path = "labelinput/image.png"
    if Path(image_path).exists():
        gui.run_interactive(image_path)
    else:
        print(f"❌ 图像不存在: {image_path}")
        print("请确保图像路径正确")