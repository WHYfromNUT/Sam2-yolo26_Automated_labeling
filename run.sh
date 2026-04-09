#!/bin/bash
# 一键运行：输入bag路径，输出标注结果，支持自定义参数

# 显示帮助
show_help() {
    echo "用法: ./run.sh --bag <bag_path> [选项]"
    echo ""
    echo "必选参数:"
    echo "  --bag, -b <path>           ROS2 bag 路径"
    echo ""
    echo "可选参数:"
    echo "  --output, -o <dir>         输出目录 (默认: /home/why/rezone_space/out)"
    echo "  --topic, -t <topic>        图像话题 (默认: /metoak/right_image)"
    echo "  --step, -s <seconds>       提取间隔秒数 (默认: 3.0)"
    echo "  --conf, -c <value>         置信度阈值 (默认: 0.25)"
    echo "  --device, -d <device>      运行设备 cuda/cpu (默认: cuda)"
    echo "  --help, -h                 显示帮助"
    echo ""
    echo "示例:"
    echo "  ./run.sh --bag /path/to/bag"
    echo "  ./run.sh -b /path/to/bag -o /my/output -t /camera/image -s 2.0 -c 0.3"
}

# 默认参数
BAG_PATH=""
OUTPUT_BASE="/home/why/rezone_space/out"
TOPIC="/metoak/right_image"
STEP=3.0
CONF=0.25
DEVICE="cuda"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --bag|-b)
            BAG_PATH="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --topic|-t)
            TOPIC="$2"
            shift 2
            ;;
        --step|-s)
            STEP="$2"
            shift 2
            ;;
        --conf|-c)
            CONF="$2"
            shift 2
            ;;
        --device|-d)
            DEVICE="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查必选参数
if [ -z "$BAG_PATH" ]; then
    echo "❌ 错误: 必须指定 --bag 参数"
    show_help
    exit 1
fi

# 检查bag是否存在
if [ ! -e "$BAG_PATH" ]; then
    echo "❌ 错误: Bag 路径不存在: $BAG_PATH"
    exit 1
fi

# 从bag路径提取名称
BAG_NAME=$(basename "$BAG_PATH" | sed 's/rosbag2_//' | sed 's/ /_/g')
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$OUTPUT_BASE/${BAG_NAME}_${TIMESTAMP}"

echo "=========================================="
echo "🚀 一键标注流水线"
echo "=========================================="
echo "📁 Bag 路径: $BAG_PATH"
echo "📸 图像话题: $TOPIC"
echo "⏱️  提取间隔: $STEP 秒"
echo "🎯 置信度阈值: $CONF"
echo "💻 运行设备: $DEVICE"
echo "📁 输出目录: $OUTPUT_DIR"
echo "=========================================="

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# ==========================================
# 步骤1: 提取图片 (ROS2环境)
# ==========================================
echo ""
echo "[1/2] 提取图片..."

# 退出 conda 环境
conda deactivate 2>/dev/null

# 加载 ROS2 环境
source /opt/ros/humble/setup.bash

# 验证 Python 版本
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "当前 Python 版本: $PYTHON_VERSION (ROS2需要 3.10)"

python3 extract_images_from_bag.py \
    --bag_path "$BAG_PATH" \
    --topics "$TOPIC" \
    --output_dir "$OUTPUT_DIR/extracted" \
    --step "$STEP"

if [ $? -ne 0 ]; then
    echo "❌ 提取失败"
    exit 1
fi

# 查找实际的图片目录
IMAGES_DIR=$(find "$OUTPUT_DIR/extracted" -type f -name "*.jpg" -o -name "*.png" | head -1 | xargs dirname 2>/dev/null)
if [ -z "$IMAGES_DIR" ]; then
    IMAGES_DIR="$OUTPUT_DIR/extracted"
fi

# ==========================================
# 步骤2: 标注图片 (PyTorch环境)
# ==========================================
echo ""
echo "[2/2] 标注图片..."

# 重新激活 conda 环境
source ~/Downloads/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

python3 Automated_labeling.py \
    --input "$IMAGES_DIR" \
    --output "$OUTPUT_DIR/annotated" \
    --conf "$CONF" \
    --device "$DEVICE"

if [ $? -ne 0 ]; then
    echo "❌ 标注失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ 完成！"
echo "📁 输出目录: $OUTPUT_DIR"
echo "   ├── extracted/     # 提取的原始图片"
echo "   └── annotated/     # 标注结果"
echo "=========================================="
