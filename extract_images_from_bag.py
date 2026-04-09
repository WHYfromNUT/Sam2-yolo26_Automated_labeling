
# !/usr/bin/env python3
"""
@Author: Xinlu Wang/王心禄 (Rezone CN) xinlu.wang@rezone-tec.com
@Date: 2026-02-06
@File: extract_images_from_bag.py
@Brief: Extract images from ROS 2 bags with throttling.
@Description: Safely extracts images from a ROS 2 bag file (dir or db3) to local disk.
              Supports multiple image topics, configurable throttling (e.g., 1 Hz),
              auto-sanitizes filenames, and generates an index CSV.
              Designed for headless execution without CLI arguments.
              NOTE: Does NOT use cv_bridge to avoid NumPy 2.x conflicts.
"""

import sys
import os
import csv
import time
import argparse
from collections import defaultdict

# ROS 2 Dependencies
try:
    import rclpy.serialization
    import rosbag2_py
    import sensor_msgs.msg
    from rclpy.time import Time
    from rosidl_runtime_py.utilities import get_message
except ImportError as e:
    print(f"Error: Missing ROS 2 Python dependencies. {e}")
    print("Please ensure you have sourced your ROS 2 environment (e.g., source /opt/ros/humble/setup.bash).")
    print("Required: rosbag2_py, rclpy, sensor_msgs")
    sys.exit(1)

# OpenCV
try:
    import cv2
    import numpy as np
except ImportError:
    print("Error: Missing opencv-python. Install via: pip install opencv-python")
    sys.exit(1)

# ==============================================================================
# CONFIGURATION AREA (Default values, can be overridden by command line)
# ==============================================================================
# Path to the bag directory OR the .db3 file directly
BAG_PATH = "/home/why/rezone_space/bags/rosbag2_2026_03_10-10_46_58_停车场"

# List of image topics to extract
IMAGE_TOPICS = [
    "/camera/color/image_raw/compressed"
]

# Output directory for frames and indices
OUTPUT_DIR = "/home/why/rezone_space/bags/output_images_rosbag2_2026_03_10-10_46_58_停车场"

# Output format: "png" or "jpg"
OUTPUT_FORMAT = "jpg"

# JPEG Quality (0-100). 100 is best quality (least compression). Default is 95.
JPEG_QUALITY = 100

# PNG Compression (0-9). 0 is no compression (fastest, largest). 3 is default.
PNG_COMPRESSION = 3

# Storage ID: "sqlite3", "mcap", or "" (auto-detect from metadata.yaml or extension)
BAG_STORAGE_ID = ""

# Minimum time interval between saved frames per topic (in seconds)
STEP_SECONDS = 3.0

# If True, use message header stamp. If False or missing, use bag recording timestamp.
USE_HEADER_STAMP = True

# If True, create a subdirectory for each topic (OUTPUT_DIR/<topic_name>/...).
# If False, put all images in OUTPUT_DIR with topic prefix in filename.
PER_TOPIC_SUBDIR = True

# If True, exit with error if ANY topic in IMAGE_TOPICS is missing from the bag.
# If False, confirm missing topics and continue with the rest.
STRICT_TOPICS = True


# ==============================================================================


def sanitize_topic_name(topic_name):
    """Replaces slashes with underscores and removes leading underscore."""
    sanitized = topic_name.replace('/', '_')
    if sanitized.startswith('_'):
        sanitized = sanitized[1:]
    return sanitized


def get_storage_id(path):
    """
    Attempts to detect storage_id from config, file extension, or metadata.yaml.
    """
    if BAG_STORAGE_ID:
        return BAG_STORAGE_ID
    if path.endswith('.mcap'):
        return 'mcap'
    if path.endswith('.db3'):
        return 'sqlite3'
    if os.path.isdir(path):
        meta_path = os.path.join(path, "metadata.yaml")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    for line in f:
                        if 'storage_identifier:' in line:
                            parts = line.split(':')
                            if len(parts) > 1:
                                return parts[1].strip()
            except Exception as e:
                print(f"Warning: Failed to parse metadata.yaml: {e}")
    return 'sqlite3'


def get_rosbag_options(path):
    storage_id = get_storage_id(path)
    print(f"Detected Storage ID: {storage_id}")
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    return storage_options, converter_options


def get_message_type(type_name):
    try:
        return get_message(type_name)
    except (ValueError, ImportError, AttributeError):
        return None


def run_extraction(bag_path, image_topics, output_dir, step_seconds, output_format,
                   jpeg_quality, png_compression, use_header_stamp, per_topic_subdir, strict_topics):
    """执行提取的主函数"""
    print("==================================================")
    print("       ROS 2 Bag Image Extractor (Headless)       ")
    print("==================================================")
    print(f"Bag Path:       {bag_path}")
    print(f"Image Topics:   {image_topics}")
    print(f"Output Dir:     {output_dir}")
    print(f"Step Seconds:   {step_seconds}")
    print(f"Per Topic Dir:  {per_topic_subdir}")
    print(f"Strict Mode:    {strict_topics}")
    print("==================================================")

    if not os.path.exists(bag_path):
        print(f"Error: Bag path does not exist: {bag_path}")
        return 1

    try:
        storage_options, converter_options = get_rosbag_options(bag_path)
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"Error opening bag: {e}")
        return 1

    all_topics_types = reader.get_all_topics_and_types()
    bag_topic_dict = {t.name: t.type for t in all_topics_types}

    valid_topics = []
    missing_topics = []

    for topic in image_topics:
        if topic in bag_topic_dict:
            valid_topics.append(topic)
        else:
            missing_topics.append(topic)

    if missing_topics:
        msg = f"Missing topics in bag: {missing_topics}"
        if strict_topics:
            print(f"Error: {msg}")
            return 1
        else:
            print(f"Warning: {msg}. Continuing with valid topics.")

    if not valid_topics:
        print("Error: No valid image topics found to extract.")
        return 1

    storage_filter = rosbag2_py.StorageFilter(topics=valid_topics)
    reader.set_filter(storage_filter)

    os.makedirs(output_dir, exist_ok=True)

    last_saved_ns_by_topic = {}
    stats = defaultdict(lambda: {"total": 0, "saved": 0, "skipped": 0, "failed": 0})

    csv_files = {}
    csv_writers = {}
    csv_header = ['seq', 'topic', 'stamp_sec', 'stamp_nanosec', 'frame_id', 'filename']

    if not per_topic_subdir:
        f = open(os.path.join(output_dir, 'index.csv'), 'w', newline='')
        w = csv.writer(f)
        w.writerow(csv_header)
        csv_files['__global__'] = f
        csv_writers['__global__'] = w
    else:
        for topic in valid_topics:
            topic_clean = sanitize_topic_name(topic)
            topic_dir = os.path.join(output_dir, topic_clean)
            os.makedirs(topic_dir, exist_ok=True)
            f = open(os.path.join(topic_dir, 'index.csv'), 'w', newline='')
            w = csv.writer(f)
            w.writerow(csv_header)
            csv_files[topic] = f
            csv_writers[topic] = w

    print("Starting extraction...")

    topic_type_class = {}
    for t_name, t_type in bag_topic_dict.items():
        if t_name in valid_topics:
            cls = get_message_type(t_type)
            if cls:
                topic_type_class[t_name] = cls

    step_ns = int(step_seconds * 1e9)

    while reader.has_next():
        (topic, data, t_bag_ns) = reader.read_next()

        if topic not in valid_topics:
            continue

        stats[topic]["total"] += 1

        msg_type = topic_type_class.get(topic)
        if not msg_type:
            stats[topic]["failed"] += 1
            continue

        try:
            msg = rclpy.serialization.deserialize_message(data, msg_type)
        except Exception as e:
            stats[topic]["failed"] += 1
            continue

        t_ns = t_bag_ns
        if use_header_stamp and hasattr(msg, 'header'):
            h_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            if h_ns > 0:
                t_ns = h_ns

        last_ns = last_saved_ns_by_topic.get(topic)
        if last_ns is not None and (t_ns - last_ns) < step_ns:
            stats[topic]["skipped"] += 1
            continue

        cv_img = None
        try:
            msg_type_str = str(type(msg))
            if 'CompressedImage' in msg_type_str:
                is_compressed_depth = 'compressedDepth' in msg.format or 'depth' in topic.lower()

                np_arr = None
                if is_compressed_depth and len(msg.data) > 12:
                    try:
                        data_bytes = bytes(msg.data)
                        png_idx = data_bytes.find(b'\x89PNG')
                        if png_idx != -1:
                            np_arr = np.frombuffer(data_bytes[png_idx:], np.uint8)
                        else:
                            np_arr = np.frombuffer(msg.data, np.uint8)
                    except Exception:
                        np_arr = np.frombuffer(msg.data, np.uint8)
                else:
                    np_arr = np.frombuffer(msg.data, np.uint8)

                cv_img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

            elif 'Image' in msg_type_str:
                encoding = getattr(msg, 'encoding', 'unknown')
                height = getattr(msg, 'height', 0)
                width = getattr(msg, 'width', 0)

                if encoding == 'bgr8':
                    cv_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width, 3)
                elif encoding == 'rgb8':
                    cv_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width, 3)
                    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
                elif encoding == 'mono8':
                    cv_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width)
                else:
                    if len(msg.data) == height * width * 3:
                        cv_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width, 3)
                    elif len(msg.data) == height * width:
                        cv_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width)
                    else:
                        stats[topic]["failed"] += 1
                        continue
            else:
                stats[topic]["failed"] += 1
                continue

        except Exception as e:
            stats[topic]["failed"] += 1
            continue

        if cv_img is None:
            stats[topic]["failed"] += 1
            continue

        current_seq = stats[topic]["saved"] + 1
        topic_clean = sanitize_topic_name(topic)

        if per_topic_subdir:
            filename = f"{current_seq:05d}.{output_format}"
            save_path = os.path.join(output_dir, topic_clean, filename)
        else:
            filename = f"{topic_clean}_{current_seq:05d}.{output_format}"
            save_path = os.path.join(output_dir, filename)

        try:
            write_params = []
            if output_format in ["jpg", "jpeg"]:
                write_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            elif output_format == "png":
                write_params = [cv2.IMWRITE_PNG_COMPRESSION, png_compression]

            success = cv2.imwrite(save_path, cv_img, write_params)
            if not success:
                stats[topic]["failed"] += 1
                continue
        except Exception as e:
            stats[topic]["failed"] += 1
            continue

        last_saved_ns_by_topic[topic] = t_ns
        stats[topic]["saved"] += 1

        frame_id = msg.header.frame_id if hasattr(msg, 'header') else ""
        sec = t_ns // 1_000_000_000
        nsec = t_ns % 1_000_000_000
        row = [0, topic, sec, nsec, frame_id, filename]

        if per_topic_subdir:
            csv_writers[topic].writerow(row)
        else:
            csv_writers['__global__'].writerow(row)

    for f in csv_files.values():
        f.close()

    print("\nProcessing Complete.")
    print("==================================================")
    print("Statistics per Topic:")
    print(f"{'Topic':<30} | {'Total':<8} | {'Saved':<8} | {'Skipped':<8} | {'Failed':<8}")
    print("-" * 75)
    for topic in valid_topics:
        s = stats[topic]
        print(f"{topic:<30} | {s['total']:<8} | {s['saved']:<8} | {s['skipped']:<8} | {s['failed']:<8}")
    print("==================================================")
    print(f"Images saved to: {os.path.abspath(output_dir)}")

    return 0


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Extract images from ROS2 bag')
    parser.add_argument('--bag_path', '-b', type=str, help='ROS2 bag path')
    parser.add_argument('--topics', '-t', nargs='+', help='Image topics')
    parser.add_argument('--output_dir', '-o', type=str, help='Output directory')
    parser.add_argument('--step', '-s', type=float, help='Step seconds between frames')
    parser.add_argument('--format', '-f', type=str, choices=['jpg', 'png'], help='Output format')
    parser.add_argument('--jpeg_quality', type=int, help='JPEG quality')
    parser.add_argument('--no_header_stamp', action='store_false', dest='use_header_stamp', help='Use bag timestamp')
    parser.add_argument('--flat', action='store_false', dest='per_topic_subdir', help='Flat output structure')
    parser.add_argument('--strict', action='store_true', help='Strict topic checking')

    return parser.parse_args()


def main():
    args = parse_args()

    # 使用命令行参数覆盖默认配置
    bag_path = args.bag_path if args.bag_path else BAG_PATH
    image_topics = args.topics if args.topics else IMAGE_TOPICS
    output_dir = args.output_dir if args.output_dir else OUTPUT_DIR
    step_seconds = args.step if args.step is not None else STEP_SECONDS
    output_format = args.format if args.format else OUTPUT_FORMAT
    jpeg_quality = args.jpeg_quality if args.jpeg_quality is not None else JPEG_QUALITY
    use_header_stamp = args.use_header_stamp if args.use_header_stamp is not None else USE_HEADER_STAMP
    per_topic_subdir = args.per_topic_subdir if args.per_topic_subdir is not None else PER_TOPIC_SUBDIR
    strict_topics = args.strict if args.strict is not None else STRICT_TOPICS

    sys.exit(run_extraction(
        bag_path=bag_path,
        image_topics=image_topics,
        output_dir=output_dir,
        step_seconds=step_seconds,
        output_format=output_format,
        jpeg_quality=jpeg_quality,
        png_compression=PNG_COMPRESSION,
        use_header_stamp=use_header_stamp,
        per_topic_subdir=per_topic_subdir,
        strict_topics=strict_topics
    ))


if __name__ == "__main__":
    main()