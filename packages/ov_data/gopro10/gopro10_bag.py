#!/usr/bin/env python3
"""Convert a GoPro10 MP4 with embedded GPMF IMU telemetry into a ROS bag.

Usage:
  gopro10_bag.py <mp4> <bag> [resize] [compress]
  gopro10_bag.py <mp4> <bag> --resize 640 --compress

Examples:
  gopro10_bag.py flight.MP4 flight.bag
  gopro10_bag.py flight.MP4 flight_640.bag 640x480
  gopro10_bag.py flight.MP4 flight_half.bag 0.5 true
  gopro10_bag.py flight.MP4 flight_lz4.bag true
"""

import argparse
import os
import sys

import cv2 as cv
import numpy as np
import rospy
from geometry_msgs.msg import Vector3
from py_gpmf_parser.gopro_telemetry_extractor import GoProTelemetryExtractor
from rosbag import Bag, Compression
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import Header
from tqdm import tqdm


CAM_TOPIC = "/cam0/image_raw"
IMU_TOPIC = "/imu0"
CAM_FRAME_ID = "cam0"
IMU_FRAME_ID = "imu0"


def parse_bool(text):
    if isinstance(text, bool):
        return text
    value = str(text).strip().lower()
    if value in ("1", "true", "yes", "y", "on", "compress", "compressed", "lz4"):
        return True
    if value in ("0", "false", "no", "n", "off", "none", "raw", "uncompressed"):
        return False
    raise ValueError(f"not a boolean value: {text}")


def looks_like_bool(text):
    try:
        parse_bool(text)
        return True
    except ValueError:
        return False


def parse_resize(text):
    if text is None:
        return 640
    value = str(text).strip().lower()
    if value in ("", "none", "no", "false", "0"):
        return None
    if "x" in value:
        width_text, height_text = value.split("x", 1)
        width = int(width_text)
        height = int(height_text)
        if width <= 0 or height <= 0:
            raise ValueError("resize dimensions must be positive")
        return width, height
    if any(c in value for c in ".e"):
        scale = float(value)
        if scale <= 0.0:
            raise ValueError("resize scale must be positive")
        return scale
    width = int(value)
    if width <= 0:
        raise ValueError("resize width must be positive")
    return width


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mp4", help="GoPro10 MP4 file with embedded GPMF telemetry")
    parser.add_argument("bag", help="output ROS bag")
    parser.add_argument("resize_pos", nargs="?", default=None,
                        help='resize: float scale, int output width preserving aspect, or "WxH"; default is 640')
    parser.add_argument("compress_pos", nargs="?", default=None, help="optional bag compression boolean; true enables LZ4")
    parser.add_argument("--resize", dest="resize_flag", default=None,
                        help='resize: float scale, int output width preserving aspect, or "WxH"; default is 640')
    parser.add_argument("--compress", dest="compress_flag", action="store_true", help="enable LZ4 bag compression")
    parser.add_argument("--no-compress", dest="compress_false_flag", action="store_true", help="disable bag compression")
    args = parser.parse_args(argv)

    # Allow: gopro10_bag.py in.mp4 out.bag true
    if args.compress_pos is None and args.resize_pos is not None and looks_like_bool(args.resize_pos):
        args.compress_pos = args.resize_pos
        args.resize_pos = None

    args.resize = parse_resize(args.resize_flag if args.resize_flag is not None else args.resize_pos)
    if args.compress_flag and args.compress_false_flag:
        parser.error("--compress and --no-compress are mutually exclusive")
    if args.compress_flag:
        args.compress = True
    elif args.compress_false_flag:
        args.compress = False
    else:
        args.compress = False if args.compress_pos is None else parse_bool(args.compress_pos)
    return args


def stamp_from_seconds(timestamp):
    return rospy.Time.from_sec(float(timestamp))


def image_msg_from_frame(frame_bgr, timestamp):
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("expected BGR image with 3 channels")
    height, width = frame_bgr.shape[:2]
    msg = Image()
    msg.header = Header(stamp=stamp_from_seconds(timestamp), frame_id=CAM_FRAME_ID)
    msg.height = int(height)
    msg.width = int(width)
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = int(width * 3)
    msg.data = frame_bgr.tobytes()
    return msg


def imu_msg(timestamp, accl, gyro):
    msg = Imu()
    msg.header = Header(stamp=stamp_from_seconds(timestamp), frame_id=IMU_FRAME_ID)
    msg.orientation_covariance[0] = -1.0
    msg.linear_acceleration = Vector3(float(accl[0]), float(accl[1]), float(accl[2]))
    msg.angular_velocity = Vector3(float(gyro[0]), float(gyro[1]), float(gyro[2]))
    return msg


def resize_frame(frame, resize):
    if resize is None:
        return frame
    if isinstance(resize, tuple):
        return cv.resize(frame, resize, interpolation=cv.INTER_AREA)
    if isinstance(resize, int):
        height, width = frame.shape[:2]
        new_width = resize
        new_height = max(1, int(round(height * (new_width / float(width)))))
        return cv.resize(frame, (new_width, new_height), interpolation=cv.INTER_AREA)
    return cv.resize(frame, None, fx=resize, fy=resize, interpolation=cv.INTER_AREA)


def extract_telemetry(mp4_path):
    extractor = GoProTelemetryExtractor(mp4_path)
    extractor.open_source()
    try:
        accl, accl_t = extractor.extract_data("ACCL")
        gyro, gyro_t = extractor.extract_data("GYRO")
        image_t = extractor.get_image_timestamps_s()
    finally:
        extractor.close_source()

    if len(accl) == 0 or len(gyro) == 0:
        raise RuntimeError("failed to extract ACCL/GYRO telemetry from MP4")
    if len(image_t) == 0:
        raise RuntimeError("failed to extract video frame timestamps from MP4")

    time_offset = min(float(accl_t[0]), float(gyro_t[0]), float(image_t[0]))
    accl_t = np.asarray(accl_t, dtype=float) - time_offset
    gyro_t = np.asarray(gyro_t, dtype=float) - time_offset
    image_t = np.asarray(image_t, dtype=float) - float(image_t[0])
    accl = np.asarray(accl, dtype=float)
    gyro = np.asarray(gyro, dtype=float)

    # Use accelerometer timestamps as the IMU output grid and interpolate gyro
    # onto that grid. GoPro GPMF commonly gives matching sample counts, but this
    # keeps the bag writer robust when they differ by a few samples.
    gyro_interp = np.empty((len(accl_t), 3), dtype=float)
    for axis in range(3):
        gyro_interp[:, axis] = np.interp(accl_t, gyro_t, gyro[:, axis])
    return accl, accl_t, gyro_interp, image_t


def write_imu(bag, accl, accl_t, gyro):
    total = len(accl_t)
    for i, timestamp in enumerate(tqdm(accl_t, total=total, desc="imu", unit="sample")):
        bag.write(IMU_TOPIC, imu_msg(timestamp, accl[i], gyro[i]), t=stamp_from_seconds(timestamp))


def write_video(bag, mp4_path, image_t, resize):
    cap = cv.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open MP4 video stream: {mp4_path}")

    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = len(image_t)
    total = min(total, len(image_t))

    frame_index = 0
    try:
        with tqdm(total=total, desc="video", unit="frame") as pbar:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if frame_index >= len(image_t):
                    break
                frame = resize_frame(frame, resize)
                timestamp = float(image_t[frame_index])
                bag.write(CAM_TOPIC, image_msg_from_frame(frame, timestamp), t=stamp_from_seconds(timestamp))
                frame_index += 1
                pbar.update(1)
    finally:
        cap.release()

    print(f"[bag] wrote images done: {frame_index}")


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if not os.path.exists(args.mp4):
        raise FileNotFoundError(args.mp4)

    print(f"[gopro10_bag] mp4={args.mp4}")
    print(f"[gopro10_bag] bag={args.bag}")
    print(f"[gopro10_bag] resize={args.resize}")
    print(f"[gopro10_bag] compression={'lz4' if args.compress else 'none'}")

    accl, accl_t, gyro, image_t = extract_telemetry(args.mp4)
    print(f"[gopro10_bag] telemetry ACCL={len(accl)} GYRO(interp)={len(gyro)} images={len(image_t)}")

    compression = Compression.LZ4 if args.compress else Compression.NONE
    with Bag(args.bag, "w", compression=compression) as bag:
        write_imu(bag, accl, accl_t, gyro)
        write_video(bag, args.mp4, image_t, args.resize)

    print(f"[gopro10_bag] done: {args.bag}")


if __name__ == "__main__":
    main()
