#!/usr/bin/env python3
"""
Grounding DINO video test.

Runs text-prompted detection on each video frame, optionally logs boxes to CSV,
and either displays the annotated stream or writes an annotated video.
"""

import argparse
import csv
import inspect
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, GroundingDinoForObjectDetection


def normalize_prompt(prompt: str) -> str:
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("text prompt is empty")
    if not prompt.endswith("."):
        prompt += "."
    return prompt


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def post_process(processor, outputs, input_ids, image_size, box_threshold, text_threshold):
    target_sizes = torch.tensor([image_size], device=outputs.logits.device)
    fn = processor.post_process_grounded_object_detection
    kwargs = {
        "outputs": outputs,
        "input_ids": input_ids,
        "target_sizes": target_sizes,
    }

    params = inspect.signature(fn).parameters
    if "threshold" in params:
        kwargs["threshold"] = box_threshold
    elif "box_threshold" in params:
        kwargs["box_threshold"] = box_threshold
    if "text_threshold" in params:
        kwargs["text_threshold"] = text_threshold

    return fn(**kwargs)[0]


def tensor_to_list(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    return value


def result_rows(result, frame_idx, timestamp):
    boxes = tensor_to_list(result.get("boxes", []))
    scores = tensor_to_list(result.get("scores", []))
    labels = result.get("text_labels", result.get("labels", []))

    rows = []
    for det_idx, box in enumerate(boxes):
        x0, y0, x1, y1 = [float(v) for v in box]
        rows.append(
            {
                "frame": frame_idx,
                "time": timestamp,
                "detection": det_idx,
                "label": labels[det_idx] if det_idx < len(labels) else "object",
                "score": scores[det_idx] if det_idx < len(scores) else 0.0,
                "x": x0,
                "y": y0,
                "dx": x1 - x0,
                "dy": y1 - y0,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
            }
        )
    return rows


def draw_detections(image_bgr, result):
    boxes = tensor_to_list(result.get("boxes", []))
    scores = tensor_to_list(result.get("scores", []))
    labels = result.get("text_labels", result.get("labels", []))

    out = image_bgr.copy()
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        x0 = max(0, min(out.shape[1] - 1, x0))
        x1 = max(0, min(out.shape[1] - 1, x1))
        y0 = max(0, min(out.shape[0] - 1, y0))
        y1 = max(0, min(out.shape[0] - 1, y1))

        score = scores[i] if i < len(scores) else 0.0
        label = labels[i] if i < len(labels) else "object"
        text = f"{label} {score:.2f}"

        color = (0, 220, 255)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_y0 = max(0, y0 - th - baseline - 4)
        cv2.rectangle(out, (x0, label_y0), (min(out.shape[1] - 1, x0 + tw + 6), y0), color, -1)
        cv2.putText(out, text, (x0 + 3, y0 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    return out


def open_writer(path, fps, width, height):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    ext = os.path.splitext(path)[1].lower()
    fourcc_name = "mp4v" if ext in [".mp4", ".m4v", ".mov"] else "MJPG"
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc_name), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {path}")
    return writer


def detect_frame(processor, model, frame_bgr, prompt, device, box_threshold, text_threshold):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    inputs = processor(images=pil_image, text=prompt, return_tensors="pt")
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    return post_process(
        processor,
        outputs,
        inputs.get("input_ids"),
        (pil_image.height, pil_image.width),
        box_threshold,
        text_threshold,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run Grounding DINO on every video frame.")
    parser.add_argument("--vid", required=True, help="Input video path.")
    parser.add_argument("--prompt", "-p", help='Text prompt, e.g. "drone racing gate. square colored foam gate."')
    parser.add_argument("--model", default="IDEA-Research/grounding-dino-tiny", help="Hugging Face model id or local model directory.")
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--device", default="auto", help='auto, cpu, cuda, or cuda:0. Default: "auto".')
    parser.add_argument("--csv", help="Optional CSV path for detections.")
    parser.add_argument("--save", help="Optional annotated video path. If set, GUI rendering is disabled.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional positive cap on processed frames.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.vid):
        print(f"Video does not exist: {args.vid}", file=sys.stderr)
        return 2

    prompt = normalize_prompt(args.prompt or input("Text prompt: "))
    device = pick_device(args.device)

    cap = cv2.VideoCapture(args.vid)
    if not cap.isOpened():
        print(f"Failed to open video: {args.vid}", file=sys.stderr)
        return 2

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames > 0:
        frame_count = min(frame_count, args.max_frames) if frame_count > 0 else args.max_frames

    print(f"Loading model: {args.model}")
    print(f"Device: {device}")
    processor = AutoProcessor.from_pretrained(args.model)
    model = GroundingDinoForObjectDetection.from_pretrained(args.model).to(device)
    model.eval()

    writer = open_writer(args.save, fps, width, height) if args.save else None
    csv_file = None
    csv_writer = None
    if args.csv:
        parent = os.path.dirname(os.path.abspath(args.csv))
        if parent:
            os.makedirs(parent, exist_ok=True)
        csv_file = open(args.csv, "w", newline="")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=["frame", "time", "detection", "label", "score", "x", "y", "dx", "dy", "x0", "y0", "x1", "y1"],
        )
        csv_writer.writeheader()

    frame_idx = 0
    progress_total = frame_count if frame_count > 0 else None
    try:
        with tqdm(total=progress_total, unit="frame") as pbar:
            while True:
                if args.max_frames > 0 and frame_idx >= args.max_frames:
                    break

                ok, frame = cap.read()
                if not ok:
                    break

                timestamp = frame_idx / fps
                result = detect_frame(processor, model, frame, prompt, device, args.box_threshold, args.text_threshold)

                if csv_writer is not None:
                    for row in result_rows(result, frame_idx, timestamp):
                        csv_writer.writerow(row)

                vis = draw_detections(frame, result)
                if writer is not None:
                    writer.write(vis)
                else:
                    cv2.imshow("Grounding DINO video detections", vis)
                    if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
                        break

                frame_idx += 1
                pbar.update(1)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if csv_file is not None:
            csv_file.close()
        if writer is None:
            cv2.destroyAllWindows()

    print(f"Processed frames: {frame_idx}")
    if args.save:
        print(f"Saved video: {args.save}")
    if args.csv:
        print(f"Saved CSV: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
