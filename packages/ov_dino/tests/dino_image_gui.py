#!/usr/bin/env python3
"""
Interactive Grounding DINO image test.

Prompts for an image path and text description, runs open-vocabulary detection,
and shows labeled bounding boxes with OpenCV.
"""

import argparse
import inspect
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, GroundingDinoForObjectDetection


def ask_image_path() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        if path:
            return path
    except Exception:
        pass

    return input("Image path: ").strip()


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


def parse_args():
    parser = argparse.ArgumentParser(description="Run Grounding DINO on one image and show bounding boxes.")
    parser.add_argument("--image", "-i", help="Image path. If omitted, a file dialog or stdin prompt is used.")
    parser.add_argument("--prompt", "-p", help='Text prompt, e.g. "drone racing gate. square gate. colored foam gate."')
    parser.add_argument("--model", default="IDEA-Research/grounding-dino-tiny", help="Hugging Face model id or local model directory.")
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--device", default="auto", help='auto, cpu, cuda, or cuda:0. Default: "auto".')
    parser.add_argument("--save", help="Optional output image path.")
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = args.image or ask_image_path()
    if not image_path or not os.path.exists(image_path):
        print(f"Image does not exist: {image_path}", file=sys.stderr)
        return 2

    prompt = normalize_prompt(args.prompt or input("Text prompt: "))
    device = pick_device(args.device)

    print(f"Loading model: {args.model}")
    print(f"Device: {device}")
    processor = AutoProcessor.from_pretrained(args.model)
    model = GroundingDinoForObjectDetection.from_pretrained(args.model).to(device)
    model.eval()

    pil_image = Image.open(image_path).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    inputs = processor(images=pil_image, text=prompt, return_tensors="pt")
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    result = post_process(
        processor,
        outputs,
        inputs.get("input_ids"),
        (pil_image.height, pil_image.width),
        args.box_threshold,
        args.text_threshold,
    )

    boxes = result.get("boxes", [])
    count = len(boxes)
    print(f"Detections: {count}")
    for label, score, box in zip(result.get("text_labels", result.get("labels", [])), tensor_to_list(result.get("scores", [])), tensor_to_list(boxes)):
      print(f"  {label}: {score:.3f} box={box}")

    vis = draw_detections(image_bgr, result)
    if args.save:
        cv2.imwrite(args.save, vis)
        print(f"Saved: {args.save}")

    window = "Grounding DINO detections"
    cv2.imshow(window, vis)
    print("Press any key in the image window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
