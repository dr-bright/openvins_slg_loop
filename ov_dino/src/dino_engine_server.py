#!/usr/bin/env python3
"""
Grounding DINO stdin/stdout inference engine.

This is an HTTP-like protocol over process pipes. Stdout is reserved for protocol
responses; all logs go to stderr.
"""

import argparse
import json
import logging
import re
import sys
import time
import urllib.parse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, GroundingDinoForObjectDetection
from transformers.utils import logging as hf_logging


CRLF = b"\r\n"
HEADER_END = b"\r\n\r\n"
SUPPORTED_IMAGE_TYPES = ("image/bmp", "image/jpeg", "image/jpg", "image/png", "image/tiff")
VERBOSITY_LEVELS = {"silent": 0, "error": 1, "warning": 2, "info": 3, "debug": 4}
VERBOSITY = VERBOSITY_LEVELS["warning"]


def set_verbosity(level: str):
    global VERBOSITY
    normalized = level.strip().lower()
    if normalized not in VERBOSITY_LEVELS:
        raise ValueError(f"invalid verbosity '{level}', expected one of {sorted(VERBOSITY_LEVELS.keys())}")
    VERBOSITY = VERBOSITY_LEVELS[normalized]

    if normalized == "silent":
        hf_logging.set_verbosity_error()
        logging.getLogger("transformers").setLevel(logging.CRITICAL)
    elif normalized == "error":
        hf_logging.set_verbosity_error()
        logging.getLogger("transformers").setLevel(logging.ERROR)
    elif normalized == "warning":
        hf_logging.set_verbosity_warning()
        logging.getLogger("transformers").setLevel(logging.WARNING)
    elif normalized == "info":
        hf_logging.set_verbosity_info()
        logging.getLogger("transformers").setLevel(logging.INFO)
    else:
        hf_logging.set_verbosity_debug()
        logging.getLogger("transformers").setLevel(logging.DEBUG)

    hf_logging.enable_default_handler()
    hf_logging.disable_progress_bar()


def log(message: str, level: str = "info"):
    if VERBOSITY >= VERBOSITY_LEVELS[level]:
        print(message, file=sys.stderr, flush=True)


@dataclass
class Request:
    method: str
    target: str
    version: str
    headers: Dict[str, str]
    body: bytes


@dataclass
class Part:
    headers: Dict[str, str]
    body: bytes


class HttpError(Exception):
    def __init__(self, status: int, reason: str, message: str):
        super().__init__(message)
        self.status = status
        self.reason = reason
        self.message = message


def read_until(stream, marker: bytes) -> Optional[bytes]:
    data = bytearray()
    while True:
        chunk = stream.read(1)
        if not chunk:
            if not data:
                return None
            raise EOFError("unexpected EOF while reading headers")
        data += chunk
        if data.endswith(marker):
            return bytes(data)


def parse_headers(raw: bytes) -> Tuple[str, Dict[str, str]]:
    text = raw.decode("iso-8859-1")
    lines = text.split("\r\n")
    start = lines[0]
    headers: Dict[str, str] = {}
    for line in lines[1:]:
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"malformed header: {line}")
        key, value = line.split(":", 1)
        headers[key.strip().lower()] = value.strip()
    return start, headers


def read_request() -> Optional[Request]:
    raw_headers = read_until(sys.stdin.buffer, HEADER_END)
    if raw_headers is None:
        return None
    raw_headers = raw_headers[: -len(HEADER_END)]
    start, headers = parse_headers(raw_headers)
    words = start.split()
    if len(words) != 3:
        raise ValueError(f"malformed request line: {start}")
    method, target, version = words
    content_length = int(headers.get("content-length", "0"))
    body = sys.stdin.buffer.read(content_length)
    if len(body) != content_length:
        raise EOFError("unexpected EOF while reading body")
    return Request(method=method.upper(), target=target, version=version, headers=headers, body=body)


def write_response(status: int, reason: str, body: bytes, content_type: str = "application/json"):
    headers = [
        f"HTTP/1.1 {status} {reason}",
        f"Content-Type: {content_type}",
        f"Content-Length: {len(body)}",
        "Connection: keep-alive",
        "",
        "",
    ]
    sys.stdout.buffer.write("\r\n".join(headers).encode("ascii"))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def json_response(status: int, reason: str, payload):
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    write_response(status, reason, body)


def parse_target(target: str):
    parsed = urllib.parse.urlsplit(target)
    query = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    prompt = urllib.parse.unquote(parsed.fragment) if parsed.fragment else query.get("prompt", "")
    return parsed.path, query, prompt


def content_type_params(value: str) -> Tuple[str, Dict[str, str]]:
    pieces = [p.strip() for p in value.split(";")]
    mime = pieces[0].lower() if pieces else ""
    params: Dict[str, str] = {}
    for piece in pieces[1:]:
        if "=" not in piece:
            continue
        key, val = piece.split("=", 1)
        val = val.strip()
        if len(val) >= 2 and val[0] == '"' and val[-1] == '"':
            val = val[1:-1]
        params[key.strip().lower()] = val
    return mime, params


def parse_multipart(body: bytes, boundary: str) -> List[Part]:
    delimiter = b"--" + boundary.encode("ascii")
    final_delimiter = delimiter + b"--"
    parts: List[Part] = []
    for section in body.split(delimiter):
        if not section or section in (b"--", b"--\r\n"):
            continue
        if section.startswith(b"--"):
            break
        if section.startswith(CRLF):
            section = section[len(CRLF) :]
        if section.endswith(CRLF):
            section = section[: -len(CRLF)]
        if section == final_delimiter:
            break
        header_pos = section.find(HEADER_END)
        if header_pos < 0:
            raise ValueError("multipart part missing header terminator")
        raw_headers = section[:header_pos]
        part_body = section[header_pos + len(HEADER_END) :]
        _, headers = parse_headers(b"PART " + raw_headers)
        parts.append(Part(headers=headers, body=part_body))
    return parts


def decode_image(body: bytes, headers: Dict[str, str]) -> Image.Image:
    content_type = headers.get("content-type")
    if not content_type:
        raise HttpError(400, "Bad Request", "image part missing Content-Type")
    mime, _ = content_type_params(content_type)
    if mime not in SUPPORTED_IMAGE_TYPES:
        raise HttpError(415, "Unsupported Media Type", f"unsupported image Content-Type: {mime}")
    try:
        import io

        return Image.open(io.BytesIO(body)).convert("RGB")
    except Exception as e:
        raise HttpError(400, "Bad Request", f"failed to decode {mime}: {e}")


class DinoEngine:
    def __init__(self, model: str, device_arg: str):
        self.device = torch.device("cuda" if device_arg == "auto" and torch.cuda.is_available() else ("cpu" if device_arg == "auto" else device_arg))
        log(f"[dino_engine] loading model={model} device={self.device}", "info")
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = GroundingDinoForObjectDetection.from_pretrained(model).to(self.device)
        self.model.eval()
        self.model_name = model
        log("[dino_engine] ready", "info")

    def detect(self, pil_images: List[Image.Image], prompt: str, box_threshold: float, text_threshold: float, max_detections: int):
        if not prompt.strip():
            raise HttpError(400, "Bad Request", "prompt is empty")
        if not prompt.strip().endswith("."):
            prompt = prompt.strip() + "."

        t0 = time.perf_counter()
        inputs = self.processor(images=pil_images, text=[prompt] * len(pil_images), return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        t1 = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(**inputs)
        t2 = time.perf_counter()
        target_sizes = torch.tensor([[img.height, img.width] for img in pil_images], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )
        t3 = time.perf_counter()

        detections = []
        for image_index, result in enumerate(results):
            boxes = result["boxes"].detach().cpu().tolist()
            scores = result["scores"].detach().cpu().tolist()
            labels = result.get("labels", result.get("text_labels", []))
            order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
            if max_detections > 0:
                order = order[:max_detections]
            for detection_index in order:
                x0, y0, x1, y1 = boxes[detection_index]
                detections.append(
                    {
                        "image": image_index,
                        "label": labels[detection_index],
                        "score": float(scores[detection_index]),
                        "x": float(x0),
                        "y": float(y0),
                        "dx": float(x1 - x0),
                        "dy": float(y1 - y0),
                    }
                )

        return {
            "ok": True,
            "detections": detections,
            "timing_ms": {
                "preprocess": (t1 - t0) * 1000.0,
                "infer": (t2 - t1) * 1000.0,
                "postprocess": (t3 - t2) * 1000.0,
            },
        }


def handle_run_dino(engine: DinoEngine, req: Request, query: Dict[str, str], prompt: str):
    box_threshold = float(query.get("box_threshold", query.get("box", "0.25")))
    text_threshold = float(query.get("text_threshold", query.get("text", "0.25")))
    max_detections = int(query.get("max_detections", "0"))
    content_type = req.headers.get("content-type")
    if not content_type:
        raise HttpError(400, "Bad Request", "request missing Content-Type")
    mime, params = content_type_params(content_type)

    images = []
    if mime.startswith("multipart/"):
        boundary = params.get("boundary", "")
        if not boundary:
            raise HttpError(400, "Bad Request", "multipart request missing boundary")
        for part in parse_multipart(req.body, boundary):
            images.append(decode_image(part.body, part.headers))
    elif mime in SUPPORTED_IMAGE_TYPES:
        images.append(decode_image(req.body, req.headers))
    else:
        raise HttpError(415, "Unsupported Media Type", f"unsupported request Content-Type: {mime}")

    if not images:
        raise HttpError(400, "Bad Request", "request contains no images")
    return engine.detect(images, prompt, box_threshold, text_threshold, max_detections)


def parse_args():
    parser = argparse.ArgumentParser(description="Grounding DINO stdin/stdout pipe server.")
    parser.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verbosity", default="warning", choices=sorted(VERBOSITY_LEVELS.keys()))
    return parser.parse_args()


def main():
    args = parse_args()
    set_verbosity(args.verbosity)
    engine = DinoEngine(args.model, args.device)
    while True:
        try:
            req = read_request()
            if req is None:
                return 0
            path, query, prompt = parse_target(req.target)
            if req.method == "GET" and path == "/health":
                json_response(200, "OK", {"ok": True, "model": engine.model_name, "device": str(engine.device)})
            elif req.method == "GET" and path == "/shutdown":
                json_response(200, "OK", {"ok": True})
                return 0
            elif req.method == "POST" and path == "/run_dino":
                payload = handle_run_dino(engine, req, query, prompt)
                json_response(200, "OK", payload)
            else:
                json_response(404, "Not Found", {"ok": False, "error": f"unknown endpoint {req.method} {path}"})
        except Exception as e:
            if isinstance(e, HttpError):
                log(f"[dino_engine] request rejected: {e.message}", "warning")
                json_response(e.status, e.reason, {"ok": False, "error": e.message})
            else:
                log(f"[dino_engine] request failed: {e}", "error")
                json_response(500, "Internal Server Error", {"ok": False, "error": str(e)})


if __name__ == "__main__":
    raise SystemExit(main())
