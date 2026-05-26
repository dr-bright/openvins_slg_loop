#!/usr/bin/env python3
"""
Export Hugging Face Grounding DINO to ONNX for the C++ experiments.

The export intentionally keeps tokenization outside the ONNX graph. It writes:
  onnx/dino.onnx
  onnx/dino_vocab.txt

The C++ test uses dino_vocab.txt to reproduce BERT uncased WordPiece tokenization.
"""

import argparse
import os
import shutil

import torch
from PIL import Image
from transformers import AutoProcessor, GroundingDinoForObjectDetection


class GroundingDinoOnnxWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, pixel_mask, input_ids, attention_mask, token_type_ids):
        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        return outputs.logits, outputs.pred_boxes


def parse_args():
    parser = argparse.ArgumentParser(description="Dump Grounding DINO ONNX and tokenizer vocabulary.")
    parser.add_argument("--model", default="IDEA-Research/grounding-dino-tiny", help="HF model id or local directory.")
    parser.add_argument("--out", default=None, help="Output ONNX path. Default: <repo>/onnx/dino.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--max-text-len", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_path = args.out or os.path.join(repo_root, "onnx", "dino.onnx")
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading processor/model: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)
    model = GroundingDinoForObjectDetection.from_pretrained(args.model)
    model.config.disable_custom_kernels = True
    model.eval()

    vocab_out = os.path.join(out_dir, "dino_vocab.txt")
    tokenizer_dir = os.path.join(out_dir, "dino_tokenizer")
    processor.tokenizer.save_pretrained(tokenizer_dir)
    vocab = os.path.join(tokenizer_dir, "vocab.txt")
    if not os.path.exists(vocab):
        raise RuntimeError(f"tokenizer did not write vocab.txt into {tokenizer_dir}")
    shutil.copyfile(vocab, vocab_out)
    print(f"Wrote tokenizer vocab: {vocab_out}")

    prompt = "drone racing gate. square colored foam gate."
    image = Image.new("RGB", (640, 480), (127, 127, 127))
    inputs = processor(
        images=image,
        text=prompt,
        padding="max_length",
        max_length=args.max_text_len,
        truncation=True,
        return_tensors="pt",
    )
    print("Processor settings:")
    print(f"  image.size input: {image.size[0]}x{image.size[1]}")
    print(f"  do_resize: {getattr(processor.image_processor, 'do_resize', None)}")
    print(f"  size: {getattr(processor.image_processor, 'size', None)}")
    print(f"  do_rescale: {getattr(processor.image_processor, 'do_rescale', None)}")
    print(f"  do_normalize: {getattr(processor.image_processor, 'do_normalize', None)}")
    print(f"  do_pad: {getattr(processor.image_processor, 'do_pad', None)}")
    print("Export example tensor shapes:")
    for name, tensor in inputs.items():
        print(f"  {name}: shape={tuple(tensor.shape)} dtype={tensor.dtype}")

    wrapper = GroundingDinoOnnxWrapper(model)
    with torch.no_grad():
        print(f"Exporting ONNX: {out_path}")
        torch.onnx.export(
            wrapper,
            (
                inputs["pixel_values"],
                inputs["pixel_mask"],
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs["token_type_ids"],
            ),
            out_path,
            input_names=["pixel_values", "pixel_mask", "input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits", "pred_boxes"],
            dynamic_axes={
                "pixel_values": {2: "image_height", 3: "image_width"},
                "pixel_mask": {1: "image_height", 2: "image_width"},
                "input_ids": {1: "text_length"},
                "attention_mask": {1: "text_length"},
                "token_type_ids": {1: "text_length"},
                "logits": {2: "text_length"},
            },
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=True
        )

    print("Done.")
    print(f"ONNX: {out_path}")
    print(f"Vocab: {vocab_out}")


if __name__ == "__main__":
    raise SystemExit(main())
