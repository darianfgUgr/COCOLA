#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline for hierarchical annotation of objects and parts using a
multimodal model hosted in Ollama. The system applies a closed-set
category selection protocol (CoCoLa), using a hierarchical dictionary
of concepts to guide inference.

Author: Darian Fernández
License: MIT
"""

from __future__ import annotations
import argparse
import base64
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from ollama import Client


# ============================================================
# Logging configuration
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# Global configuration
# ============================================================
DEFAULT_MODEL: str = "llava:7b"
IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".png", ".jpeg")
DEFAULT_CROP_MARGIN: int = 5

OLLAMA_CLIENT: Optional[Client] = None
CONCEPTS: Dict[str, Any] = {}


# ============================================================
# JSON loading functions
# ============================================================
def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Loads a generic JSON file as a dictionary."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON {path}: {e}")
        return None


def load_concepts_from_json(path: Path) -> Dict[str, Any]:
    """Loads the hierarchical concept dictionary."""
    try:
        data = load_json(path)
        if isinstance(data, dict):
            logger.info(f"✓ {len(data)} root categories loaded.")
            return data
        logger.error("The concepts file must contain a dictionary at the root.")
        return {}
    except Exception as e:
        logger.error(f"Error loading concepts: {e}")
        return {}


# ============================================================
# Image functions
# ============================================================
def load_image(
    image_name: str,
    image_dir: Union[str, Path],
    extensions: Tuple[str, ...] = IMAGE_EXTENSIONS,
) -> Optional[Image.Image]:
    """Loads an image by its base name without extension."""
    for ext in extensions:
        path = Path(image_dir) / f"{image_name}{ext}"
        if path.exists():
            try:
                return Image.open(path).convert("RGB")
            except Exception as e:
                logger.error(f"Error opening {path}: {e}")
                return None
    logger.error(f"Image not found for: {image_name}")
    return None


def pil_to_base64(image: Image.Image, fmt: str = "PNG") -> Optional[str]:
    """Converts a PIL image to base64."""
    try:
        buffer = io.BytesIO()
        image.save(buffer, format=fmt)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        logger.error(f"Error converting to base64: {e}")
        return None


def draw_bbox(img: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    """Draws a red bounding box."""
    x1, y1, x2, y2 = map(int, bbox)
    output = img.copy()
    draw = ImageDraw.Draw(output)
    draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
    return output


def crop_with_margin(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    margin: int = DEFAULT_CROP_MARGIN,
) -> Image.Image:
    """Crops a region with an additional margin."""
    x1, y1, x2, y2 = map(int, bbox)
    x1c = max(x1 - margin, 0)
    y1c = max(y1 - margin, 0)
    x2c = min(x2 + margin, image.width)
    y2c = min(y2 + margin, image.height)
    return image.crop((x1c, y1c, x2c, y2c))


def preprocess_image(
    image: Image.Image,
    size: Tuple[int, int] = (512, 512),
    smoothing: bool = True,
) -> Image.Image:
    """Optional resizing and smoothing."""
    out = image.copy()
    if smoothing:
        out = out.filter(ImageFilter.SMOOTH_MORE)
    out = out.resize(size, resample=Image.LANCZOS)
    return out


# ============================================================
# Ollama query functions
# ============================================================
def query_ollama(
    img_b64: str,
    labels: Optional[List[str]],
    model: str,
) -> str:
    """
    CoCoLa closed-set inference: selects exactly one label from a list.
    """
    global OLLAMA_CLIENT

    if labels is None or len(labels) == 0:
        return "unknown"

    prompt = (
        "You are an expert image annotator.\n"
        "Analyze ONLY the object highlighted in the image.\n"
        f"Possible labels: {', '.join(labels)}.\n"
        "Select EXACTLY ONE label from the list.\n"
        "Respond with ONLY ONE WORD from the list or 'unknown'."
    )

    try:
        response = OLLAMA_CLIENT.chat(
            model=model,
            messages=[{"role": "user", "content": prompt, "images": [img_b64]}],
        )
        return response["message"]["content"].strip().lower()
    except Exception as e:
        logger.error(f"Error in Ollama query: {e}")
        return "unknown"


# ============================================================
# Recursive processing of parts
# ============================================================
def process_part(
    part: Dict[str, Any],
    full_image: Image.Image,
    object_bbox: Tuple[int, int, int, int],
    concept_tree: Dict[str, Any],
    model_name: str,
) -> Optional[Dict[str, Any]]:
    bbox = part.get("bbox")
    if bbox is None:
        return None

    img_drawn = draw_bbox(full_image, bbox)
    crop = crop_with_margin(img_drawn, object_bbox, margin=5)
    crop = preprocess_image(crop, size=(512, 512), smoothing=False)

    img_b64 = pil_to_base64(crop)
    if img_b64 is None:
        return None

    labels = list(concept_tree.keys())
    predicted = query_ollama(img_b64, labels, model_name)

    subtree = concept_tree.get(predicted, {})

    subparts_out = []
    for sp in part.get("parts", []):
        p = process_part(sp, full_image, object_bbox, subtree, model_name)
        if p is not None:
            subparts_out.append(p)

    return {
        "class": part.get("class", ""),
        "predicted_part_con_labels": predicted,
        "parts": subparts_out,
    }


# ============================================================
# Main JSON processing
# ============================================================
def process_json(
    data: Dict[str, Any],
    image: Image.Image,
    model_name: str,
) -> Dict[str, Any]:
    output = {"image_name": data["image_name"], "objects": []}

    for obj in data.get("objects", []):
        bbox = obj.get("bbox")
        if bbox is None:
            continue

        img_drawn = draw_bbox(image, bbox)
        img_b64 = pil_to_base64(img_drawn)
        if img_b64 is None:
            continue

        labels = list(CONCEPTS.keys())
        predicted_obj = query_ollama(img_b64, labels, model_name)

        subtree = CONCEPTS.get(predicted_obj, {})

        obj_out = {
            "class": obj.get("class", ""),
            "predicted_class_con_labels": predicted_obj,
            "parts": [],
        }

        for part in obj.get("parts", []):
            p = process_part(part, image, bbox, subtree, model_name)
            if p is not None:
                obj_out["parts"].append(p)

        output["objects"].append(obj_out)

    return output


# ============================================================
# Save JSON
# ============================================================
def save_output(
    data: Dict[str, Any], image_name: str, model: str, out_dir: Path
) -> None:
    model_dir = out_dir / model
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{image_name}_gpt.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================
# Main
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="CoCoLa pipeline for hierarchical annotation with Ollama."
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="JSON file containing the hierarchical concept dictionary.",
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Folder containing dataset JSON annotation files.",
    )
    parser.add_argument(
        "--input_images",
        type=str,
        required=True,
        help="Folder containing the images.",
    )
    parser.add_argument("--output", type=str, required=True, help="Output folder.")

    args = parser.parse_args()

    global OLLAMA_CLIENT, CONCEPTS
    OLLAMA_CLIENT = Client(host=f"http://localhost:{args.port}")

    logger.info(f"Connecting to Ollama on port {args.port}")

    labels_path = Path(args.labels)
    json_dir = Path(args.input_json)
    img_dir = Path(args.input_images)
    out_dir = Path(args.output)

    CONCEPTS = load_concepts_from_json(labels_path)

    json_files = sorted(json_dir.glob("*.json"))
    logger.info(f"JSON files found: {len(json_files)}")

    for jf in json_files:
        data = load_json(jf)
        if data is None:
            continue

        img = load_image(data["image_name"], img_dir)
        if img is None:
            continue

        logger.info(f"Processing {data['image_name']} ...")

        out = process_json(data, img, args.model)
        save_output(out, data["image_name"], args.model, out_dir)

    logger.info("✓ Pipeline completed successfully.")


if __name__ == "__main__":
    main()
