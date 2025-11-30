#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Open-CoCoLa: Open-Set Concept Labeling
======================================

Experimental pipeline for annotation based on free-form descriptions
and concept assignment through FAISS + embeddings.

Workflow:
  (1) The multimodal model generates a free-form label.
  (2) That label is converted into an embedding.
  (3) The nearest concept is obtained through FAISS (closed projection).
  (4) The FAISS-predicted concept is used to traverse the hierarchical part tree.
  (5) All parts are recursively processed.

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
import pickle
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from ollama import Client, embeddings


# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# Default parameters
# ============================================================
DEFAULT_MODEL: str = "llava:7b"
DEFAULT_EMBED_MODEL: str = "nomic-embed-text:latest"
DEFAULT_IMAGE_EXT: Tuple[str, ...] = (".jpg", ".jpeg", ".png")

DEFAULT_CROP_MARGIN: int = 5
DEFAULT_FAISS_THRESHOLD: float = 0.50

OLLAMA_CLIENT: Optional[Client] = None
CONCEPTS: Dict[str, Any] = {}

FAISS_INDEX: Optional[faiss.IndexFlatIP] = None
FAISS_CONCEPTS: List[str] = []


# ============================================================
# JSON utilities
# ============================================================
def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON {path}: {e}")
        return None


def load_concepts(path: Path) -> Dict[str, Any]:
    data = load_json(path)
    if not isinstance(data, dict):
        logger.error(
            "The hierarchical dictionary must be a JSON file containing a dict at the root."
        )
        return {}
    logger.info(f"✓ {len(data)} root categories loaded.")
    return data


# ============================================================
# Image loading and preprocessing
# ============================================================
def load_image(name: str, img_dir: Path) -> Optional[Image.Image]:
    for ext in DEFAULT_IMAGE_EXT:
        p = img_dir / f"{name}{ext}"
        if p.exists():
            try:
                return Image.open(p).convert("RGB")
            except Exception as e:
                logger.error(f"Error loading image {p}: {e}")
                return None
    logger.error(f"Image not found: {name} in {img_dir}")
    return None


def draw_bbox(img: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    x1, y1, x2, y2 = map(int, bbox)
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
    return out


def crop_margin(
    img: Image.Image,
    bbox: Tuple[int, int, int, int],
    margin: int = DEFAULT_CROP_MARGIN,
) -> Image.Image:
    x1, y1, x2, y2 = map(int, bbox)
    return img.crop(
        (
            max(x1 - margin, 0),
            max(y1 - margin, 0),
            min(x2 + margin, img.width),
            min(y2 + margin, img.height),
        )
    )


def preprocess(img: Image.Image, size: Tuple[int, int] = (512, 512)) -> Image.Image:
    """Resize + smoothing."""
    out = img.copy()
    out = out.filter(ImageFilter.SMOOTH_MORE)
    return out.resize(size, Image.LANCZOS)


def pil_to_b64(img: Image.Image) -> Optional[str]:
    try:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        logger.error(f"Error converting to base64: {e}")
        return None


# ============================================================
# Ollama queries
# ============================================================
def query_freeform(b64: str, model: str) -> str:
    """Free-form prediction: generates one descriptive word."""
    prompt = (
        "You are an expert image annotator.\n"
        "Analyze ONLY the content inside the red bounding box.\n"
        "Respond with ONE WORD describing the object.\n"
        "Do not include explanations."
    )

    try:
        resp = OLLAMA_CLIENT.chat(
            model=model,
            messages=[{"role": "user", "content": prompt, "images": [b64]}],
        )
        return resp["message"]["content"].strip().lower()
    except Exception as e:
        logger.error(f"Error in Ollama freeform query: {e}")
        return "unknown"


def query_constrained(b64: str, labels: List[str], model: str) -> str:
    """Closed-set prediction (parts): selects exactly one label."""
    if not labels:
        return "unknown"

    prompt = (
        "You are an expert image annotator.\n"
        "Analyze ONLY the red bounding box.\n"
        f"Possible labels: {', '.join(labels)}.\n"
        "Respond with ONE WORD from the list."
    )

    try:
        resp = OLLAMA_CLIENT.chat(
            model=model,
            messages=[{"role": "user", "content": prompt, "images": [b64]}],
        )
        return resp["message"]["content"].strip().lower()
    except Exception as e:
        logger.error(f"Error in Ollama constrained query: {e}")
        return "unknown"


# ============================================================
# Embeddings + FAISS
# ============================================================
def load_faiss(index_path: Path, metadata_path: Path) -> None:
    global FAISS_INDEX, FAISS_CONCEPTS

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"FAISS metadata not found: {metadata_path}")

    FAISS_INDEX = faiss.read_index(str(index_path))

    with open(metadata_path, "rb") as f:
        FAISS_CONCEPTS = list(pickle.load(f))

    logger.info(f"✓ FAISS loaded: {len(FAISS_CONCEPTS)} concepts")


def embed_text(text: str, model: str) -> Optional[np.ndarray]:
    try:
        rep = embeddings(model=model, prompt=text)
        v = np.asarray(rep["embedding"], dtype=np.float32)
        v /= np.linalg.norm(v)
        return v
    except Exception as e:
        logger.error(f"Error obtaining embedding: {e}")
        return None


def faiss_match(text: str, threshold: float) -> str:
    if text in ["unknown", "", None]:
        return "unknown"

    if FAISS_INDEX is None:
        logger.error("FAISS index not initialized.")
        return "unknown"

    v = embed_text(text, DEFAULT_EMBED_MODEL)
    if v is None:
        return "unknown"

    sim, idx = FAISS_INDEX.search(v.reshape(1, -1), 1)
    score: float = float(sim[0][0])
    j: int = int(idx[0][0])

    if score < threshold:
        return "unknown"

    if j < 0 or j >= len(FAISS_CONCEPTS):
        return "unknown"

    return FAISS_CONCEPTS[j]


# ============================================================
# Recursive part processing
# ============================================================
def process_part(
    part: Dict[str, Any],
    full_image: Image.Image,
    obj_bbox: Tuple[int, int, int, int],
    concept_tree: Dict[str, Any],
    model: str,
) -> Optional[Dict[str, Any]]:
    bbox = part.get("bbox")
    if bbox is None:
        return None

    img = draw_bbox(full_image, bbox)
    crop = crop_margin(img, obj_bbox)
    crop = preprocess(crop)
    b64 = pil_to_b64(crop)
    if b64 is None:
        return None

    labels = list(concept_tree.keys())
    pred = query_constrained(b64, labels, model)

    subtree = concept_tree.get(pred, {})

    out_parts = []
    for sp in part.get("parts", []):
        p = process_part(sp, full_image, obj_bbox, subtree, model)
        if p is not None:
            out_parts.append(p)

    return {
        "class": part.get("class", ""),
        "predicted_part_con_labels": pred,
        "parts": out_parts,
    }


# ============================================================
# Main image-level processing
# ============================================================
def process_image_json(
    item: Dict[str, Any],
    img: Image.Image,
    model: str,
    faiss_thresh: float,
) -> Dict[str, Any]:
    out = {"image_name": item["image_name"], "objects": []}

    for obj in item.get("objects", []):
        bbox = obj.get("bbox")
        if bbox is None:
            continue

        img_draw = draw_bbox(img, bbox)
        b64 = pil_to_b64(img_draw)
        if b64 is None:
            continue

        # 1. Free-form prediction
        freeform = query_freeform(b64, model)

        # 2. Projection to concept via FAISS
        mapped = faiss_match(freeform, faiss_thresh)

        # 3. Part hierarchy
        subtree = CONCEPTS.get(mapped, {})

        obj_out = {
            "class": obj.get("class", ""),
            "predicted_class_sin_labels": freeform,
            "faiss_predicted": mapped,
            "parts": [],
        }

        for part in obj.get("parts", []):
            p = process_part(part, img, bbox, subtree, model)
            if p is not None:
                obj_out["parts"].append(p)

        out["objects"].append(obj_out)

    return out


# ============================================================
# Save
# ============================================================
def save_output(data: Dict[str, Any], name: str, model: str, out_dir: Path) -> None:
    dest = out_dir / model
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / f"{name}_gpt.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================
# Main
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Open-CoCoLa Pipeline")

    # Models
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--embed_model", type=str, default=DEFAULT_EMBED_MODEL)

    # Paths
    parser.add_argument("--concepts", type=str, required=True)
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--input_images", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    # FAISS
    parser.add_argument("--faiss_index", type=str, required=True)
    parser.add_argument("--faiss_metadata", type=str, required=True)
    parser.add_argument(
        "--faiss_threshold", type=float, default=DEFAULT_FAISS_THRESHOLD
    )

    args = parser.parse_args()

    global OLLAMA_CLIENT, CONCEPTS, FAISS_INDEX, FAISS_CONCEPTS, DEFAULT_EMBED_MODEL
    DEFAULT_EMBED_MODEL = args.embed_model

    # Connect to Ollama multimodal model
    OLLAMA_CLIENT = Client(host=f"http://localhost:{args.port}")
    logger.info(f"Connecting to Ollama on port {args.port}")

    # Load concepts
    CONCEPTS = load_concepts(Path(args.concepts))

    # Load FAISS index
    load_faiss(Path(args.faiss_index), Path(args.faiss_metadata))

    json_dir = Path(args.input_json)
    img_dir = Path(args.input_images)
    out_dir = Path(args.output)

    files = sorted(json_dir.glob("*.json"))
    logger.info(f"Files to process: {len(files)}")

    for f in files:
        data = load_json(f)
        if data is None:
            continue

        img = load_image(data["image_name"], img_dir)
        if img is None:
            continue

        logger.info(f"Processing {data['image_name']} ...")
        result = process_image_json(data, img, args.model, args.faiss_threshold)
        save_output(result, data["image_name"], args.model, out_dir)

    logger.info("✓ Open-CoCoLa completed.")


if __name__ == "__main__":
    main()
