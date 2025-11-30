#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Embeddings Builder for Open-CoCoLa
=================================

Este script construye:
  (1) Embeddings normalizados para una lista de conceptos.
  (2) Un índice FAISS basado en similitud por producto interno (coseno).
  (3) Un archivo de metadata con la lista de conceptos en el mismo orden que FAISS.

Está diseñado para ser usado dentro de un pipeline de etiquetado semántico
(Open-CoCoLa) donde las etiquetas generadas libremente se proyectan al
espacio de conceptos mediante similitud vectorial.

Autor: Darian Fernández
Licencia: MIT
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import argparse
import pickle
import logging
import sys
from pathlib import Path

import numpy as np
import faiss
from ollama import embeddings

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================
# Funciones auxiliares
# ============================================================
def read_concepts(path: Path) -> List[str]:
    """Lee un archivo .txt con un concepto por línea."""
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")

    with open(path, "r", encoding="utf-8") as f:
        concepts = [line.strip() for line in f if line.strip()]

    if not concepts:
        raise ValueError(f"El archivo {path} no contiene conceptos válidos.")

    logger.info(f"✓ {len(concepts)} conceptos cargados.")
    return concepts


def get_embedding(text: str, model: str, retries: int = 3) -> Optional[List[float]]:
    """Obtiene el embedding con reintentos seguros."""
    for attempt in range(retries):
        try:
            rep = embeddings(model=model, prompt=text)
            return rep["embedding"]
        except Exception as e:
            if attempt == retries - 1:
                logger.error(f"Error obteniendo embedding para '{text}': {e}")
                return None
            logger.warning(f"Reintento {attempt + 1}/{retries} para '{text}'")
    return None


def generate_embeddings(
    concepts: List[str],
    model: str,
) -> Tuple[np.ndarray, List[str]]:
    """
    Genera embeddings normalizados L2.
    Devuelve:
        matriz_embeddings (N, d)
        conceptos_validos (lista filtrada en el mismo orden)
    """
    embeddings_list: List[List[float]] = []
    valid_concepts: List[str] = []

    logger.info(f"Generando embeddings para {len(concepts)} conceptos...")

    for i, concept in enumerate(concepts):
        if (i + 1) % 20 == 0:
            logger.info(f"Procesados: {i + 1}/{len(concepts)}")

        emb = get_embedding(concept, model)
        if emb is not None:
            embeddings_list.append(emb)
            valid_concepts.append(concept)

    if not embeddings_list:
        raise RuntimeError("No se pudo generar ningún embedding válido.")

    mat = np.array(embeddings_list, dtype=np.float32)

    # Normalización L2
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat /= norms

    logger.info(f"✓ {len(valid_concepts)} embeddings generados correctamente.")
    return mat, valid_concepts


def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """Crea un índice FAISS basado en producto interno."""
    dim: int = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    logger.info(f"✓ Índice FAISS creado (dim={dim}, items={vectors.shape[0]})")
    return index


def save_index(
    index: faiss.IndexFlatIP,
    concepts: List[str],
    out_dir: Path,
) -> None:
    """Guarda FAISS + metadata en disco."""
    out_dir.mkdir(parents=True, exist_ok=True)

    index_path = out_dir / "concepts_index.faiss"
    meta_path = out_dir / "concepts_metadata.pkl"

    faiss.write_index(index, str(index_path))

    with open(meta_path, "wb") as f:
        pickle.dump(concepts, f)

    logger.info(f"✓ Índice guardado en: {index_path}")
    logger.info(f"✓ Metadata guardada en: {meta_path}")


# ============================================================
# Main CLI
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Creador de embeddings + FAISS para Open-CoCoLa"
    )

    parser.add_argument(
        "--concepts_txt",
        type=str,
        required=True,
        help="Archivo .txt con un concepto por línea.",
    )
    parser.add_argument(
        "--embed_model",
        type=str,
        default="nomic-embed-text:latest",
        help="Modelo de embeddings de Ollama.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Carpeta donde guardar FAISS + metadata.",
    )

    args = parser.parse_args()

    txt_path = Path(args.concepts_txt)
    out_path = Path(args.output_dir)

    # 1. Leer conceptos
    concepts_raw = read_concepts(txt_path)

    # 2. Generar embeddings normalizados
    vectors, concepts = generate_embeddings(concepts_raw, args.embed_model)

    # 3. Crear índice FAISS
    index = build_faiss_index(vectors)

    # 4. Guardar índice y metadata
    save_index(index, concepts, out_path)

    logger.info("✓ Proceso completado.")


if __name__ == "__main__":
    main()
