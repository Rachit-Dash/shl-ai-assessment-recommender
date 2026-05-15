import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np

from utils.helpers import catalog_item_text


DEFAULT_CATALOG_PATH = Path("catalog/shl_catalog.json")
DEFAULT_INDEX_PATH = Path("embeddings/faiss.index")
DEFAULT_METADATA_PATH = Path("embeddings/metadata.pkl")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_catalog(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Catalog file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("Catalog JSON must be a list")
    items = [item for item in data if isinstance(item, dict) and item.get("name") and item.get("url")]
    if not items:
        raise ValueError("Catalog has no usable items")
    return items


def build_index(catalog_path: Path, index_path: Path, metadata_path: Path, model_name: str = MODEL_NAME) -> None:
    import faiss
    from sentence_transformers import SentenceTransformer

    items = load_catalog(catalog_path)
    texts = [catalog_item_text(item) for item in items]
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    vectors = np.asarray(embeddings, dtype="float32")
    if vectors.ndim != 2 or vectors.shape[0] != len(items):
        raise ValueError("Unexpected embedding shape")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with metadata_path.open("wb") as file:
        pickle.dump(items, file)

    print(f"Built FAISS index with {len(items)} items")
    print(f"Index: {index_path}")
    print(f"Metadata: {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SHL catalog embeddings and FAISS index.")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG_PATH), help="Catalog JSON path")
    parser.add_argument("--index", default=str(DEFAULT_INDEX_PATH), help="FAISS index output path")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA_PATH), help="Metadata pickle output path")
    parser.add_argument("--model", default=MODEL_NAME, help="SentenceTransformer model name")
    args = parser.parse_args()
    build_index(Path(args.catalog), Path(args.index), Path(args.metadata), args.model)


if __name__ == "__main__":
    main()
