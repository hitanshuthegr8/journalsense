"""Offline index builder. Run this, commit the output, deploy the result.

WHY THIS EXISTS
---------------
model2.py used to fetch 1,000 journals from OpenAlex and embed all of them inside
the Streamlit button handler - about 16s of network plus 80s of encoding, paid on
the first query of every cold process. That is not deployable: free hosting tiers
either time out or run out of memory before the first result renders.

Embedding the corpus is a build step, not a request step. This script does it once
and writes an artifact the app loads at startup.

Output (Models/index/):
    embeddings.npy   float32 [n, 768], L2-normalised, row i == journals.json[i]
    manifest.json    corpus size, model name, dim, source digest

The rows are positionally aligned to Models/eval/journals.json - the same pinned
fixture the benchmark runs against, so the deployed app and the reported numbers
describe the same corpus. manifest.json records a digest of that file; the app
refuses to start if they no longer match.

USAGE
-----
    python build_index.py
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).parent
CORPUS = HERE / "eval" / "journals.json"
OUT_DIR = HERE / "index"
MODEL_NAME = "allenai-specter"


def journal_index_text(j: dict) -> str:
    """MUST stay identical to model2.journal_index_text.

    If these drift, the deployed app stops matching the benchmark. model2.py
    imports this function rather than duplicating it, so there is one definition.
    """
    parts = [j["display_name"]]
    alts = (j.get("alternate_titles") or [])[:2]
    if alts:
        parts.append(" / ".join(alts))
    if j.get("host_organization_name"):
        parts.append(j["host_organization_name"])
    topics = [t["display_name"] for t in (j.get("topics") or [])[:12]]
    if topics:
        parts.append("Publishes research on: " + "; ".join(topics))
    return " | ".join(parts)


def corpus_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> None:
    journals = json.loads(CORPUS.read_text(encoding="utf-8"))
    print(f"corpus       : {len(journals)} journals from {CORPUS.name}")

    texts = [journal_index_text(j) for j in journals]
    print(f"loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("encoding ... (this is the slow part, and the whole point of doing it here)")
    embs = np.asarray(
        model.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=True),
        dtype=np.float32,
    )

    # L2-normalise once, at build time, so the app's inner-product search is cosine
    # similarity without doing this work per request.
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs /= norms

    OUT_DIR.mkdir(exist_ok=True)
    np.save(OUT_DIR / "embeddings.npy", embs)

    manifest = {
        "corpus_size": len(journals),
        "embedding_model": MODEL_NAME,
        "embedding_dim": int(embs.shape[1]),
        "normalised": True,
        "corpus_file": f"eval/{CORPUS.name}",
        "corpus_sha256_16": corpus_digest(CORPUS),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    mb = embs.nbytes / 1_048_576
    print(f"\nwrote {OUT_DIR / 'embeddings.npy'}  shape={embs.shape}  {mb:.1f} MB")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
