"""
AI Journal Recommender - semantic journal discovery over OpenAlex.

Retrieval: SPECTER (768-d) scientific-document embeddings, L2-normalised into a
FAISS IndexFlatIP so inner product == cosine similarity. Exact search is used
deliberately: at 1k vectors the index scan is ~0.1 ms of a ~153 ms query, so an
ANN index would trade recall for time we aren't spending. See Models/eval/.

The corpus and its embeddings are built OFFLINE by build_index.py and loaded at
startup. Nothing is fetched or encoded per request except the user's own query.

All journal metrics are real OpenAlex values - a snapshot from index build time,
not live. Nothing is synthesised.
"""

import hashlib
import json
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer


HERE = Path(__file__).parent
CORPUS_PATH = HERE / "eval" / "journals.json"
INDEX_DIR = HERE / "index"


# ------------------------------------------------------------------
# 1. Embedding model
@st.cache_resource(show_spinner=False, ttl=24 * 3600)
def load_embedder():
    return SentenceTransformer("allenai-specter")


# ------------------------------------------------------------------
# 3. Load the corpus and its precomputed embeddings
#
# This used to fetch 1,000 journals from OpenAlex and embed all of them inside the
# button handler: ~16s of network plus ~80s of encoding on every cold process. That
# is a build step, not a request step - see build_index.py. The app now ships with
# the corpus and its embeddings and just loads them.
#
# Consequence worth being explicit about: journal metrics are a SNAPSHOT taken when
# the index was built, not live values. The UI says so.
@st.cache_resource(show_spinner=False)
def load_corpus_and_embeddings():
    if not CORPUS_PATH.exists():
        st.error(f"Corpus fixture missing: {CORPUS_PATH}")
        return None, None, None
    emb_path, man_path = INDEX_DIR / "embeddings.npy", INDEX_DIR / "manifest.json"
    if not emb_path.exists() or not man_path.exists():
        st.error("Prebuilt index missing. Run `python build_index.py` first.")
        return None, None, None

    journals = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    embs = np.load(emb_path).astype(np.float32)
    manifest = json.loads(man_path.read_text(encoding="utf-8"))

    # The embeddings are positionally aligned to the corpus. If the corpus changed
    # without a rebuild, every row would point at the wrong journal and the app
    # would return confidently wrong results - so fail loudly instead.
    digest = hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest()[:16]
    if digest != manifest.get("corpus_sha256_16"):
        st.error(
            "Index is stale: the corpus has changed since the embeddings were built. "
            "Re-run `python build_index.py`."
        )
        return None, None, None
    if len(journals) != embs.shape[0]:
        st.error(f"Corpus has {len(journals)} journals but index has {embs.shape[0]} rows.")
        return None, None, None

    return journals, embs, manifest


# ------------------------------------------------------------------
# 4. Research domains (OpenAlex topic hierarchy: domain > field > subfield)
#
# The previous implementation read `x_concepts`, which OpenAlex has removed -
# it returned an empty set, so selecting any domain filtered out every result.
def journal_domains(j: dict) -> list:
    return sorted({
        t["domain"]["display_name"]
        for t in (j.get("topics") or [])
        if (t.get("domain") or {}).get("display_name")
    })


def extract_journal_domains(journals) -> list:
    domains = set()
    for j in journals:
        domains.update(journal_domains(j))
    return sorted(domains)


# ------------------------------------------------------------------
# 5. FAISS index from the precomputed embeddings
#
# Adding 1,000 pre-normalised vectors to a flat index is sub-millisecond. The
# expensive part - encoding them - happened offline in build_index.py.
@st.cache_resource(show_spinner=False)
def build_faiss_index(_embs):
    index = faiss.IndexFlatIP(_embs.shape[1])
    index.add(_embs)                          # already L2-normalised at build time
    return index


# ------------------------------------------------------------------
# 6. Key-phrase extraction (DISPLAY ONLY - has no effect on ranking)
#
# This previously used spaCy noun chunks. spaCy has been removed: the pinned model was
# a 2.x build running against a 3.x runtime, so it never loaded, and the except-branch
# silently fell back to spacy.blank("en") - which has no noun-chunk parser - meaning
# this function had already been degraded to its crude fallback with no warning. It also
# shelled out to `spacy download` at runtime, which cannot work inside a container.
#
# Replaced with a frequency count over stopword-filtered n-grams via scikit-learn, which
# is already a dependency. Deterministic, no model download, and honest about being a
# heuristic rather than linguistic parsing.
def extract_key_phrases(text, top_k=5):
    try:
        vec = CountVectorizer(ngram_range=(2, 3), stop_words="english", min_df=1)
        counts = vec.fit_transform([text])
        ranked = sorted(
            zip(vec.get_feature_names_out(), counts.toarray()[0]),
            key=lambda kv: (-kv[1], kv[0]),
        )
        return [phrase for phrase, count in ranked[:top_k] if count > 0]
    except ValueError:
        # Raised when the text is empty or contains only stopwords.
        return []


# ------------------------------------------------------------------
# 7. Real OpenAlex metrics - no synthetic values
#
# This previously returned random.random() impact factors and randomly sampled
# Scopus/WoS/UGC-CARE indexing badges, rendered to users as if they were real.
def journal_metrics(j: dict) -> dict:
    stats = j.get("summary_stats") or {}
    return {
        # OpenAlex's 2-year mean citedness: same definition as an impact factor,
        # computed by OpenAlex. Not the Clarivate JIF, and not claimed to be.
        "citedness_2yr": round(stats.get("2yr_mean_citedness") or 0.0, 2),
        "h_index": stats.get("h_index") or 0,
        "i10_index": stats.get("i10_index") or 0,
        "works_count": j.get("works_count") or 0,
        "cited_by_count": j.get("cited_by_count") or 0,
        "in_doaj": bool(j.get("is_in_doaj")),
        "is_oa": bool(j.get("is_oa")),
        "is_core": bool(j.get("is_core")),
    }


# ------------------------------------------------------------------
# 8. Recommend journals
def recommend_journals(query, journals, index, model, domains=None, top_k=10):
    if not journals or index.ntotal == 0:
        return []

    q = model.encode([query], convert_to_numpy=True)
    q = np.asarray(q, dtype=np.float32)
    faiss.normalize_L2(q)

    # Over-fetch 3x: the domain filter is applied post-retrieval, so searching
    # for exactly top_k would return a short list once anything is filtered out.
    scores, ids = index.search(q, min(top_k * 3, index.ntotal))

    recs = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0 or idx >= len(journals):
            continue
        j = journals[idx]
        j_domains = journal_domains(j)
        if domains and not set(domains) & set(j_domains):
            continue
        recs.append({
            "title": j["display_name"],
            "publisher": j.get("host_organization_name") or "N/A",
            "issn": j.get("issn_l") or "N/A",
            "url": j.get("homepage_url") or j.get("id"),
            "domains": j_domains,
            "topics": [t["display_name"] for t in (j.get("topics") or [])[:5]],
            "score": float(score),
            "metrics": journal_metrics(j),
        })
        if len(recs) >= top_k:
            break
    return recs


# ------------------------------------------------------------------
# 9. Streamlit UI
def main():
    st.title("AI Journal Recommender")
    st.write("Paste your paper title and abstract, then hit **Suggest Journals**.")

    # Both loads happen at page load, not on the button press. They are cached for the
    # life of the process, so the cost is paid once at startup rather than being charged
    # to whoever clicks first.
    with st.spinner("Loading index..."):
        journals, embs, manifest = load_corpus_and_embeddings()
    if journals is None:
        return
    with st.spinner("Loading embedding model (first run downloads ~440 MB)..."):
        embedder = load_embedder()
    index = build_faiss_index(embs)
    st.caption(
        f"Searching {len(journals):,} OpenAlex journals ranked by citation impact. "
        f"Metrics are a snapshot from when the index was built, not live values."
    )

    st.sidebar.header("Filters")
    selected_domains = st.sidebar.multiselect("Research Domains", extract_journal_domains(journals))
    min_citedness = st.sidebar.slider("Min. 2-yr mean citedness", 0.0, 30.0, 0.0, step=0.5)
    doaj_only = st.sidebar.checkbox("DOAJ-listed only")
    oa_only = st.sidebar.checkbox("Open access only")
    num_rec = st.sidebar.slider("Number of recommendations", 1, 10, 3)

    title = st.text_input("Paper Title")
    abstract = st.text_area("Paper Abstract", height=200)

    if not st.button("Suggest Journals"):
        return
    if not title.strip() or not abstract.strip():
        st.error("Both title and abstract are required.")
        return

    query = f"{title} {abstract}"

    with st.spinner("Extracting key topics..."):
        phrases = extract_key_phrases(query)
    st.subheader("Key Topics")
    st.write(" - ".join(phrases) or "N/A")

    recs = recommend_journals(query, journals, index, embedder, selected_domains, top_k=30)
    if not recs:
        st.warning("No journals matched. Try clearing the domain filter.")
        return

    shown = 0
    st.subheader("Recommendations")
    for r in recs:
        m = r["metrics"]
        if m["citedness_2yr"] < min_citedness:
            continue
        if doaj_only and not m["in_doaj"]:
            continue
        if oa_only and not m["is_oa"]:
            continue

        shown += 1
        st.markdown(f"**{shown}. {r['title']}**")
        st.markdown(f"- Publisher: {r['publisher']}  |  ISSN-L: {r['issn']}")
        st.markdown(f"- Similarity: {r['score']:.3f}")
        st.markdown(f"- Domains: {', '.join(r['domains']) or 'N/A'}")
        st.markdown(f"- Topics: {', '.join(r['topics']) or 'N/A'}")
        st.markdown(
            f"- 2-yr mean citedness: {m['citedness_2yr']} | "
            f"h-index: {m['h_index']:,} | works: {m['works_count']:,}"
        )
        badges = [
            name for name, on in [
                ("DOAJ", m["in_doaj"]),
                ("Open Access", m["is_oa"]),
                ("OpenAlex core", m["is_core"]),
            ] if on
        ]
        st.markdown(f"- {' | '.join(badges) if badges else 'No OA/DOAJ listing'}")
        st.markdown(f"- [Journal page]({r['url']})")
        st.write("")
        if shown >= num_rec:
            break

    if shown == 0:
        st.warning("No journals match your filters. Try broadening your criteria.")

    st.caption(
        "Metrics are live OpenAlex values. '2-yr mean citedness' is OpenAlex's own "
        "measure, not the Clarivate Journal Impact Factor."
    )


if __name__ == "__main__":
    # Show the user something useful, then re-raise so the full traceback still reaches
    # the server log. The previous version swallowed every exception, which made a
    # network failure and a genuine bug look identical from the outside.
    try:
        main()
    except Exception:
        st.error("An unexpected error occurred. See the server log for the traceback.")
        raise
