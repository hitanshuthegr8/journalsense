"""
AI Journal Recommender - semantic journal discovery over OpenAlex.

Retrieval: SPECTER (768-d) scientific-document embeddings, L2-normalised into a
FAISS IndexFlatIP so inner product == cosine similarity. Exact search is used
deliberately: at 1k vectors the index scan is ~0.1 ms of a ~153 ms query, so an
ANN index would trade recall for time we aren't spending. See Models/eval/.

All journal metrics shown are real OpenAlex values. Nothing is synthesised.
"""

import faiss
import numpy as np
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

OPENALEX_SOURCES = "https://api.openalex.org/sources"

# Only the fields we actually use - keeps payloads small and fetches fast.
SOURCE_FIELDS = ",".join([
    "id", "display_name", "alternate_titles", "issn_l", "type",
    "works_count", "cited_by_count", "homepage_url",
    "host_organization_name", "topics", "summary_stats",
    "is_in_doaj", "is_oa", "is_core",
])


# ------------------------------------------------------------------
# 1. Embedding model
@st.cache_resource(show_spinner=False, ttl=24 * 3600)
def load_embedder():
    return SentenceTransformer("allenai-specter")


# ------------------------------------------------------------------
# 3. Fetch journal metadata
#
# NOTE: OpenAlex removed `description` and `abbreviated_title` from source
# records - they were empty for 100% of 1,000 sampled journals, which meant the
# index was silently embedding titles alone. Topical scope now comes from the
# `topics` field instead. `type:journal` excludes repositories and aggregators
# (Zenodo, Figshare, PubMed) that pollute the default listing.
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_openalex_journals(target: int = 1000, per_page: int = 200):
    journals, cursor = [], "*"
    try:
        while len(journals) < target:
            resp = requests.get(OPENALEX_SOURCES, timeout=30, params={
                "filter": "type:journal",
                "sort": "cited_by_count:desc",
                "per-page": per_page,
                "cursor": cursor,
                "select": SOURCE_FIELDS,
            })
            if resp.status_code != 200:
                st.warning(f"OpenAlex API error: {resp.status_code}")
                break
            data = resp.json()
            journals.extend(data.get("results", []))
            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break
        return journals[:target]
    except requests.exceptions.Timeout:
        st.error(f"OpenAlex timed out after 30s with {len(journals)} journals fetched.")
        return []
    except requests.exceptions.RequestException as e:
        st.error(f"Network error contacting OpenAlex ({type(e).__name__}): {e}")
        return []
    except ValueError as e:
        # requests raises ValueError from .json() on a malformed body.
        st.error(f"OpenAlex returned a response that is not valid JSON: {e}")
        return []


def journal_index_text(j: dict) -> str:
    """The string that actually gets embedded: identity + topical scope.

    Raised Recall@10 from 39.2% to 51.1% on a 423-query eval set versus the
    previous title-only text (see Models/eval/results.json).
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
# 5. FAISS index over journal embeddings
@st.cache_resource(show_spinner=False)
def build_faiss_index(journals, _model):
    texts = [journal_index_text(j) for j in journals]
    if not texts:
        st.error("No journals available to index")
        return faiss.IndexFlatIP(768)

    embs = _model.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
    embs = np.asarray(embs, dtype=np.float32)
    faiss.normalize_L2(embs)                 # so inner product == cosine
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
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

    with st.spinner("Loading journal database..."):
        journals = fetch_openalex_journals()
    if not journals:
        st.error("Could not load journals from OpenAlex. Try again shortly.")
        return
    st.caption(f"Indexing {len(journals):,} journals ranked by citation impact (OpenAlex).")

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
    embedder = load_embedder()

    with st.spinner("Extracting key topics..."):
        phrases = extract_key_phrases(query)
    st.subheader("Key Topics")
    st.write(" - ".join(phrases) or "N/A")

    with st.spinner("Building recommendation index..."):
        index = build_faiss_index(journals, embedder)

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
