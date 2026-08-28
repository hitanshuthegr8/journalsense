<div align="center">

# JournalSense

**Semantic journal recommendation for academic papers.**
Paste a title and abstract; get the journals whose published scope actually matches it.

<sub>Originally built as a team project with Team HackStreet. This repository is my continued work on it.</sub>

</div>

---

## Evaluation

Retrieval quality is measured against a **423-query ground-truth benchmark** built with no
human labelling: for each query, the "correct" journal is the one that *actually published*
that paper. Full harness and reproduction steps in [`Models/eval/`](Models/eval/).

| System | R@1 | R@5 | **R@10** | R@20 | MRR@10 |
|---|---|---|---|---|---|
| **SPECTER + topical document text** *(shipped)* | 12.1 | 37.6 | **51.1** | 63.8 | 0.233 |
| SPECTER, title-only document text | 12.5 | 30.7 | 39.2 | 50.4 | 0.203 |
| TF-IDF lexical baseline | 11.6 | 30.7 | 40.0 | 50.8 | 0.194 |
| Hybrid RRF (semantic + lexical) | 13.5 | 36.9 | 48.5 | 62.9 | 0.235 |

*423 queries · 218 distinct gold journals · 1,000-journal corpus · random baseline R@10 = 1.0%*

**Two findings worth stating plainly:**

**1. The document representation was the whole game — Recall@10 went 39.2 → 51.1.**
OpenAlex had removed the `description` and `abbreviated_title` fields from source records.
They were empty for 100% of 1,000 sampled journals, so the index was silently embedding
journal *titles alone*. Rebuilding the indexed text from the `topics` field instead
recovered 12 points. Nothing about the model or the search changed.

**2. The hybrid scored 2.6 points *worse* than semantic-only, so it was not shipped.**
Reciprocal-rank fusion of the dense and lexical rankings won on R@1 and MRR but lost on
R@10 and R@20. For a tool that shows a shortlist, R@10 is the metric that matters, so the
shipped system is semantic-only. The losing experiment is kept in the benchmark rather than
deleted.

### Performance

| | |
|---|---|
| Cold query latency (p50 / p95) | `153.1 ms` / `188.9 ms` |
| FAISS index scan alone (p50) | `0.1 ms` |
| App cold start | `~20 s` (torch import + SPECTER load) |
| Index load at startup | `7.8 ms` |
| Embedding dimension | 768 |

Embedding the 1,000 journals takes ~80 s, so it is done **offline** by
`build_index.py` and the result is committed as a 2.9 MB artifact. It used to run
inside the request handler, which made the first query of every cold process wait
~150 s. Embedding a fixed corpus is a build step, not a request step.

The index scan is **~0.1% of query time** — the SPECTER encoder dominates completely. That
is why the index is an exact `IndexFlatIP` rather than an approximate one: at this corpus
size an ANN index would trade recall away for time that isn't being spent there anyway.

A batching ablation is also included: batch=1 → batch=32 gives only a **1.14×** speedup on
200 documents, because the encoder is compute-bound rather than overhead-bound.

---

## Limitations

Stated up front, because they bound what the numbers mean.

- **The corpus is 1,000 journals out of ~206,000 in OpenAlex**, selected by descending
  citation count. That is a deliberate trade for index build time, but it means the long
  tail of niche and regional venues is **not represented** — which is precisely the case a
  researcher most needs help with. Scaling the corpus is the main outstanding work.
- **Recall@1 of 12.1% understates real-world quality**, and the ground truth is why. The
  label is the venue a paper was *actually* published in — but for most papers several
  journals would have been a perfectly good fit, and the author picked one for reasons
  unrelated to topical match: turnaround time, open-access policy, a prior relationship
  with an editor. Every appropriate-but-not-chosen venue is scored as a miss. Recall@10 is
  the honest metric for a shortlisting tool.
- `2yr_mean_citedness` is **OpenAlex's own measure**, not the Clarivate Journal Impact
  Factor. It is the same definition, computed by a different body, and is not presented as
  a JIF.

---

## How it works

```
OFFLINE  ·  build_index.py                    ← run once, output committed
  OpenAlex /sources ──▶ document text ──▶ SPECTER (768-d) ──▶ L2-normalise
  type:journal          name | publisher                          │
  sort by citations     | topics                                  ▼
                                                    index/embeddings.npy (2.9 MB)
                                                    index/manifest.json  (digest)

SERVING  ·  model2.py
  startup:  load embeddings.npy ──▶ FAISS IndexFlatIP     (7.8 ms)
            verify manifest digest vs corpus              (fail loud on drift)
                                                                │
  request:  title + abstract ──▶ SPECTER ──▶ L2-normalise ──────┤
                                                                ▼
                                             search ──▶ filter ──▶ ranked journals
```

- **SPECTER** (`allenai-specter`) — embeddings trained on citation relationships between
  scientific papers, so documents that cite each other land near each other. That is the
  right prior for "which venue publishes work like this".
- **FAISS `IndexFlatIP`** — vectors are L2-normalised, so inner product *is* cosine
  similarity. Exact search, for the reason given above.
- **Metrics are real OpenAlex values** — a snapshot taken when the index was built,
  which the UI states. Nothing is synthesised.

---

## Running it

```bash
cd Models
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run model2.py
```

The corpus and its embeddings ship with the repo, so there is no build step and no API
key to configure. `build_index.py` regenerates them if you change the corpus.

Full instructions, including the other two Streamlit tools, in
[`SETUP_GUIDE.md`](SETUP_GUIDE.md). Architecture and known issues in
[`ARCHITECTURE.md`](ARCHITECTURE.md).

### Reproducing the benchmark

```bash
cd Models/eval
python verify_fixture.py    # assert the corpus fixture matches the eval set
python bench.py             # rebuilds results.json
```

`journals.json` and `texts.json` are committed as a **pinned fixture**. They are not
regenerated, because `fetch_corpus.py` sorts by citation count and citation counts change
daily — a fresh fetch silently reorders the corpus, after which the positional gold labels
in `eval_set.json` point at the wrong journals and the benchmark reports confidently wrong
numbers without erroring. `verify_fixture.py` exists to make that failure loud.

---

## Project layout

| Path | What |
|---|---|
| `Models/model2.py` | The recommender (Streamlit) |
| `Models/build_index.py` | Offline embedding build → `Models/index/` |
| `Models/eval/` | Benchmark harness, eval set, pinned corpus fixture |
| `Models/dashboard.py` | Bibliometric charts for a topic (standalone tool) |
| `Models/keywordFinder.py` | Paper search + keyword extraction (standalone tool) |
| `src/` | React landing page |

---

## Links

<div align="center">
  <a href="https://journalsense.vercel.app/"><img src="https://img.shields.io/badge/Live_Site-00B0FF?style=for-the-badge&logoColor=white" alt="Live Site"></a>
  <a href="https://www.youtube.com/watch?v=uFNCtUuNHvA"><img src="https://img.shields.io/badge/▶️_Demo_Video-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="Demo Video"></a>
  <a href="https://drive.google.com/file/d/1rudH1-UmFbe6DF0D5w5_6G7qrn_397dz/view?usp=sharing"><img src="https://img.shields.io/badge/📄_Info_Deck-blue?style=for-the-badge&logo=google-drive&logoColor=white" alt="Info Deck"></a>
</div>

> The live site is currently a landing page. It does not yet call the recommender —
> wiring it to an API is the next phase of work.

---

## Built with

`sentence-transformers` (SPECTER) · `faiss-cpu` · `scikit-learn` · `streamlit` ·
`pandas` · `plotly` · OpenAlex API

<div align="center">
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="">
</div>
