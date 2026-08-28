# JournalSense — Architecture

Semantic journal recommender. Given a paper title + abstract, it returns the academic
journals whose published scope is closest to that paper, using SPECTER sentence embeddings
and a FAISS similarity index over journal metadata fetched live from OpenAlex.

> Status note: the repo contains **three independent Streamlit apps and one React landing
> page that never communicate**. There is no HTTP API and no shared library. Each Streamlit
> app re-implements its own OpenAlex client.
>
> Section 5 lists findings from the initial read. Items marked **[FIXED]** were resolved in
> the hygiene pass; the rest are open.

---

## 1. Entry points

| Entry point | Launch | What it is |
|---|---|---|
| `Models/model2.py` | `streamlit run Models/model2.py` | **The product.** Journal recommender. |
| `Models/dashboard.py` | `streamlit run Models/dashboard.py` | Bibliometric charts for a topic. Standalone. |
| `Models/keywordFinder.py` | `streamlit run Models/keywordFinder.py` | Paper search + keyword extraction. Standalone. |
| `Models/eval/fetch_corpus.py` | `python fetch_corpus.py` | Offline. Writes `journals.json`, `texts.json`, `corpus_stats.json`. |
| `Models/eval/build_eval.py` | `python build_eval.py` | Offline. Writes `eval_set.json`. Requires `journals.json`. |
| `Models/eval/bench.py` | `python bench.py` | Offline. Writes `results.json`. Requires all three JSONs above. |
| `Models/eval/verify_fixture.py` | `python verify_fixture.py` | Offline guard. Asserts the pinned corpus still matches the gold labels. Exit 1 on drift. |
| `src/` (root React) | `npm run dev` | Landing page. No backend calls. |

`.devcontainer/devcontainer.json` auto-launches `model2.py` on port 8501 in Codespaces.

---

## 2. Data flow — the recommender

```
 OpenAlex /sources                     model2.py:65  fetch_openalex_journals()
   filter=type:journal                 cursor-paginated, 200/page, target 1000
   sort=cited_by_count:desc            @st.cache_data(ttl=3600)
        |
        v
 journal_index_text(j)                 model2.py:87
   "<name> | <alt titles> | <publisher> | Publishes research on: <up to 12 topics>"
        |                              <- this string is what actually gets embedded
        v
 SentenceTransformer("allenai-specter")  model2.py:47  @st.cache_resource(ttl=24h)
   encode(batch_size=32) -> (1000, 768) float32
        |
        v
 faiss.normalize_L2  ->  IndexFlatIP     model2.py:119  build_faiss_index()
   L2-normalised, so inner product == cosine similarity.  @st.cache_resource
        |
        v
 QUERY: f"{title} {abstract}" -> encode -> normalize_L2     model2.py:180
        |
        v
 index.search(q, min(top_k*3, ntotal))  <- 3x over-fetch, because the domain
        |                                  filter is applied AFTER retrieval
        v
 post-filter: domains -> then in the UI loop: citedness / DOAJ / OA
        |
        v
 ranked list: title, publisher, ISSN-L, url, domains, topics, score, metrics
```

**Two representations of the same journal record exist** and must stay in sync:
`model2.py:journal_index_text()` (serving) and `eval/fetch_corpus.py:index_text()` (eval).
They are currently identical in behaviour but are duplicated source.

---

## 3. Where state lives / what is recomputed

Nothing persists to disk in the serving path. All state is Streamlit in-memory cache,
scoped to the server process and lost on restart.

| Thing | Mechanism | Cost | Recomputed when |
|---|---|---|---|
| Journal metadata (1000 records) | `@st.cache_data(ttl=3600)` | `16.24 s` fetch* | hourly, or on restart |
| SPECTER model | `@st.cache_resource(ttl=24h)` | `51.6 s` load* | daily, or on restart |
| FAISS index (1000 x 768) | `@st.cache_resource` | `80.1 s` build* | when the journals list changes, or on restart |
| **Query embedding** | **none** | — | **every single query** |
| dashboard / keywordFinder results | `@st.cache_data(ttl=3600)` / `st.session_state` | — | per session |

\* measured values taken from `Models/eval/corpus_stats.json` and `Models/eval/results.json`.

**The critical structural fact:** `build_faiss_index()` is called at `model2.py:246` —
*inside the button handler*, after the user clicks. On a cold process the first query pays
model load + index build before it can answer. The `153.1 ms` p50 in `results.json` is
warm-process latency only, and `faiss_scan_only_ms` is `0.1 ms` — the search is negligible
and the encoder dominates.

---

## 4. Modules

**`Models/model2.py`** — the recommender. Fetches sources, builds the embedded text, builds
the index, embeds the query, searches, filters, renders. Also does display-only key-phrase
extraction (`extract_key_phrases`), which feeds the "Key Topics" line and **has no
influence on ranking**. This used spaCy noun chunks; spaCy has been removed (finding 7)
and it now uses a scikit-learn n-gram frequency count. `journal_metrics()` (line 174) returns
only real OpenAlex `summary_stats` values, and the UI explicitly disclaims that
`2yr_mean_citedness` is not the Clarivate JIF. Header comments document three previously
fixed defects: OpenAlex removing `description`/`abbreviated_title`, the removal of
`x_concepts` breaking the domain filter, and a prior version returning `random.random()`
impact factors.

**`Models/dashboard.py`** — unrelated to the recommender. Topic + year range → OpenAlex
`/works` → Plotly charts. **[FIXED]** Previously contained four `generate_sample_*`
functions plus `safe_visualize`, which substituted fabricated data on any failure. These are
removed; missing data now renders an explicit empty state.

**`Models/keywordFinder.py`** — unrelated to the recommender. Three tabs (Search / Extract
Keywords / Paper Explorer) wired together through `st.session_state`. Three swappable
extraction backends: TF-IDF, YAKE, and `all-MiniLM-L6-v2` embeddings (a KeyBERT-style
candidate-ranking approach). Renders a wordcloud and re-ranks results by TF-IDF similarity
to a chosen keyword.

**`Models/eval/`** — the offline benchmark, and the strongest work in the repo.
`fetch_corpus.py` pulls the corpus. `build_eval.py` builds ground truth with **no human
labelling**: sample 220 journals, pull their top-cited articles, and use "the journal this
paper actually appeared in" as the gold label. `bench.py` compares four systems on that set.
Per `results.json`: enriched-text SPECTER reaches `recall@10 = 51.1`, the original
title-only representation `39.2`, TF-IDF `40.0`, and hybrid RRF `48.5` — the hybrid scored
**worse** than semantic-only, which is why the shipped system is semantic-only. The eval set
is `423` queries over `218` distinct gold journals against a `1000`-journal corpus.

**Frontends** — `src/` is a static landing page (Hero / Features / Footer / canvas
animation). It issues no network request to any backend. **[FIXED]** The second React app
(`project/`, "MesmerizeAbstractBot") has been deleted along with its feature card.

---

## 5. Findings — wrong, unused, or duplicated

### Broken (verified by reading the code against the OpenAlex API contract)

1. **[FIXED] `bench.py` could not run as committed.** It opens `journals.json` and
   `texts.json` at lines 6–7; neither file existed in the repo. Both are now committed as a
   pinned fixture.
2. **[PARTLY FIXED] `eval_set.json` stores `gold_journal_idx`, a positional index into
   `journals.json`.** Refetching the corpus reorders it (citation counts change daily), after
   which every gold index silently points at the wrong journal. Measured on a fresh fetch:
   **126 of 423 gold labels (30%) had drifted**, by -5 to +2 positions. `bench.py` still uses
   the positional index rather than the stable `gold_journal_id` present in the same record —
   that remains unfixed, per the constraint not to modify eval logic. Mitigated two ways: the
   corpus is committed as a pinned fixture, and `verify_fixture.py` fails loudly on drift.
3. **[FIXED] `dashboard.py:112` and `keywordFinder.py:315` send `per_page`.** The OpenAlex parameter
   is `per-page` (hyphen). `model2.py:72` and both eval scripts use the correct spelling.
   Effect: the "Number of Papers to Analyze" slider and `per_page=25` are both ignored and
   the API returns its default page size.
4. **[FIXED] `dashboard.py:171` reads `authorship['institution']`.** OpenAlex authorships expose
   `institutions` (plural, a list). The key is never present, so the institutions list is
   always empty and the chart **always** falls back to `generate_sample_institution_data()`
   — hardcoded Harvard/Stanford/MIT rows — even when the API call succeeded.
5. **`keywordFinder.py:349` reads `work.get("abstract", ...)`.** OpenAlex works carry
   `abstract_inverted_index`, not `abstract`; `eval/build_eval.py:14` correctly de-inverts
   it. So every paper reports "No abstract available" and all keyword extraction silently
   degrades to the title-only fallback path.
6. **`keywordFinder.py:98` shadows the parameter `n`.** `n` is the requested keyword count
   (default 20), but the n-gram loop `for n in range(...)` rebinds it to `2`, so line 131's
   `[:n]` returns 2 keywords. The inner `break` at line 105 also only exits the inner loop,
   so the 200-candidate cap never stops the outer one.
7. **[FIXED] The spaCy model pin is incompatible.** `requirements.txt` pins `en_core_web_sm-2.2.0`
   (a spaCy 2.x model) against a 3.x runtime. `load_spacy_model()` catches the failure and
   falls back to `spacy.blank("en")`, which has **no noun-chunk parser** — so
   `extract_key_phrases` silently drops to its crude word-pair fallback with no warning.
8. **[FIXED] `model2.py:38` shells out to `pip`/`spacy download` at runtime.** Downloading a package
   from inside a request path will fail or produce non-reproducible images once containerised.

### Fabricated data rendered as real

9. **[FIXED]** `project/src/services/gemini.ts` — `await delay(3000)` plus string templating that emits
   invented statistics ("37% reduction in error rates", "p<0.001", "42% increase"). Also
   contains a **live Google API key on line 4**, present in commit `eb27460`.
10. **[FIXED]** `dashboard.py` — `generate_sample_data`, `generate_sample_oa_data`,
    `generate_sample_institution_data`, `generate_sample_concept_data`, and the fallbacks
    inside `safe_visualize` produce plausible trend lines that render identically to real
    ones. Per finding #4, one of these fires on every run.

### Unused / dead

11. `process_data()` computes and returns `publication_trends` and `citation_metrics`, but
    `main()` renders neither. Two of five computed datasets are discarded.
12. `Models/eval/corpus_stats.json` is written by `fetch_corpus.py` and read by nothing.

### Duplicated

14. **[FIXED]** Two React apps with byte-identical `vite.config.ts` and duplicated
    Tailwind/ESLint/tsconfig/postcss. Neither `package.json` is named `journalsense`.
15. The OpenAlex client is written three times (`model2.py`, `dashboard.py`,
    `keywordFinder.py`) with three different parameter conventions and three different
    error-handling styles. No shared module.
16. The embedded-text builder exists twice (`model2.py:87`, `eval/fetch_corpus.py:19`). If
    these ever drift, the benchmark stops measuring the shipped system.
17. **[FIXED]** A personal email address was hardcoded twice in `keywordFinder.py` (lines 308, 333) as the
    OpenAlex polite-pool address, while `dashboard.py:84` uses the placeholder
    `example@domain.com`, so that app never enters the polite pool at all.

### Error handling

18. **[PARTLY FIXED]** Bare `except:` at `keywordFinder.py:172` and `:417`. `except Exception: continue` at
    `build_eval.py:44`. A broad `except Exception` wrapping all of `main()` at
    `model2.py:295`. None distinguish "no results" from "the network failed".
