# JournalSense — Setup and Run Guide

## Prerequisites

- **Python** 3.12 (3.9+ should work; the pins are verified against 3.12)
- **Node.js** v16 or higher
- `pip` and `npm`

---

## Part 1 — Backend (Python / Streamlit)

### 1. Create and activate a virtual environment

```bash
cd Models
python -m venv venv
```

```bash
# Windows
venv\Scripts\activate
```

```bash
# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Every package is pinned to an exact version. If you change one, change the pin — an
unpinned dependency makes the benchmark unreproducible and the container unrebuildable.

> **Note:** there is no spaCy model to download. spaCy was removed — its pinned model was a
> 2.x build running against a 3.x runtime, so it never loaded, and its only consumer
> (display-only key-phrase extraction) now uses scikit-learn instead.

### 3. Configure environment

```bash
cp ../.env.example ../.env
```

Set `OPENALEX_MAILTO` to your email. OpenAlex uses it to put your requests in the "polite
pool", which gets better rate limits. It is not a secret. If you leave it unset the apps
still work — they just omit the parameter rather than sending a fake address.

### 4. Run an app

```bash
streamlit run model2.py
```

| App | Command | What it is |
|---|---|---|
| **Journal recommender** | `streamlit run model2.py` | The product |
| Scholarly dashboard | `streamlit run dashboard.py` | Bibliometric charts for a topic |
| Keyword explorer | `streamlit run keywordFinder.py` | Paper search + keyword extraction |

Default port is `8501`. Use `--server.port 8502` to change it.

> **First run is slow.** The SPECTER model downloads (~440 MB) and loads in ~52 s, then the
> FAISS index builds in ~80 s. Both are cached for the life of the process, so subsequent
> queries are fast — but note that the index build currently happens *inside* the request
> handler, so the first query after a cold start pays the whole cost. Moving it to an
> offline build step is the next phase of work.

---

## Part 2 — Frontend (React / Vite)

```bash
npm install
npm run dev
```

Starts on `http://localhost:5173`.

This is currently a **landing page only**. It makes no network calls and does not reach the
recommender. `VITE_API_BASE_URL` is defined in `.env.example` for when it does.

---

## Part 3 — Reproducing the benchmark

```bash
cd Models/eval
python verify_fixture.py
```

This must pass before you trust anything `bench.py` prints. It asserts that the committed
corpus fixture still matches the gold labels in `eval_set.json`.

```bash
python bench.py
```

Rebuilds `results.json`. Takes several minutes — it loads SPECTER and builds two full
indexes.

> **Do not run `fetch_corpus.py` unless you mean to.** It overwrites `journals.json` and
> `texts.json`, which are committed as a pinned fixture. `fetch_corpus.py` sorts by citation
> count, citation counts change daily, and a fresh fetch reorders the corpus — after which
> the positional gold labels in `eval_set.json` point at the wrong journals and `bench.py`
> reports wrong numbers **without erroring**. Run `verify_fixture.py` if you suspect drift;
> `git checkout Models/eval/journals.json Models/eval/texts.json` restores the pin.

---

## Troubleshooting

**`ValueError: numpy.dtype size changed` on importing sklearn**
A scikit-learn built against the numpy 1.x C ABI is installed alongside numpy 2.x. Install
the pinned versions: `pip install -r requirements.txt --force-reinstall`.

**FAISS install fails**
`pip install faiss-cpu`. There is no official Windows build of GPU FAISS.

**Port already in use**
Streamlit: `--server.port 8502`. Vite: `npm run dev -- --port 3000`.

**Module not found**
Confirm the virtualenv is activated, then reinstall from `requirements.txt`.

---

## Project structure

```
JournalSense/
├── Models/
│   ├── model2.py           # the recommender
│   ├── dashboard.py        # bibliometric charts (standalone)
│   ├── keywordFinder.py    # keyword explorer (standalone)
│   ├── requirements.txt    # exact pins
│   └── eval/
│       ├── fetch_corpus.py     # builds the corpus (do not run casually)
│       ├── build_eval.py       # builds ground truth from real publication venues
│       ├── bench.py            # the benchmark
│       ├── verify_fixture.py   # guard: fixture vs gold labels
│       ├── journals.json       # PINNED corpus fixture
│       ├── texts.json          # PINNED indexed text, aligned to journals.json
│       ├── eval_set.json       # 423 queries + gold labels
│       └── results.json        # benchmark output
├── src/                    # React landing page
├── ARCHITECTURE.md         # data flow, state, known issues
└── .env.example
```
