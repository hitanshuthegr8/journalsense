"""
Ground-truth eval set: real papers whose ACTUAL publication venue is in our index.
Query = paper title + abstract.  Relevant doc = the journal that published it.
No human labelling required -- the label is where the paper really appeared.
"""
import json, random, time, requests

random.seed(42)
journals = json.load(open("journals.json", encoding="utf-8"))
by_id = {j["id"]: i for i, j in enumerate(journals)}

def deinvert(inv):
    if not inv: return ""
    pos = [(p, w) for w, ps in inv.items() for p in ps]
    pos.sort()
    return " ".join(w for _, w in pos)

sample = random.sample(journals, 220)
evals, calls = [], 0
t0 = time.perf_counter()

for j in sample:
    sid = j["id"].rsplit("/", 1)[-1]
    try:
        r = requests.get("https://api.openalex.org/works", timeout=30, params={
            "filter": f"primary_location.source.id:{sid},has_abstract:true,type:article",
            "per-page": 2, "sort": "cited_by_count:desc",
            "select": "id,title,abstract_inverted_index,primary_location",
        })
        calls += 1
        if r.status_code != 200: continue
        for w in r.json().get("results", []):
            abs_txt = deinvert(w.get("abstract_inverted_index"))
            title = w.get("title") or ""
            if len(abs_txt) < 200 or not title:      # need a substantive query
                continue
            src = (w.get("primary_location") or {}).get("source") or {}
            if src.get("id") not in by_id:
                continue
            evals.append({
                "query": f"{title}. {abs_txt[:1200]}",
                "gold_journal_id": src["id"],
                "gold_journal_idx": by_id[src["id"]],
                "gold_journal_name": src.get("display_name"),
            })
    except Exception:
        continue

elapsed = time.perf_counter() - t0
gold_set = len({e["gold_journal_id"] for e in evals})
print(json.dumps({
    "eval_queries": len(evals),
    "distinct_gold_journals": gold_set,
    "api_calls": calls,
    "build_seconds": round(elapsed, 1),
    "mean_query_chars": round(sum(len(e["query"]) for e in evals) / max(1, len(evals))),
}, indent=2))
json.dump(evals, open("eval_set.json", "w", encoding="utf-8"))
