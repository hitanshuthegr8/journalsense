import json, statistics, time
import faiss, numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

R = {}
journals = json.load(open("journals.json", encoding="utf-8"))
texts    = json.load(open("texts.json", encoding="utf-8"))
evals    = json.load(open("eval_set.json", encoding="utf-8"))
# reproduce the ORIGINAL model2.py indexed text (title + empty fields)
old_texts = [f"{j['display_name']} — {j.get('abbreviated_title','') or ''}\nScope: {j.get('description','') or ''}"
             for j in journals]
print(f"corpus={len(texts)} evals={len(evals)}")

print("loading allenai-specter ...")
t0 = time.perf_counter(); model = SentenceTransformer("allenai-specter")
R["model_load_seconds"] = round(time.perf_counter()-t0, 1)
print(f"  loaded in {R['model_load_seconds']}s")

def embed(ts, bs):
    e = model.encode(ts, batch_size=bs, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(e, dtype=np.float32)

def build(ts, bs=32):
    t0 = time.perf_counter()
    embs = embed(ts, bs)
    faiss.normalize_L2(embs)
    idx = faiss.IndexFlatIP(embs.shape[1]); idx.add(embs)
    return idx, embs, time.perf_counter()-t0

# ---------- 1. index build + batching ablation ----------
print("building enriched index (batch=32) ...")
idx_new, emb_new, t_new = build(texts, 32)
R["embedding_dim"] = int(emb_new.shape[1])
R["index_build_seconds_batch32"] = round(t_new, 2)
R["index_docs_per_sec_batch32"] = round(len(texts)/t_new, 1)
print(f"  {t_new:.2f}s  ({R['index_docs_per_sec_batch32']} docs/s)")

SUB = texts[:200]
_,_,t_b32 = build(SUB, 32)
_,_,t_b1  = build(SUB, 1)
R["batching_ablation_200docs"] = {"batch1_s": round(t_b1,2), "batch32_s": round(t_b32,2),
                                  "speedup_x": round(t_b1/t_b32, 2)}
print(f"  batching: {t_b1:.2f}s -> {t_b32:.2f}s = {t_b1/t_b32:.2f}x")

print("building title-only (original) index ...")
idx_old, _, t_old = build(old_texts, 32)

# ---------- 2. latency ----------
qs = [e["query"] for e in evals[:100]]
def q_latency(index, queries, cache=None):
    out=[]
    for q in queries:
        t0=time.perf_counter()
        if cache is not None and q in cache: v=cache[q]
        else:
            v=embed([q],1); faiss.normalize_L2(v)
            if cache is not None: cache[q]=v
        index.search(v, 30)
        out.append((time.perf_counter()-t0)*1000)
    return out
def s(v): 
    v=sorted(v); return {"p50":round(statistics.median(v),1),
                         "p95":round(v[int(len(v)*0.95)],1),"mean":round(statistics.mean(v),1)}
cold = q_latency(idx_new, qs)
cache={}; q_latency(idx_new, qs, cache); warm = q_latency(idx_new, qs, cache)
R["query_latency_cold_ms"]=s(cold); R["query_latency_warm_ms"]=s(warm)
R["cache_speedup_x"]=round(statistics.mean(cold)/statistics.mean(warm),1)
print(f"  cold p50={R['query_latency_cold_ms']['p50']}ms  warm p50={R['query_latency_warm_ms']['p50']}ms  ({R['cache_speedup_x']}x)")

v=embed([qs[0]],1); faiss.normalize_L2(v)
scan=[]
for _ in range(200):
    t0=time.perf_counter(); idx_new.search(v,30); scan.append((time.perf_counter()-t0)*1000)
R["faiss_scan_only_ms"]=s(scan)

# ---------- 3. retrieval quality ----------
gold = np.array([e["gold_journal_idx"] for e in evals])
print("encoding eval queries ...")
Q = embed([e["query"] for e in evals], 32); faiss.normalize_L2(Q)

def ranks_from_scores(S):
    order = np.argsort(-S, axis=1)
    return np.array([np.where(order[i]==gold[i])[0][0] if gold[i] in order[i] else 10**6
                     for i in range(len(gold))])
def metrics(rk):
    return {"recall@1":round(float((rk<1).mean()*100),1),
            "recall@5":round(float((rk<5).mean()*100),1),
            "recall@10":round(float((rk<10).mean()*100),1),
            "recall@20":round(float((rk<20).mean()*100),1),
            "MRR@10":round(float(np.mean([1/(r+1) if r<10 else 0 for r in rk])),3)}

S_new = Q @ emb_new.T
R["dense_specter_enriched"] = metrics(ranks_from_scores(S_new))
emb_old = np.asarray(idx_old.reconstruct_n(0, idx_old.ntotal))
R["dense_specter_titleonly_ORIGINAL"] = metrics(ranks_from_scores(Q @ emb_old.T))

vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), sublinear_tf=True, min_df=1)
D = normalize(vec.fit_transform(texts)); Qv = normalize(vec.transform([e["query"] for e in evals]))
S_lex = np.asarray((Qv @ D.T).todense())
R["lexical_tfidf_enriched"] = metrics(ranks_from_scores(S_lex))

def rrf(mats, k=60):
    tot = np.zeros_like(mats[0])
    for M in mats:
        order = np.argsort(-M, axis=1)
        rr = np.empty_like(order)
        for i in range(order.shape[0]): rr[i, order[i]] = np.arange(order.shape[1])
        tot += 1.0/(k+rr+1)
    return tot
R["hybrid_rrf_dense_plus_lexical"] = metrics(ranks_from_scores(rrf([S_new, S_lex])))
R["eval_set"] = {"queries": len(evals), "distinct_gold_journals": len(set(gold.tolist())),
                 "corpus_size": len(texts), "random_baseline_recall@10_pct": round(1000/len(texts)*10/len(texts)*100,3)}

json.dump(R, open("results.json","w"), indent=2)
print("\n"+json.dumps(R, indent=2))
