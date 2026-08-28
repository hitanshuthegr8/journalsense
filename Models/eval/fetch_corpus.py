"""Fetch a clean, impact-ranked journal corpus from OpenAlex /sources."""
import json, time, requests

SELECT = ",".join(["id","display_name","alternate_titles","issn_l","type",
                   "works_count","cited_by_count","homepage_url",
                   "host_organization_name","topics","summary_stats"])

def fetch(n=1000, per_page=200):
    out, cursor = [], "*"
    t0 = time.perf_counter()
    while len(out) < n:
        r = requests.get("https://api.openalex.org/sources", timeout=30, params={
            "filter": "type:journal",
            "sort": "cited_by_count:desc",
            "per-page": per_page, "cursor": cursor, "select": SELECT,
        })
        r.raise_for_status()
        d = r.json()
        out.extend(d["results"])
        cursor = d.get("meta", {}).get("next_cursor")
        if not cursor: break
    return out[:n], time.perf_counter() - t0, d["meta"]["count"]

def index_text(j):
    """Text that actually gets embedded: identity + topical scope."""
    topics = [t["display_name"] for t in (j.get("topics") or [])[:12]]
    alts = (j.get("alternate_titles") or [])[:2]
    parts = [j["display_name"]]
    if alts: parts.append(" / ".join(alts))
    if j.get("host_organization_name"): parts.append(j["host_organization_name"])
    if topics: parts.append("Publishes research on: " + "; ".join(topics))
    return " | ".join(parts)

if __name__ == "__main__":
    journals, secs, total = fetch()
    texts = [index_text(j) for j in journals]
    lens = sorted(len(t) for t in texts)
    empty_topics = sum(1 for j in journals if not j.get("topics"))
    stats = {
        "total_journals_in_openalex": total,
        "corpus_size": len(journals),
        "fetch_seconds": round(secs, 2),
        "journals_missing_topics": empty_topics,
        "index_text_chars_median": lens[len(lens)//2],
        "index_text_chars_p10": lens[int(len(lens)*0.1)],
        "index_text_chars_p90": lens[int(len(lens)*0.9)],
    }
    print(json.dumps(stats, indent=2))
    print("\nsample:")
    for t in texts[:3]: print("  -", t[:150])
    json.dump(journals, open("journals.json","w",encoding="utf-8"))
    json.dump(texts, open("texts.json","w",encoding="utf-8"))
    json.dump(stats, open("corpus_stats.json","w"), indent=2)
