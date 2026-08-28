"""Guard: assert the committed corpus fixture still matches the eval set's gold labels.

WHY THIS EXISTS
---------------
`eval_set.json` identifies the correct journal for each query in two ways:

    gold_journal_id   -> a stable OpenAlex source ID  (e.g. https://openalex.org/S137773608)
    gold_journal_idx  -> a POSITION in journals.json  (e.g. 417)

`bench.py` reads the position. But `journals.json` is produced by `fetch_corpus.py`,
which sorts by `cited_by_count:desc` - and citation counts change daily. Re-fetching
the corpus silently reorders it, after which every position points at the wrong journal
and the benchmark reports confidently wrong numbers. Nothing crashes.

When this was first checked against a freshly fetched corpus, 126 of 423 gold labels
(30%) had drifted, by between -5 and +2 positions.

That is why `journals.json` and `texts.json` are committed as a PINNED FIXTURE rather
than regenerated. This script exists so that the pin can never break silently: run it
before trusting any number that comes out of bench.py.

USAGE
-----
    python verify_fixture.py          # exits 0 if aligned, 1 if not

Run it in CI alongside the eval gate.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
REQUIRED = ["journals.json", "texts.json", "eval_set.json"]


def main() -> int:
    missing = [f for f in REQUIRED if not (HERE / f).exists()]
    if missing:
        print(f"FAIL: fixture files missing: {', '.join(missing)}")
        print("      Regenerate with `python fetch_corpus.py`, but note that a fresh")
        print("      fetch will NOT match the committed eval_set.json positions.")
        return 1

    journals = json.loads((HERE / "journals.json").read_text(encoding="utf-8"))
    texts = json.loads((HERE / "texts.json").read_text(encoding="utf-8"))
    evals = json.loads((HERE / "eval_set.json").read_text(encoding="utf-8"))

    if len(texts) != len(journals):
        print(f"FAIL: texts.json has {len(texts)} rows, journals.json has {len(journals)}.")
        print("      These are positionally aligned and must be regenerated together.")
        return 1

    by_id = {j["id"]: i for i, j in enumerate(journals)}

    aligned, drifted, absent = 0, [], 0
    for e in evals:
        gid, gidx = e["gold_journal_id"], e["gold_journal_idx"]
        if gid not in by_id:
            absent += 1
            continue
        if by_id[gid] == gidx:
            aligned += 1
        else:
            drifted.append((e.get("gold_journal_name"), gidx, by_id[gid]))

    print(f"corpus size          : {len(journals)}")
    print(f"eval queries         : {len(evals)}")
    print(f"gold labels aligned  : {aligned}")
    print(f"gold labels drifted  : {len(drifted)}")
    print(f"gold labels absent   : {absent}")

    if drifted or absent:
        print()
        print("FAIL: the fixture no longer matches eval_set.json.")
        print("      bench.py resolves gold labels BY POSITION, so it would report")
        print("      wrong numbers without erroring. Do not trust its output.")
        for name, was, now in drifted[:5]:
            print(f"        - {name!r}: expected index {was}, found at {now}")
        if len(drifted) > 5:
            print(f"        ... and {len(drifted) - 5} more")
        print()
        print("      Fix: restore the committed journals.json/texts.json (git checkout),")
        print("      or rebuild the eval set against the new corpus with build_eval.py.")
        return 1

    print()
    print("OK: fixture is aligned. bench.py results are trustworthy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
