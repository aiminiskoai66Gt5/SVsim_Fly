"""Refresh i18n/cards_zh_tw.json from the official card list API.

    python scripts/fetch_official_cards.py [--lang cht] [--out i18n/cards_zh_tw.json]

The official site (shadowverse-wb.com) serves card data from
``/web/CardList/cardList?offset=N`` (30 ids per page, related cards included in
``card_details``) and selects the language with a ``lang`` request header
(``cht`` = Traditional Chinese, ``en``, ``ja``...). Only names, stats and ability
text are stored; no images. Requires the ``requests`` package.
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://shadowverse-wb.com"


def fetch_all(lang: str, session) -> dict:
    out, offset, total, pages = {}, 0, None, 0
    while total is None or offset < total:
        r = session.get(f"{BASE}/web/CardList/cardList", params={"offset": offset},
                        headers={"lang": lang, "Accept": "application/json"}, timeout=30)
        r.raise_for_status()
        d = r.json()["data"]
        total = d["count"]
        for cid, det in (d.get("card_details") or {}).items():
            c = det["common"]
            out[str(cid)] = {"name": c["name"], "cost": c["cost"], "atk": c["atk"], "life": c["life"],
                             "type": c["type"], "class": c["class"], "is_token": bool(c.get("is_token")),
                             "card_set_id": c.get("card_set_id"), "skill_text": c.get("skill_text"),
                             "evo_skill_text": (det.get("evo") or {}).get("skill_text")}
        step = len(d.get("sort_card_id_list") or [])
        if not step:
            break
        offset += step
        pages += 1
        print(f"[fetch] {lang}: {offset}/{total}", file=sys.stderr)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="cht")
    ap.add_argument("--out", default=os.path.join(ROOT, "i18n", "cards_zh_tw.json"))
    args = ap.parse_args(argv)
    import requests
    s = requests.Session()
    zh = fetch_all(args.lang, s)
    en = fetch_all("en", s)
    merged = {}
    for cid in sorted(zh):
        c = zh[cid]
        merged[cid] = {"name_zh_tw": c["name"], "name_en": en.get(cid, {}).get("name"), "cost": c["cost"],
                       "atk": c["atk"], "life": c["life"], "type": c["type"], "class": c["class"],
                       "is_token": c["is_token"], "card_set_id": c["card_set_id"],
                       "skill_text_zh_tw": c["skill_text"], "evo_skill_text_zh_tw": c["evo_skill_text"]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=0)
    print(f"wrote {args.out}: {len(merged)} cards")
    return 0


if __name__ == "__main__":
    sys.exit(main())
