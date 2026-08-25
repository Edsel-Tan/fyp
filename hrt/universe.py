#!/usr/bin/env python3
"""Reconstruct point-in-time S&P 500 membership from Wikipedia.

Walks the "Historical components of the S&P 500" change log backwards from the
current constituent list to recover membership on any past date.

Outputs hrt/artifacts/universe.json with:
  current   -- today's constituents
  asof_YYYY-MM-DD -- reconstructed membership on that date
"""
import json, os, re, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
CACHE = os.path.join(ART, "wiki")
UA = {"User-Agent": "hrt-repro/1.0 (academic reproduction)"}

PAGES = {
    "current": "List_of_S%26P_500_companies",
    "changes": "Historical_components_of_the_S%26P_500",
}

MONTHS = {m: i + 1 for i, m in enumerate(
    "January February March April May June July August September October "
    "November December".split())}


def wikitext(name):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name + ".wiki")
    if not os.path.exists(path):
        url = f"https://en.wikipedia.org/w/index.php?title={PAGES[name]}&action=raw"
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            open(path, "wb").write(r.read())
    return open(path, encoding="utf-8").read()


def strip(cell):
    """Wikitext cell -> plain ticker string."""
    c = re.sub(r"<ref.*?(/>|</ref>)", "", cell, flags=re.S)
    c = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", c)
    c = re.sub(r"\[\[([^\]]*)\]\]", r"\1", c)
    # ticker cells are wrapped in {{NyseSymbol|MMM}} / {{NasdaqSymbol|AAPL}}
    c = re.sub(r"\{\{[A-Za-z ]*Symbol\|([^}|]+)\}\}", r"\1", c)
    c = re.sub(r"\{\{[^}]*\}\}", "", c)
    c = re.sub(r"<[^>]+>", "", c)
    return c.replace("'''", "").strip()


def parse_date(s):
    s = strip(s)
    m = re.match(r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})", s)
    if m:
        return f"{int(m.group(3)):04d}-{MONTHS[m.group(1)]:02d}-{int(m.group(2)):02d}"
    m = re.match(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", s)
    if m:
        return f"{int(m.group(3)):04d}-{MONTHS[m.group(2)]:02d}-{int(m.group(1)):02d}"
    return None


def table_rows(text, anchor):
    """Yield lists of plain-text cells for the wikitable containing `anchor`."""
    i = text.find(anchor)
    start = text.rfind("{|", 0, i)
    body = text[start:text.find("\n|}", start)]
    # refs carry templates full of pipes -- drop them before splitting on '|'
    body = re.sub(r"<ref[^>]*/>", "", body)
    body = re.sub(r"<ref.*?</ref>", "", body, flags=re.S)

    for chunk in re.split(r"\n\|-[^\n]*", body)[1:]:
        cells, cur = [], None
        for line in chunk.split("\n"):
            line = line.rstrip()
            if line.startswith("!") or line.startswith("|}") or line.startswith("|+"):
                continue
            if line.startswith("|"):
                if cur is not None:
                    cells.append(cur)
                cur = line.lstrip("|")
                parts = cur.split("||")
                if len(parts) > 1:
                    cells.extend(parts[:-1])
                    cur = parts[-1]
            elif cur is not None:
                cur += "\n" + line
        if cur is not None:
            cells.append(cur)
        cells = [strip(c) for c in cells]
        if any(cells):
            yield cells


def current_constituents():
    text = wikitext("current")
    out = {}
    for cells in table_rows(text, 'id="constituents"'):
        if len(cells) >= 6 and re.fullmatch(r"[A-Z][A-Z.\-]{0,6}", cells[0]):
            out[cells[0]] = {"security": cells[1], "sector": cells[2],
                             "date_added": parse_date(cells[5])}
    return out


def changes():
    text = wikitext("changes")
    out = []
    for cells in table_rows(text, 'id="changes"'):
        if len(cells) < 5:
            continue
        d = parse_date(cells[0])
        if not d:
            continue
        added, removed = cells[1], cells[3]
        ok = lambda t: bool(re.fullmatch(r"[A-Z][A-Z.\-]{0,6}", t))
        out.append({"date": d,
                    "added": added if ok(added) else None,
                    "removed": removed if ok(removed) else None})
    out.sort(key=lambda r: r["date"])
    return out


def membership_asof(asof, cur, chg):
    """Rewind today's membership back to `asof` (inclusive)."""
    members = set(cur)
    unresolved = []
    for row in reversed([c for c in chg if c["date"] > asof]):
        # undo: a stock added after `asof` was not a member; a removed one was
        if row["added"]:
            members.discard(row["added"])
        if row["removed"]:
            members.add(row["removed"])
        if not row["added"] and not row["removed"]:
            unresolved.append(row["date"])
    return sorted(members), unresolved


def main():
    cur, chg = current_constituents(), changes()
    os.makedirs(ART, exist_ok=True)
    out = {"current": sorted(cur), "n_changes": len(chg),
           "changes_span": [chg[0]["date"], chg[-1]["date"]]}
    for asof in ("2015-01-01", "2021-01-01", "2022-01-01"):
        mem, unres = membership_asof(asof, cur, chg)
        out[f"asof_{asof}"] = mem
        print(f"{asof}: {len(mem)} members  (unparsed change rows after date: {len(unres)})")
    print(f"current: {len(cur)}  changes parsed: {len(chg)} "
          f"({chg[0]['date']} .. {chg[-1]['date']})")
    json.dump(out, open(os.path.join(ART, "universe.json"), "w"), indent=1)
    print("wrote", os.path.join(ART, "universe.json"))


if __name__ == "__main__":
    main()
