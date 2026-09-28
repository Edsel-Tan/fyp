"""Link Polymarket LoL series/game markets to lolesports matches and games."""
import re, pandas as pd, numpy as np
from rapidfuzz import fuzz

def norm(s):
    s = s.lower().replace('dn freecs', 'dn soopers').replace('esports', '').replace('gaming', '').replace('team ', '').replace('e-sports', '')
    return re.sub(r'[^a-z0-9]+', ' ', s).strip()

pm = pd.read_parquet('../pm_lol_markets.parquet')
pm = pm[pm.market_slug.str.match(r'^lol-.*-\d{4}-\d{2}-\d{2}(-game\d)?$')].copy()
pm = pm[pm.outs.map(len) == 2]
pm['date'] = pd.to_datetime(pm.market_slug.str.extract(r'(\d{4}-\d{2}-\d{2})')[0], utc=True)
pm['gnum'] = pm.market_slug.str.extract(r'-game(\d)$')[0].astype(float)
m = pd.read_parquet('../data/matches.parquet')
m = m[m.state == 'completed'].copy()
m['n1'] = m.t1.map(norm); m['n2'] = m.t2.map(norm)
m['day'] = m.start.dt.floor('D')

ACAD = {'challengers', 'academy', 'youth', 'ga', 'fenix', 'fénix', 'blue', 'nord', 'rising', 'bees', 'b'}

def sim(a, b):
    a, b = norm(a), norm(b)
    if (set(a.split()) & ACAD) != (set(b.split()) & ACAD):
        return 0
    return max(fuzz.ratio(a, b), fuzz.token_set_ratio(a, b) - 5, 100 if a and a == b else 0)

rows = []
for r in pm.itertuples():
    a, b = r.outs
    c = m[(m.start >= r.date - pd.Timedelta('1D')) & (m.start <= r.date + pd.Timedelta('2D'))]
    best = None
    mm = re.match(r'^lol-([a-z0-9]+)-([a-z0-9]+)-\d{4}', r.market_slug)
    codes = {re.sub(r'\d+$', '', mm.group(1)), re.sub(r'\d+$', '', mm.group(2))} if mm else set()
    for x in c.itertuples():
        a1, a2, b1, b2 = sim(a, x.t1), sim(a, x.t2), sim(b, x.t1), sim(b, x.t2)
        flip = a2 + b1 > a1 + b2
        s = min(a2, b1) if flip else min(a1, b2)
        lc = {re.sub(r'\d+$', '', str(x.t1c).lower()), re.sub(r'\d+$', '', str(x.t2c).lower())}
        if codes and lc == codes:
            s = max(s, 99)
        elif codes & lc and sorted([a1, a2, b1, b2])[-2] >= 90:
            s = max(s, 90)
        # prefer same calendar date
        s -= 3 * abs((x.start.normalize() - r.date).days)
        if best is None or s > best[0]:
            best = (s, x.match_id, x.t1 if not flip else x.t2, x.t2 if not flip else x.t1, x.league, x.start, x.bo, x.w1, x.w2, x.t1, x.o1)
    if best:
        rows.append(dict(market_slug=r.market_slug, condition_id=r.condition_id, gnum=r.gnum, out_a=a, out_b=b,
                         winner_pm=r.winning_outcome_label, score=best[0], match_id=best[1], team_a=best[2], team_b=best[3],
                         league=best[4], start=best[5], bo=best[6], vol=r.vol, fills=r.n,
                         lol_winner=best[9] if best[10] == 'win' else None, lol_t1=best[9]))
L = pd.DataFrame(rows)
L.to_parquet('../data/pm_links_raw.parquet')
print(L.score.describe())
print(L[L.score < 80].sort_values('vol', ascending=False).head(20)[['market_slug', 'out_a', 'out_b', 'team_a', 'team_b', 'score']].to_string())
