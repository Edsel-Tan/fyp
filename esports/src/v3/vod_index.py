"""v3 step 1: VOD index. For every labelled game, the broadcast VOD to pull minimap frames from.

getEventDetails lists per-game vods (provider, parameter, locale, firstFrameTime). VOD second v shows wall time
firstFrameTime + v (verified on GEN-KT, PROGRESS.md "Visualisations"). We keep YouTube vods, preferring en-US,
then en-*, then any locale (the minimap is the same in every language feed). Writes data/v3/vods.parquet.
"""
import sys, os, json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lolapi

D = os.path.join(os.path.dirname(__file__), '../../data')
OUT = f'{D}/v3'
RANK = lambda v: (v['locale'] != 'en-US', not v['locale'].startswith('en'), v['locale'])


def one(mid):
    e = lolapi.event_details(str(mid))
    rows = []
    for g in (e or {}).get('match', {}).get('games', []):
        vs = sorted([v for v in g.get('vods', []) if v.get('provider') == 'youtube' and v.get('firstFrameTime')], key=RANK)
        for k, v in enumerate(vs[:3]):  # keep up to 3 candidates; the extractor falls back if one is dead
            rows.append(dict(game_id=g['id'], match_id=str(mid), rank=k, yt=v['parameter'], locale=v['locale'],
                             first_frame=v['firstFrameTime'], start_ms=v.get('startMillis'), end_ms=v.get('endMillis')))
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    t = pd.read_parquet(f'{D}/game_table.parquet')
    mids = t.match_id.astype(str).unique().tolist()
    with ThreadPoolExecutor(16) as ex:
        rows = [r for rs in ex.map(one, mids) for r in rs]
    v = pd.DataFrame(rows)
    v = v[v.game_id.isin(t.game_id.astype(str))]
    v.to_parquet(f'{OUT}/vods.parquet')
    print(f'{len(mids)} matches; games with a YouTube vod: {v.game_id.nunique()} / {t.game_id.nunique()}')
    print(v[v['rank'] == 0].locale.value_counts().head(10).to_string())


if __name__ == '__main__':
    main()
