"""Freeze v4 for the live tracker (2026-10-05): ~/fyp-pmlive/lollive/data/v4/v4_model.pkl and src/v4feat.py.

v4 = eval_v4's `sel window`: v2 (STATS + prior_new, each x (1, m, m^2)) plus the feature groups chosen by forward selection on the
validation slice that live can compute (H fight in progress, S soul race / drake types, G relative gold, R respawn), fit on the
study's train split (before 2025-07). Test split: -0.0007 [-0.0012, -0.0001] vs v2; vs the market and live: ns (PROGRESS "v4").
Checks before writing:
  1. the frozen model reproduces eval_v4's test-split predictions exactly
  2. the live code path (lollive common.design + v4feat.window_feats on lollive's own feature frame) reproduces eval_v4's live
     rescoring of the saved feeds (data/v4/live_rescored.parquet, column `q_sel window`)
features.py is copied verbatim to lollive src/v4feat.py, so training and live run the same feature code.
"""
import os, sys, shutil, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_pregame import SPLIT, ll
from eval_v4 import load, X_of

D = '../data'
LL = os.path.expanduser('~/fyp-pmlive/lollive')
NAME = 'sel window'


def main():
    M = pickle.load(open(f'{D}/v4/models.pkl', 'rb'))['models'][NAME]
    assert 'leg' not in M['cols'], 'the live model must be window-only'
    spec = (M['cols'], M['basis'], M['prior'])
    # 1. test split
    f = load()
    te = (f.t_start >= SPLIT).to_numpy()
    X = X_of(f[te].reset_index(drop=True), NAME, spec)
    assert list(X.columns) == M['xcols']
    q = M['model'].predict_proba(X)[:, 1]
    pr = pd.read_parquet(f'{D}/ingame_test_preds_v4.parquet')
    assert np.abs(pr[f'q_{NAME}'].to_numpy() - q).max() < 1e-12
    ft = f[te].reset_index(drop=True)
    pg = lambda p: pd.Series(ll(ft.blue_win.astype(int).to_numpy(), p), index=ft.game_id).groupby(level=0).mean().mean()
    ll_v4, ll_v2 = pg(q), pg(pr['q_lr v2'].to_numpy())
    print(f'1. test split reproduces: v4 {ll_v4:.4f} vs v2 {ll_v2:.4f} per-game log loss ({len(set(ft.game_id))} games)')

    # 2. live code path, run from lollive's own modules on its saved feeds
    os.makedirs(f'{LL}/data/v4', exist_ok=True)
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'features.py'), f'{LL}/src/v4feat.py')
    out = dict(model=M['model'], cols=M['cols'], xcols=M['xcols'], basis=M['basis'], prior=M['prior'], groups=M.get('groups'),
               name='v4 (eval_v4 sel window)', split=str(SPLIT), built='2026-10-05', test_ll=ll_v4, test_ll_v2=ll_v2)
    pickle.dump(out, open(f'{LL}/data/v4/v4_model.pkl', 'wb'))
    sys.path.insert(0, f'{LL}/src')
    import v4live
    V = v4live.V4()
    j = pd.read_parquet(f'{D}/v4/live_rescored.parquet')
    # eval_v4 part 4 used the tick's own clock (the trader's anchor); the saved feed re-anchors differently in games the trader
    # joined mid-game, so compare only ticks whose clocks agree (in the trader, v4 and v2 share one frame table and clock)
    err, skipped = [], 0
    for gid, tk in list(j.groupby('game_id'))[:25]:
        raw = pd.read_parquet(f'{LL}/data/frames60/{gid}.parquet').drop_duplicates('ts')
        for _, r in tk.iloc[::7].iterrows():
            upto = raw[pd.to_datetime(raw.ts, format='ISO8601') <= r.ts]  # the feed as the trader held it at that frame
            o = v4live.live.features(upto)
            if o is None or abs(o.clock.iloc[-1] - r.clock) > 1:
                skipped += 1
                continue
            res = V.predict(upto, r.prior_new, o=o)
            if res is not None:
                err.append(abs(res[0] - r[f'q_{NAME}']))
    err = np.array(err)
    print(f'2. live code path vs eval_v4 rescoring: {len(err)} ticks, max |diff| {err.max():.2e} '
          f'({skipped} ticks skipped: saved-feed clock differs from the trader\'s)')
    assert err.max() < 1e-9
    print(f'wrote {LL}/data/v4/v4_model.pkl and {LL}/src/v4feat.py')


if __name__ == '__main__':
    main()
