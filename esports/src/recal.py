"""Direction 2 follow-up: segment recalibration of the in-game model (prior_new oof, eval_ingame2).

confidence.py found `lr all` overconfident on tier-1 and PM-linked games (calibration slope ~0.89 on test).
Here a logit recalibration z' = a_s + b_s z per segment s (tier-1 / tier-2 league) is fit on the validation slice
(last 91 days of train; the model is fit on the rest of train), then applied to the test predictions of the model
fit on all of train. Scored on all test games and on PM-linked games, clustered CI vs no recalibration.
Writes data/ingame_test_preds_v3.parquet (q_lr all := recalibrated prior_new model) for backtest.py.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from eval_pregame import BURN, SPLIT, ll, boot_diff
from eval_ingame import STATS, design
from confidence import TIER1, logit, calib

D = '../data'


def main():
    f = pd.read_parquet(f'{D}/ingame.parquet')
    f = f[(f.t_start >= BURN) & (f.state != 'finished')].reset_index(drop=True)
    f = f.merge(pd.read_parquet(f'{D}/prior2.parquet'), on='game_id', how='left')
    y = f.blue_win.astype(int).to_numpy()
    tr, te = (f.t_start < SPLIT).to_numpy(), (f.t_start >= SPLIT).to_numpy()
    va = tr & (f.t_start >= SPLIT - pd.Timedelta('91D')).to_numpy()
    X = design(f.assign(prior_logit=f.prior_new), STATS + ['prior'])
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    zv = mk().fit(X[tr & ~va], y[tr & ~va]).decision_function(X[va])
    zt = mk().fit(X[tr], y[tr]).decision_function(X[te])
    seg = f.league.isin(TIER1).to_numpy()
    zr = zt.copy()
    for s in (True, False):
        a, b = calib(y[va][seg[va] == s], 1 / (1 + np.exp(-zv[seg[va] == s])))
        print(f'{"tier1" if s else "tier2"}: validation recalibration a={a:+.3f} b={b:.3f}  (validation frames {np.sum(seg[va] == s)})')
        zr[seg[te] == s] = a + b * zt[seg[te] == s]
    ft = f[te].reset_index(drop=True)
    yt = y[te]
    pm = ft.game_id.isin(set(pd.read_parquet(f'{D}/market_frames_lag0.parquet').game_id)).to_numpy()
    q0, q1 = 1 / (1 + np.exp(-zt)), 1 / (1 + np.exp(-zr))
    gm = ft.drop_duplicates('game_id').set_index('game_id').match_id
    for name, s in (('all test', np.ones(len(ft), bool)), ('tier1', seg[te]), ('PM-linked', pm)):
        a = pd.Series(ll(yt[s], q1[s]), index=ft.game_id[s]).groupby(level=0).mean()
        b = pd.Series(ll(yt[s], q0[s]), index=ft.game_id[s]).groupby(level=0).mean()
        d, lo, hi = boot_diff(a.to_numpy(), b.reindex(a.index).to_numpy(), gm.reindex(a.index).to_numpy())
        print(f'{name:10s} per-game ll {b.mean():.4f} -> {a.mean():.4f}  diff {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  '
              f'test slope before {calib(yt[s], q0[s])[1]:.3f} after {calib(yt[s], q1[s])[1]:.3f}')
    old = pd.read_parquet(f'{D}/ingame_test_preds.parquet')
    assert (old.game_id.to_numpy() == ft.game_id.to_numpy()).all()
    old.assign(**{'q_lr all': q1}).to_parquet(f'{D}/ingame_test_preds_v3.parquet')


def rolling():
    """Causal rolling recalibration: month M uses (a, b) fit on the model's predictions in months M-3..M-1
    (validation months supply the history for the first test months)."""
    f = pd.read_parquet(f'{D}/ingame.parquet')
    f = f[(f.t_start >= BURN) & (f.state != 'finished')].reset_index(drop=True)
    f = f.merge(pd.read_parquet(f'{D}/prior2.parquet'), on='game_id', how='left')
    y = f.blue_win.astype(int).to_numpy()
    tr, te = (f.t_start < SPLIT).to_numpy(), (f.t_start >= SPLIT).to_numpy()
    va = tr & (f.t_start >= SPLIT - pd.Timedelta('91D')).to_numpy()
    X = design(f.assign(prior_logit=f.prior_new), STATS + ['prior'])
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    z = np.full(len(f), np.nan)
    z[va] = mk().fit(X[tr & ~va], y[tr & ~va]).decision_function(X[va])
    z[te] = mk().fit(X[tr], y[tr]).decision_function(X[te])
    mo = f.t_start.dt.tz_convert('UTC').dt.tz_localize(None).dt.to_period('M').to_numpy()
    zr = z.copy()
    for m in sorted(set(mo[te])):
        cur = te & (mo == m)
        hist = (va | te) & (mo < m) & (mo >= m - 3)
        a, b = calib(y[hist], 1 / (1 + np.exp(-z[hist])))
        zr[cur] = a + b * z[cur]
    ft = f[te].reset_index(drop=True); yt = y[te]
    q0, q1 = 1 / (1 + np.exp(-z[te])), 1 / (1 + np.exp(-zr[te]))
    pm = ft.game_id.isin(set(pd.read_parquet(f'{D}/market_frames_lag0.parquet').game_id)).to_numpy()
    gm = ft.drop_duplicates('game_id').set_index('game_id').match_id
    for name, s in (('all test', np.ones(len(ft), bool)), ('PM-linked', pm)):
        a = pd.Series(ll(yt[s], q1[s]), index=ft.game_id[s]).groupby(level=0).mean()
        b = pd.Series(ll(yt[s], q0[s]), index=ft.game_id[s]).groupby(level=0).mean()
        d, lo, hi = boot_diff(a.to_numpy(), b.reindex(a.index).to_numpy(), gm.reindex(a.index).to_numpy())
        print(f'rolling 3-month recal {name:10s} {b.mean():.4f} -> {a.mean():.4f}  diff {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  '
              f'slope before {calib(yt[s], q0[s])[1]:.3f} after {calib(yt[s], q1[s])[1]:.3f}')
    # month-by-month slope of the raw model (the drift itself)
    s = pd.DataFrame(dict(m=mo[va | te], y=y[va | te], z=z[va | te])).groupby('m').apply(
        lambda d: calib(d.y.to_numpy(), 1 / (1 + np.exp(-d.z.to_numpy())))[1])
    print('calibration slope by month:', s.round(2).to_dict())


if __name__ == '__main__':
    main()
    rolling()
