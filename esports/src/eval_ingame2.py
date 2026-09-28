"""Directions 4-5, in-game: new prior and draft / team-strength interactions on top of `lr all`.

Same frames, split and model family as eval_ingame (every term x (1, m, m^2)). Variants:
  ours lr all        eval_ingame as is (prior fit and applied in-sample on train games)
  prior_base oof     same pre-game features, but the train prior is out-of-fold (eval_pregame2)
  prior_new oof      pre-game prior with the new draft/strength features (all new minus h2h)
  + scaling          composition late-game lean (scaling), with time interactions and x gold
  + conv x gold      team conversion rating x gold difference
  + hinge x role gold  sum_k gold_r_k x (hinge_b_k + hinge_r_k): a lead on a high-leverage player is worth more
  + all              all three
  recency weighted   `+ all` with train frames weighted by 0.5 ** (age / 180 d) at the split
  monthly retrain    `+ all` refit at the start of every test month on everything before it
Per-game mean log loss, clustered CI vs `ours lr all` and vs `prior_new oof`, and per-minute buckets.
Writes data/ingame2_test_preds.parquet (frame preds, same row order as ingame_test_preds).
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from eval_pregame import BURN, SPLIT, ll, boot_diff
from eval_ingame import STATS, PRE, design

D = '../data'


def fitq(X, y, tr, te, w=None):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    m.fit(X[tr], y[tr], logisticregression__sample_weight=None if w is None else w[tr])
    return m.predict_proba(X[te])[:, 1]


def main():
    f = pd.read_parquet(f'{D}/ingame.parquet')
    f = f[(f.t_start >= BURN) & (f.state != 'finished')].reset_index(drop=True)
    f = f.merge(pd.read_parquet(f'{D}/prior2.parquet'), on='game_id', how='left') \
         .merge(pd.read_parquet(f'{D}/pregame2.parquet'), on='game_id', how='left')
    y = f.blue_win.astype(int).to_numpy()
    tr, te = (f.t_start < SPLIT).to_numpy(), (f.t_start >= SPLIT).to_numpy()
    g = f.drop_duplicates('game_id')
    pm = LogisticRegression(max_iter=2000).fit(g[g.t_start < SPLIT][PRE] / 400, g[g.t_start < SPLIT].blue_win.astype(int))
    f['prior_logit'] = pm.decision_function(f[PRE] / 400)
    m = f.clock / 600
    f['scal_gold'] = f.scaling * f.gold / 1000
    f['conv_gold'] = f.conv * f.gold / 1000
    f['hinge_gold'] = sum(f[f'gold_r{k + 1}'] / 1000 * (f[f'hinge_b_{k}'] + f[f'hinge_r_{k}']) for k in range(5))

    def X(extra, prior='prior'):
        cols = STATS + extra
        Xd = design(f.assign(prior_logit=f[prior] if prior != 'prior' else f.prior_logit), cols + ['prior'])
        return Xd

    Q = {}
    Q['ours lr all'] = fitq(X([]), y, tr, te)
    Q['prior_base oof'] = fitq(X([], 'prior_base'), y, tr, te)
    Q['prior_new oof'] = fitq(X([], 'prior_new'), y, tr, te)
    Q['+ scaling'] = fitq(X(['scaling', 'scal_gold'], 'prior_new'), y, tr, te)
    Q['+ conv x gold'] = fitq(X(['conv_gold'], 'prior_new'), y, tr, te)
    Q['+ hinge x role gold'] = fitq(X(['hinge_gold'], 'prior_new'), y, tr, te)
    XA = X(['scaling', 'scal_gold', 'conv_gold', 'hinge_gold'], 'prior_new')
    Q['+ all'] = fitq(XA, y, tr, te)
    age = (SPLIT - f.t_start).dt.days.to_numpy()
    Q['+ all, recency weighted'] = fitq(XA, y, tr, te, w=0.5 ** (np.clip(age, 0, None) / 180))
    # monthly retrain: prior_new for test rows is fixed at the split (refitting the pre-game model monthly is a TODO)
    q = np.full(len(f), np.nan)
    month = f.t_start.dt.tz_convert('UTC').dt.tz_localize(None).dt.to_period('M')
    for mo in sorted(month[te].unique()):
        cur = (month == mo).to_numpy() & te
        past = (f.t_start < pd.Timestamp(mo.start_time, tz='UTC')).to_numpy()
        q[cur] = fitq(XA, y, past, cur)
    Q['+ all, monthly retrain'] = q[te]

    ft = f[te].reset_index(drop=True)
    yt = y[te]
    gm = ft.drop_duplicates('game_id').set_index('game_id').match_id
    pg = {k: pd.Series(ll(yt, v), index=ft.game_id).groupby(level=0).mean() for k, v in Q.items()}
    bucket = (ft.clock // 300 * 5).clip(upper=40).astype(int)
    rows = []
    for k, v in Q.items():
        r = dict(model=k, per_game=pg[k].mean(), acc=((v > .5) == yt).mean())
        for ref in ('ours lr all', 'prior_new oof'):
            d, lo, hi = boot_diff(pg[k].to_numpy(), pg[ref].reindex(pg[k].index).to_numpy(), gm.reindex(pg[k].index).to_numpy())
            r[f'd_vs_{ref.split()[0]}'] = d; r[f'ci_{ref.split()[0]}'] = f'[{lo:+.4f}, {hi:+.4f}]'
        r.update(pd.Series(ll(yt, v)).groupby(bucket.to_numpy()).mean().rename(lambda b: f'{b}m').to_dict())
        rows.append(r)
    R = pd.DataFrame(rows)
    pd.set_option('display.width', 300)
    print(R.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    R.to_csv(f'{D}/../results/ingame2_eval.csv', index=False)
    old = pd.read_parquet(f'{D}/ingame_test_preds.parquet')
    assert (old.game_id.to_numpy() == ft.game_id.to_numpy()).all()
    old.assign(**{f'q2_{k}': v for k, v in Q.items()}).to_parquet(f'{D}/ingame2_test_preds.parquet')


if __name__ == '__main__':
    main()
