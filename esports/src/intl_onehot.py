"""Next step 1: does player identity (one-hot) help at international events, where the regional rating pools meet?

PLAN's "Faker" worry: Elo is pooled by opponents, and before an international event most of a player's games are
against their own league, so a rating may misplace whole leagues. One-hot player effects are also estimated from
within-league games, so they share the problem; this checks whether they add anything on international games.
Same split as eval_pregame. Models (C chosen on the last 91 days of train, then refit on all train):
  all features       eval_pregame's logistic (14 rating/form features)
  prior_new          eval_pregame2's `all new minus h2h` (adds league Elo, lane Elo, draft)
  one-hot            +1 per blue player, -1 per red player
  one-hot + elo      plus elo_play_32, elo_team_32
  one-hot | base     one-hot as a residual on top of the out-of-fold `prior_base` logit (the prior enters scaled x100,
                     so the L2 penalty barely touches it; train rows use the out-of-fold prior, test rows the train fit)
  one-hot | new      the same on top of `prior_new`
Scores on all test games, the international subset, and international games between leagues (league_elo != 0),
with match-clustered CIs vs `all features` and vs `prior_new`.
"""
import numpy as np, pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from eval_pregame import BURN, SPLIT, INTL, ll, boot_diff, player_design
from eval_pregame2 import BASE

D = '../data'


def main():
    p = pd.read_parquet(f'{D}/pregame.parquet').merge(pd.read_parquet(f'{D}/pregame2.parquet')[['game_id', 'league_elo']], on='game_id') \
        .merge(pd.read_parquet(f'{D}/prior2.parquet'), on='game_id')
    p = p[p.t_start >= BURN].reset_index(drop=True)
    tr, te = (p.t_start < SPLIT).to_numpy(), (p.t_start >= SPLIT).to_numpy()
    va = tr & (p.t_start >= SPLIT - pd.Timedelta('91D')).to_numpy()
    fit = tr & ~va
    y = p.blue_win.astype(int).to_numpy()
    idx = {pl: j for j, pl in enumerate(sorted({x for l in p.b_players[tr] for x in l} | {x for l in p.r_players[tr] for x in l}))}
    P1 = player_design(p, idx)

    def mat(dense):
        if not dense:
            return P1
        sc = p.loc[tr, dense].std().replace(0, 1).to_numpy()
        return sp.hstack([P1, sp.csr_matrix(p[dense].to_numpy() / sc)]).tocsr()

    def run(X, Cs=(0.01, 0.03, 0.1, 0.3, 1, 3)):
        C = min((log_loss(y[va], LogisticRegression(C=C, max_iter=5000).fit(X[fit], y[fit]).predict_proba(X[va])[:, 1]), C) for C in Cs)[1]
        m = LogisticRegression(C=C, max_iter=5000).fit(X[tr], y[tr])
        return m.predict_proba(X[te])[:, 1], C, log_loss(y[tr], m.predict_proba(X[tr])[:, 1])

    def dense_lr(cols):
        sc = p.loc[tr, cols].std().replace(0, 1)
        return run(sp.csr_matrix(p[cols].to_numpy() / sc.to_numpy()), Cs=(0.003, 0.01, 0.03, 0.1, 1))

    sig = lambda z: 1 / (1 + np.exp(-z))
    Q = {'all features': dense_lr(BASE), 'prior_new': (sig(p.prior_new[te].to_numpy()), np.nan, np.nan),
         'one-hot': run(mat([])), 'one-hot + elo': run(mat(['elo_play_32', 'elo_team_32']))}
    for nm, col in (('one-hot | base', 'prior_base'), ('one-hot | new', 'prior_new')):
        X = sp.hstack([P1, sp.csr_matrix(p[[col]].to_numpy() * 100)]).tocsr()
        Q[nm] = run(X)

    pt = p[te].reset_index(drop=True)
    yt, grp = y[te], pt.match_id.to_numpy()
    subs = {'all': np.ones(len(pt), bool), 'intl': pt.league.isin(INTL).to_numpy()}
    subs['intl cross-league'] = subs['intl'] & (pt.league_elo != 0).to_numpy()
    rows = []
    for k, (q, C, trll) in Q.items():
        r = dict(model=k, C=C, train_ll=trll)
        for s, m in subs.items():
            l = ll(yt[m], q[m])
            r[f'll_{s}'] = l.mean()
            for ref in ('all features', 'prior_new'):
                d, lo, hi = boot_diff(l, ll(yt[m], Q[ref][0][m]), grp[m])
                r[f'd_{s}_vs_{ref.split()[0]}'] = f'{d:+.4f} [{lo:+.4f}, {hi:+.4f}]'
        rows.append(r)
    R = pd.DataFrame(rows)
    print(f'test n={te.sum()}  intl n={subs["intl"].sum()}  cross-league n={subs["intl cross-league"].sum()}  '
          f'players {len(idx)}  train intl games {(tr & p.league.isin(INTL).to_numpy()).sum()}')
    pd.set_option('display.width', 400); pd.set_option('display.max_columns', 50)
    for s in subs:
        print(f'\n== {s}')
        print(R[['model', 'C', 'train_ll', f'll_{s}', f'd_{s}_vs_all', f'd_{s}_vs_prior_new']].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    R.to_csv(f'{D}/../results/intl_onehot.csv', index=False)


if __name__ == '__main__':
    main()
