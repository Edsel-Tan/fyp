"""Recreate published LoL / MOBA win-probability models on our data and score them next to ours.

Same split and frames as eval_ingame (train 2024-04..2025-06, test 2025-07+, 60 s frames, `finished` frames dropped).
The last 91 days of train are the validation slice (temperature scaling, NN early stopping).

Recreations (what each paper does; what we can reproduce with livestats):
  ours lr all        eval_ingame's all-stats x (1, m, m^2) logistic with the pre-game prior
  ours lr no-prior   the same without the prior (the literature has no team-strength input)
  snapshot@10/15     Kaggle "Diamond 10 min" / Oracle's Elixir GD@15 style: one LR fit on the frame nearest 10:00 (15:00)
  hodge per-minute   Hodge et al. 2021 (IEEE ToG, pro Dota 2): one model per game minute, features = current state plus
                     5-minute deltas (their sliding window). LR, as in their best live system
  riot/aws wp        Riot x AWS broadcast WP (2023 dev diary): gradient boosting on time, gold share, XP (level here), alive,
                     towers, drakes and soul, inhibitor respawn, baron and elder buff timers; no team strength
                     (every game starts at 50 %). XGBoost is not installed, so LightGBM stands in
  silva rnn / lstm   Silva et al. 2018 (SBGames): a recurrent net over the per-minute state sequence, output at every minute
  kim mlp (+T, DU)   Kim et al. 2020 (IEEE CoG): MLP [d,256,256,2]; plain, + temperature scaling, and their data-uncertainty
                     loss (logit mean and log-sigma heads, cross-entropy of the MC-averaged softmax, K=20)
  junior pet leak    Junior & Campelo 2023: slices games at 20/40/60/80 % of *final* game length. Scored two ways:
                     (a) our causal model on those frames (what PET accuracy really measures), (b) LightGBM given
                     PET as a feature, which a live model cannot know
Metrics: per-game mean log loss (each game weighted equally), per-frame Brier, accuracy, ECE (10 equal-width bins),
and log loss at fixed clocks; match-clustered bootstrap CI vs `ours lr all`.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from eval_pregame import BURN, SPLIT, ll, boot_diff
from eval_ingame import STATS, PRE, design

D = '../data'
torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE = ['gold', 'kills', 'towers', 'inhibs', 'barons', 'drakes', 'elders', 'soul', 'level', 'cs', 'alive']


def ece(y, q, bins=10):
    """Binary ECE on P(blue win), equal-width bins over [0, 1]."""
    b = np.minimum((q * bins).astype(int), bins - 1)
    df = pd.DataFrame(dict(b=b, y=y, q=q)).groupby('b').agg(n=('y', 'size'), y=('y', 'mean'), q=('q', 'mean'))
    return (df.n * (df.y - df.q).abs()).sum() / len(y)


def lr_fit(Xtr, ytr, C=1.0):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(Xtr, ytr)


def add_timers(f):
    """Riot-WP style buff/respawn state from the 60 s frames (event time ~ frame time - 30 s)."""
    f = f.sort_values(['game_id', 'clock']).copy()
    g = f.groupby('game_id')
    for col, dur, name in [('barons', 180, 'baron_buff'), ('elders', 150, 'elder_buff'), ('inhibs', 300, 'inhib_down')]:
        d = g[col].diff().fillna(0)
        for s, sign in (('b', 1), ('r', -1)):
            ev = f.clock.where(d * sign > 0) - 30  # the side whose count rose
            last = ev.groupby(f.game_id).ffill()
            f[f'{name}_{s}'] = np.clip(dur - (f.clock - last), 0, dur).fillna(0)
        f[name] = f[f'{name}_b'] - f[f'{name}_r']
    for c in BASE + ['gold_r1', 'gold_r2', 'gold_r3', 'gold_r4', 'gold_r5']:  # 5-min deltas (Hodge sliding window)
        f[f'd5_{c}'] = f[c] - g[c].shift(5).fillna(0)
    f['gold_share'] = f.gold / f.tot_gold.clip(lower=1)
    return f


# ---------------- neural models ----------------
class MLP(nn.Module):
    def __init__(self, d, du=False):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.mu = nn.Linear(256, 2)
        self.ls = nn.Linear(256, 2) if du else None

    def forward(self, x, K=20):
        h = self.body(x)
        mu = self.mu(h)
        if self.ls is None:
            return torch.log_softmax(mu, -1)
        sd = torch.exp(self.ls(h))
        u = mu[None] + sd[None] * torch.randn(K, *mu.shape, device=x.device)
        return torch.log(torch.softmax(u, -1).mean(0) + 1e-9)  # log E[p], Kim et al. eq. 11-12


def train_mlp(Xtr, ytr, Xva, yva, du=False, epochs=20):
    m = MLP(Xtr.shape[1], du).to(DEV)
    opt = torch.optim.Adam(m.parameters(), 1e-4)
    Xt, yt = torch.tensor(Xtr, dtype=torch.float32, device=DEV), torch.tensor(ytr, device=DEV)
    Xv, yv = torch.tensor(Xva, dtype=torch.float32, device=DEV), torch.tensor(yva, device=DEV)
    best, state = 9, None
    for ep in range(epochs):
        m.train()
        for i in torch.randperm(len(Xt), device=DEV).split(512):
            opt.zero_grad()
            nn.functional.nll_loss(m(Xt[i]), yt[i]).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            v = nn.functional.nll_loss(m(Xv, K=100), yv).item()
        if v < best:
            best, state = v, {k: t.clone() for k, t in m.state_dict().items()}
    m.load_state_dict(state)
    return m


def mlp_logit(m, X):
    m.eval()
    with torch.no_grad():
        lp = m(torch.tensor(X, dtype=torch.float32, device=DEV), K=200).cpu().numpy()
    return lp[:, 1] - lp[:, 0]


class RNN(nn.Module):
    def __init__(self, d, cell):
        super().__init__()
        self.r = (nn.RNN if cell == 'rnn' else nn.LSTM)(d, 64, batch_first=True)
        self.o = nn.Linear(64, 1)

    def forward(self, x):
        return self.o(self.r(x)[0]).squeeze(-1)


def seqs(f, cols, mu, sd):
    out = []
    for gid, d in f.groupby('game_id', sort=False):
        out.append((gid, torch.tensor(((d[cols] - mu) / sd).to_numpy(), dtype=torch.float32), float(d.blue_win.iloc[0]), d.index.to_numpy()))
    return out


def pad(batch):
    x = nn.utils.rnn.pad_sequence([b[1] for b in batch], batch_first=True)
    L = torch.tensor([len(b[1]) for b in batch])
    mask = torch.arange(x.shape[1])[None] < L[:, None]
    y = torch.tensor([b[2] for b in batch])[:, None].expand(-1, x.shape[1])
    return x.to(DEV), y.to(DEV), mask.to(DEV)


def train_rnn(tr_s, va_s, cell, epochs=30):
    torch.manual_seed(0)
    m = RNN(tr_s[0][1].shape[1], cell).to(DEV)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    best, state = 9, None
    for ep in range(epochs):
        m.train()
        for i in np.array_split(np.random.default_rng(ep).permutation(len(tr_s)), len(tr_s) // 64):
            x, y, mk = pad([tr_s[j] for j in i])
            opt.zero_grad()
            nn.functional.binary_cross_entropy_with_logits(m(x)[mk], y[mk]).backward()
            opt.step()
        v = rnn_eval(m, va_s)[1]
        if v < best:
            best, state = v, {k: t.clone() for k, t in m.state_dict().items()}
    m.load_state_dict(state)
    return m


def rnn_eval(m, s):
    m.eval()
    idx, lg, tot, n = [], [], 0, 0
    with torch.no_grad():
        for k in range(0, len(s), 256):
            b = s[k:k + 256]
            x, y, mk = pad(b)
            z = m(x)
            tot += nn.functional.binary_cross_entropy_with_logits(z[mk], y[mk], reduction='sum').item()
            n += mk.sum().item()
            for j, e in enumerate(b):
                idx.append(e[3]); lg.append(z[j, :len(e[3])].cpu().numpy())
    return pd.Series(np.concatenate(lg), index=np.concatenate(idx)), tot / n


# ---------------- main ----------------
def main():
    f = pd.read_parquet(f'{D}/ingame.parquet')
    f = f[(f.t_start >= BURN) & (f.state != 'finished')]
    f = add_timers(f).reset_index(drop=True)
    y = f.blue_win.astype(int).to_numpy()
    tr = (f.t_start < SPLIT).to_numpy()
    te = ~tr
    va = tr & (f.t_start >= SPLIT - pd.Timedelta('91D')).to_numpy()
    fit = tr & ~va  # fit slice when a validation slice is needed
    g = f.drop_duplicates('game_id')
    pm = LogisticRegression(max_iter=2000).fit(g[g.t_start < SPLIT][PRE] / 400, g[g.t_start < SPLIT].blue_win.astype(int))
    f['prior_logit'] = pm.decision_function(f[PRE] / 400)
    minute = (f.clock // 60).astype(int)
    Q = {}  # name -> test-frame probabilities (NaN where the model does not predict)

    # ours
    for name, cols in [('ours lr all', STATS + ['prior']), ('ours lr no-prior', STATS)]:
        X = design(f, cols)
        Q[name] = lr_fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    X = design(f, STATS + ['prior'])
    z = lr_fit(X[fit], y[fit]).decision_function(X)
    T = min(np.linspace(.5, 2, 61), key=lambda t: ll(y[va], 1 / (1 + np.exp(-z[va] / t))).mean())
    print(f'ours lr all: temperature on validation slice T={T:.3f}')

    # snapshot@10 / @15: fit on the frame nearest the mark, score only those test frames
    for mk in (10, 15):
        near = (f.clock - mk * 60).abs()
        pick = (near == near.groupby(f.game_id).transform('min')) & (near < 45)
        m = lr_fit(f.loc[pick & tr, BASE], y[pick & tr])
        q = np.full(len(f), np.nan)
        q[pick & te] = m.predict_proba(f.loc[pick & te, BASE])[:, 1]
        Q[f'snapshot@{mk} lr'] = q[te]

    # hodge per-minute models (minutes 3..40, later minutes pooled into 40)
    hc = BASE + [f'd5_{c}' for c in BASE]
    mm = minute.clip(3, 40)
    q = np.full(len(f), np.nan)
    for t in range(3, 41):
        a, b = tr & (mm == t).to_numpy(), te & (mm == t).to_numpy()
        if b.any():
            q[b] = lr_fit(f.loc[a, hc], y[a], C=0.1).predict_proba(f.loc[b, hc])[:, 1]
    Q['hodge per-minute lr'] = q[te]

    # riot/aws WP feature set
    rc = ['clock', 'gold_share', 'level', 'alive', 'towers', 'drakes', 'soul', 'inhib_down_b', 'inhib_down_r',
          'baron_buff_b', 'baron_buff_r', 'elder_buff_b', 'elder_buff_r']
    gb = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    Q['riot/aws wp (gbm)'] = gb.fit(f.loc[tr, rc], y[tr]).predict_proba(f.loc[te, rc])[:, 1]
    gb2 = lgb.LGBMClassifier(**gb.get_params())
    Q['riot/aws wp + prior'] = gb2.fit(f.loc[tr, rc].assign(prior=f.prior_logit[tr]), y[tr]).predict_proba(
        f.loc[te, rc].assign(prior=f.prior_logit[te]))[:, 1]

    # kim et al. MLPs (inputs: every stat, timers, clock, prior; champion one-hots are not used)
    kc = STATS + ['clock', 'baron_buff', 'elder_buff', 'inhib_down', 'prior_logit']
    sc = StandardScaler().fit(f.loc[fit, kc])
    Xk = sc.transform(f[kc])
    mp = train_mlp(Xk[fit], y[fit], Xk[va], y[va])
    zk = mlp_logit(mp, Xk)
    Tk = min(np.linspace(.5, 3, 101), key=lambda t: ll(y[va], 1 / (1 + np.exp(-zk[va] / t))).mean())
    Q['kim mlp'] = 1 / (1 + np.exp(-zk[te]))
    Q[f'kim mlp + temp (T={Tk:.2f})'] = 1 / (1 + np.exp(-zk[te] / Tk))
    md = train_mlp(Xk[fit], y[fit], Xk[va], y[va], du=True)
    Q['kim mlp DU-loss'] = 1 / (1 + np.exp(-mlp_logit(md, Xk)[te]))

    # silva RNN / LSTM over the minute sequence (causal: output at each minute uses frames up to it)
    sq = BASE + ['gold_r1', 'gold_r2', 'gold_r3', 'gold_r4', 'gold_r5', 'clock', 'prior_logit']
    mu, sd = f.loc[fit, sq].mean(), f.loc[fit, sq].std()
    S_fit, S_va, S_te = (seqs(f[msk], sq, mu, sd) for msk in (fit, va, te))
    for cell in ('rnn', 'lstm'):
        m = train_rnn(S_fit, S_va, cell)
        z = rnn_eval(m, S_te)[0].reindex(np.flatnonzero(te))
        Q[f'silva {cell}'] = 1 / (1 + np.exp(-z.to_numpy()))

    # ---------------- scoring ----------------
    ft = f[te].reset_index(drop=True)
    yt = y[te]
    gm = ft.drop_duplicates('game_id').set_index('game_id').match_id
    ref = pd.Series(ll(yt, Q['ours lr all']), index=ft.game_id)
    rows = []
    for k, q in Q.items():
        ok = ~np.isnan(q)
        l = pd.Series(ll(yt[ok], q[ok]), index=ft.game_id[ok])
        pg = l.groupby(level=0).mean()
        rg = ref[ok].groupby(level=0).mean()
        d, lo, hi = boot_diff(pg.to_numpy(), rg.reindex(pg.index).to_numpy(), gm.reindex(pg.index).to_numpy())
        r = dict(model=k, frames=ok.sum(), per_game_ll=pg.mean(), ref_ll_same_frames=rg.mean(), d_vs_ours=d, lo=lo, hi=hi,
                 brier=((q[ok] - yt[ok]) ** 2).mean(), acc=((q[ok] > .5) == yt[ok]).mean(), ece=ece(yt[ok], q[ok]))
        for mk in (5, 10, 15, 20, 25, 30):
            s = ok & (ft.clock // 60 == mk).to_numpy()
            r[f'll@{mk}'] = ll(yt[s], q[s]).mean() if s.any() else np.nan
        rows.append(r)
    R = pd.DataFrame(rows)
    pd.set_option('display.width', 250)
    print(R.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    R.to_csv(f'{D}/../results/lit_compare.csv', index=False)

    # junior & campelo: PET slices (percent of the *final* game length)
    dur = f.groupby('game_id').clock.transform('max')
    f['pet'] = f.clock / dur
    out = []
    gbp = lgb.LGBMClassifier(**gb.get_params()).fit(f.loc[tr, STATS + ['clock', 'pet']], y[tr])
    qpet = gbp.predict_proba(f.loc[te, STATS + ['clock', 'pet']])[:, 1]
    gbc = lgb.LGBMClassifier(**gb.get_params()).fit(f.loc[tr, STATS + ['clock']], y[tr])
    qc = gbc.predict_proba(f.loc[te, STATS + ['clock']])[:, 1]
    pt = f.pet[te].to_numpy()
    for p in (.2, .4, .6, .8):
        near = pd.Series(np.abs(pt - p), index=ft.index)
        pick = (near == near.groupby(ft.game_id).transform('min')).to_numpy()
        clk = ft.clock[pick] / 60
        o = dict(pet=p, games=pick.sum(), mean_min=clk.mean())
        for k, q in (('ours lr all', Q['ours lr all']), ('gbm causal', qc), ('gbm + PET feature', qpet)):
            o[f'acc {k}'] = ((q[pick] > .5) == yt[pick]).mean()
            o[f'll {k}'] = ll(yt[pick], q[pick]).mean()
        out.append(o)
    P = pd.DataFrame(out)
    print('\nPET slices (Junior & Campelo 2023), test games:')
    print(P.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    P.to_csv(f'{D}/../results/lit_compare_pet.csv', index=False)
    ft.assign(**{f'q_{k}': v for k, v in Q.items()}).to_parquet(f'{D}/lit_compare_preds.parquet')


if __name__ == '__main__':
    main()
