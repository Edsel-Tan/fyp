"""Worlds 2026 Monte Carlo (Play-In -> Swiss -> knockout) on a game-level win-probability matrix, and the v6 pick'em
probabilities (2026-10-15).

Format (lolesports schedule + Inven / mmorpg.com primers, 2026-10-15):
  Play-In  C9, KC, FUR, MVK; double elimination, all Bo5; R1 C9-KC and FUR-MVK; the winner enters Swiss in tier 4.
  Swiss    16 teams, 3 wins advance / 3 losses out. R1: tier 1 v tier 4, tier 2 v tier 3, drawn at random with no
           same-region pair. Later rounds: random pairing within a record, no rematches. Bo1, except any match that
           can advance or eliminate (a team at 2 wins or 2 losses), which is Bo3. (Matches the schedule: R3 = 4 Bo1 + 4 Bo3.)
  Knockout the two 3-0 teams each drawn against a 3-2 team, the other four at random; all Bo5. Assumed bracket:
           SF1 = QF(3-0 #1) v QF(random), SF2 = QF(3-0 #2) v QF(random). Same-region separation is not modelled.
Games inside a series are i.i.d. at the game-1 probability (no fearless-draft or momentum terms).

Game matrix (v6): zbar_ij = side-averaged union-prior logit (priors.py, pre-draft fit). v6 blends the market's pre-game
price into the prior: z6 = z + (w0 + w1 cross)(logit pm - z), w0 -0.059, w1 0.516 (v6_model.pkl). No per-match markets
exist yet, so the market's game price is modelled as sigma(m_i - m_j) with team ratings m fitted so this simulator
reproduces Polymarket's "Worlds 2026: Winner" mids (overround removed). Teams quoted below 0.6 % (tick floor, longshot bias; LYON at 0.4 % is too rare to fit at n = 40k) are not
informative and keep their v6 rating.
Usage: python pickems/sim.py   (needs data/pickems/priors.parquet)
"""
import os, sys, json, itertools
import numpy as np, pandas as pd, requests
from multiprocessing import Pool

D = os.path.expanduser('~/fyp/esports/data/pickems')
REG = {'GEN': 'LCK', 'HLE': 'LCK', 'T1': 'LCK', 'DK': 'LCK', 'AL': 'LPL', 'BLG': 'LPL', 'TES': 'LPL', 'IG': 'LPL',
       'G2': 'LEC', 'MKOI': 'LEC', 'KC': 'LEC', 'TLAW': 'LCS', 'LYON': 'LCS', 'C9': 'LCS', 'TSW': 'LCP', 'CFO': 'LCP',
       'MVK': 'LCP', 'LOS': 'CBLOL', 'FUR': 'CBLOL'}
TIERS = [['GEN', 'AL', 'TLAW', 'G2'], ['LOS', 'TSW', 'HLE', 'BLG'], ['LYON', 'MKOI', 'T1', 'TES'], ['CFO', 'IG', 'DK', 'PI']]
PI = [('C9', 'KC'), ('FUR', 'MVK')]
TEAMS = list(REG)
IX = {c: i for i, c in enumerate(TEAMS)}
W0, W1 = -0.05887944524834938, 0.5157092749163774
PM_NAMES = {'GEN': 'Gen.G', 'T1': 'T1', 'HLE': 'Hanwha Life Esports', 'DK': 'Dplus KIA', 'BLG': 'Bilibili Gaming',
            'G2': 'G2 Esports', 'KC': 'Karmine Corp', 'TSW': 'Team Secret Whales', 'CFO': 'CTBC Flying Oyster',
            'MVK': 'MVK Esports', 'AL': "Anyone's Legend", 'TES': 'Top Esports', 'IG': 'Invictus Gaming',
            'MKOI': 'Movistar KOI', 'C9': 'Cloud9', 'TLAW': 'Team Liquid', 'LYON': 'LYON', 'LOS': 'LOS'}
sg = lambda x: 1 / (1 + np.exp(-x))
lgt = lambda p: np.log(p / (1 - p))


def series(p, bo):
    if bo == 1:
        return p
    if bo == 3:
        return p * p * (3 - 2 * p)
    return p ** 3 * (10 - 15 * p + 6 * p * p)


def one(G, rng):
    """One tournament. Returns (pi_winner, {team: (w, l)}, qf, sf, finalists, champion)."""
    def play(a, b, bo):
        return (a, b) if rng.random() < series(G[IX[a], IX[b]], bo) else (b, a)
    (w1, l1), (w2, l2) = play(*PI[0], 5), play(*PI[1], 5)
    uw, ul = play(w1, w2, 5)
    lw, _ = play(l1, l2, 5)
    lf, _ = play(ul, lw, 5)
    piw, _ = play(uw, lf, 5)
    tiers = [[piw if c == 'PI' else c for c in t] for t in TIERS]
    rec = {c: [0, 0] for t in tiers for c in t}
    met = set()
    # round 1
    for a_t, b_t in ((0, 3), (1, 2)):
        A = tiers[a_t]
        while True:
            B = list(rng.permutation(tiers[b_t]))
            if all(REG[a] != REG[b] for a, b in zip(A, B)):
                break
        for a, b in zip(A, B):
            w, l = play(a, b, 1)
            rec[w][0] += 1; rec[l][1] += 1; met.add(frozenset((a, b)))
    for _ in range(4):
        groups = {}
        for c, (w, l) in rec.items():
            if w < 3 and l < 3:
                groups.setdefault((w, l), []).append(c)
        for (w, l), g in groups.items():
            bo = 3 if (w == 2 or l == 2) else 1
            for tries in range(200):
                g2 = list(rng.permutation(g))
                pairs = list(zip(g2[::2], g2[1::2]))
                if tries == 199 or all(frozenset(p) not in met for p in pairs):
                    break
            for a, b in pairs:
                ww, ll = play(a, b, bo)
                rec[ww][0] += 1; rec[ll][1] += 1; met.add(frozenset((a, b)))
    t30 = [c for c, r in rec.items() if r == [3, 0]]
    t32 = list(rng.permutation([c for c, r in rec.items() if r == [3, 2]]))
    rest = list(rng.permutation([c for c, r in rec.items() if r == [3, 1]] + t32[2:]))
    qf = [(t30[0], t32[0]), (rest[0], rest[1]), (t30[1], t32[1]), (rest[2], rest[3])]
    qw = [play(a, b, 5)[0] for a, b in qf]
    s1, _ = play(qw[0], qw[1], 5)
    s2, _ = play(qw[2], qw[3], 5)
    ch, _ = play(s1, s2, 5)
    return piw, rec, qw, (s1, s2), ch


def _chunk(args):
    G, n, seed = args
    rng = np.random.default_rng(seed)
    T = len(TEAMS)
    out = dict(pi=np.zeros(T), adv=np.zeros(T), sf=np.zeros(T), fin=np.zeros(T), ch=np.zeros(T),
               rec={k: np.zeros(T) for k in ('3-0', '3-1', '3-2', '2-3', '1-3', '0-3')})
    for _ in range(n):
        piw, rec, qw, fin, ch = one(G, rng)
        out['pi'][IX[piw]] += 1
        for c, (w, l) in rec.items():
            out['rec'][f'{w}-{l}'][IX[c]] += 1
            out['adv'][IX[c]] += w == 3
        for c in qw:
            out['sf'][IX[c]] += 1
        for c in fin:
            out['fin'][IX[c]] += 1
        out['ch'][IX[ch]] += 1
    return out


def simulate(G, n=40000, seed=0, procs=40):
    k = procs
    with Pool(k) as p:
        parts = p.map(_chunk, [(G, n // k, seed * 1000 + i) for i in range(k)])
    N = (n // k) * k
    df = pd.DataFrame(index=TEAMS)
    for key in ('pi', 'adv', 'sf', 'fin', 'ch'):
        df[key] = sum(x[key] for x in parts) / N
    for r in ('3-0', '3-1', '3-2', '2-3', '1-3', '0-3'):
        df[r] = sum(x['rec'][r] for x in parts) / N
    df.attrs['n'] = N
    return df


def v6_matrix():
    P = pd.read_parquet(f'{D}/priors.parquet')
    z = {(a, b): v for a, b, v in zip(P.a, P.b, P.z_predraft)}
    T = len(TEAMS)
    Z = np.zeros((T, T))  # side-averaged probability -> logit
    for a in TEAMS:
        for b in TEAMS:
            if a != b:
                Z[IX[a], IX[b]] = lgt(0.5 * (sg(z[a, b]) + 1 - sg(z[b, a])))
    # Bradley-Terry ratings of the v6 matrix (least squares on the logits; for the table and the market anchor)
    A, y = [], []
    for i, j in itertools.combinations(range(T), 2):
        r = np.zeros(T); r[i], r[j] = 1, -1
        A.append(r); y.append(Z[i, j])
    A.append(np.ones(T)); y.append(0)
    r = np.linalg.lstsq(np.array(A), np.array(y), rcond=None)[0]
    return Z, r


def pm_outrights():
    ev = requests.get('https://gamma-api.polymarket.com/events', params={'slug': 'worlds-2026-winner'}, timeout=20).json()[0]
    inv = {v: k for k, v in PM_NAMES.items()}
    out = {}
    for m in ev['markets']:
        c = inv.get(m.get('groupItemTitle'))
        if c and m.get('bestBid') is not None and m.get('bestAsk') is not None:
            out[c] = (float(m['bestBid']) + float(m['bestAsk'])) / 2
    return pd.Series(out)


def fit_market(r, target, iters=14, n=40000):
    """Team ratings m whose simulated title odds match the market's (fitted teams only; others held at v6's r)."""
    m = r.copy()
    fit = [IX[c] for c in target.index]
    for it in range(iters):
        G = sg(m[:, None] - m[None, :])
        s = simulate(G, n=n, seed=100 + it)['ch'].to_numpy()
        err = lgt(target.to_numpy()) - lgt(np.clip(s[fit], 1e-4, 1))
        m[fit] += 0.35 * err
        print(f'  market fit it {it}: max |logit err| {np.abs(err).max():.3f}', flush=True)
    return m


def main():
    Z, r = v6_matrix()
    J = json.load(open(f'{D}/teams.json'))
    cross = np.array([[REG[a] != REG[b] for b in TEAMS] for a in TEAMS], float)
    out = {}
    out['v6_prior'] = simulate(sg(Z), seed=1)
    pm = pm_outrights()
    pm = pm / pm.sum()  # remove the overround
    tgt = pm[pm >= 0.006]
    print('market title odds (normalised), fitted teams:', tgt.round(4).to_dict(), flush=True)
    m = fit_market(r, tgt)
    PMZ = m[:, None] - m[None, :]
    Z6 = Z + (W0 + W1 * cross) * (PMZ - Z)
    out['market'] = simulate(sg(PMZ), seed=2)
    out['v6'] = simulate(sg(Z6), seed=3, n=200000)
    G6 = sg(Z6)
    np.save(f'{D}/G_v6.npy', G6)
    rat = pd.DataFrame(dict(v6_prior_rating=r, market_rating=m, region=[REG[c] for c in TEAMS]), index=TEAMS)
    rat.to_csv(f'{D}/ratings.csv')
    for k, df in out.items():
        df.to_csv(f'{D}/sim_{k}.csv')
    pd.DataFrame(G6, index=TEAMS, columns=TEAMS).to_csv(f'{D}/game_matrix_v6.csv')
    pd.set_option('display.width', 200)
    print(rat.sort_values('v6_prior_rating', ascending=False).round(3))
    for k, df in out.items():
        print(f'\n== {k} (n={df.attrs["n"]})')
        print(df.sort_values('ch', ascending=False).round(3))


if __name__ == '__main__':
    main()
