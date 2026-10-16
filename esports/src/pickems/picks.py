"""Pick'em recommendations from sim.py's v6 run (2026-10-15).

Play-In: series probabilities of the two opening Bo5s and the Play-In winner.
Swiss:  the picks that maximise the expected number correct: the 2 likeliest 3-0 teams, the 2 likeliest 0-3 teams
        (never a 3-0 pick), and the advance picks ranked by P(3-1 or 3-2) among the rest, plus P(advance) for
        pick'ems that score "advances" regardless of record.
Knockout: title / final odds (the bracket is drawn after Swiss; rerun then).
Usage: python pickems/picks.py
"""
import os
import numpy as np, pandas as pd
import sim

D = sim.D


def main():
    S = pd.read_csv(f'{D}/sim_v6.csv', index_col=0)
    M = pd.read_csv(f'{D}/sim_market.csv', index_col=0)
    P0 = pd.read_csv(f'{D}/sim_v6_prior.csv', index_col=0)
    G = np.load(f'{D}/G_v6.npy')
    ix = sim.IX
    print('== Play-In (Bo5)')
    for a, b in sim.PI:
        g = G[ix[a], ix[b]]
        print(f'  {a} v {b}: game {g:.3f}, series {a} {sim.series(g, 5):.3f}')
    print('  Play-In winner:', S.pi[S.pi > 0].sort_values(ascending=False).round(3).to_dict())
    sw = S[S.index.isin([c for t in sim.TIERS for c in t if c != 'PI']) | (S.pi > 0)]
    sw = sw.assign(adv_31_32=sw['3-1'] + sw['3-2'])
    p30 = sw['3-0'].sort_values(ascending=False)
    p03 = sw['0-3'].drop(p30.index[:2]).sort_values(ascending=False)
    rest = sw.drop(list(p30.index[:2]) + list(p03.index[:2]))
    adv = rest.adv_31_32.sort_values(ascending=False)
    print('\n== Swiss (expected number correct is maximised by the marginal leaders)')
    print('  3-0:', p30.head(5).round(3).to_dict())
    print('  0-3:', p03.head(5).round(3).to_dict())
    print('  advance 3-1/3-2 (excluding the 3-0 and 0-3 picks):', adv.head(8).round(3).to_dict())
    tab = sw[['adv', '3-0', '3-1', '3-2', '2-3', '1-3', '0-3', 'adv_31_32']].copy()
    tab['adv_v6prior'] = P0.adv.reindex(tab.index)
    tab['adv_market'] = M.adv.reindex(tab.index)
    print(tab.sort_values('adv', ascending=False).round(3))
    print('\n== Title / final / semis (v6, v6 prior alone, market)')
    k = pd.DataFrame(dict(ch=S.ch, fin=S.fin, sf=S.sf, ch_v6prior=P0.ch, ch_market=M.ch)).sort_values('ch', ascending=False)
    print(k.round(3))


if __name__ == '__main__':
    main()
