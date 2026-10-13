"""v4 step 1: the candidate in-game features from PROGRESS "Missing inputs and feature construction" (2026-10-05), per 60 s frame.

Everything is causal within a game (current and earlier frames only) and, except the item group, uses only `window` fields, so the
live tracker can compute it from its own feed. Events seen between two frames are dated at their midpoint (+-30 s at 60 s sampling).
All features are blue minus red unless noted. Groups (eval_v4.py adds each to v2 on its own, then all together):

  T transient state (construction 3)
    inh_dn     enemy inhibitors currently down (destroyed < 300 s ago, capped at 3): blue's count minus red's
    bar_buf    baron buff time left / 180 s (blue minus red); ends at 180 s, holders' deaths are ignored
    eld_buf    elder buff time left / 150 s
  S soul race and drake types (construction 4)
    drk_sq     nb^2 - nr^2 of elemental drakes (convex: a drake nearer soul is worth more)
    soul_pt    (blue on 3) - (red on 3), before either has soul
    dt_<type>  per-type elemental drake count difference (infernal, ocean, mountain, cloud, hextech, chemtech)
  R respawn (inputs C, construction 5)
    dead_s     sum of dead players' expected respawn seconds left / 60 (base respawn by level x the 15/30/45-min time factor;
               a death since the previous frame is placed mid-interval)
    dead_g     the same, each weighted by the dead player's totalGold (k): a dead carry costs more
  H fight in progress (input B)
    hp_al      mean currentHealth / maxHealth of alive players
    low_hp     alive players under 35 % health (blue count minus red count)
    fight_n    kills by both teams in the last minute (side-symmetric, enters only via fight_g)
    fight_g    fight_n x gold lead (k)
    dk1, dg1   kill / gold (k) lead change in the last minute
               ("last minute" = since the latest frame >= 50 s older: the previous row at 60 s sampling, and the same window on
               a dense live feed; a row-to-row difference there spans ~0.2 s, which was a train/serve skew)
  D 5-minute deltas (Hodge et al., on top of v2)
    dg5, dk5, dt5, dd5   change in gold (k), kills, towers, drakes over the last 300 s
  B bounties (input E, construction 6)
    bty        shutdown gold at risk (k): per player min(1000, 100 * max(0, streak - 1) + 0.15 * max(0, gold - game mean)),
               streak = kills since the player's last death
    objb       objective-bounty proxy after 14:00: (max(0, -gold - 3000) - max(0, gold - 3000)) / 1000
  I item power spikes (input D; details60)
    leg        completed legendary items (Data Dragon: no build path out, >= 2,200 gold, purchasable, not boots)
    leg_d1     change since the previous frame (a fresh spike)
    it_bad     1 if the details frame is missing (then leg = leg_d1 = 0)
  L level thresholds (construction 7)
    ultr       ranks unlocked: sum over players of [lvl>=6] + [lvl>=11] + [lvl>=16]
    lvl_r1..5  per-role level difference
  G relative gold (construction 2)
    gsh        (b_gold - r_gold) / (b_gold + r_gold)
    gsat       tanh(gold lead / 5000)
Writes data/v4/feat.parquet keyed (game_id, ts), same rows as ingame.parquet.
"""
import glob, json, math, os, sys
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data')
B, R = range(1, 6), range(6, 11)
TYPES = ['infernal', 'ocean', 'mountain', 'cloud', 'hextech', 'chemtech']
GROUPS = {
    'T': ['inh_dn', 'bar_buf', 'eld_buf'],
    'S': ['drk_sq', 'soul_pt'] + [f'dt_{k}' for k in TYPES],
    'R': ['dead_s', 'dead_g'],
    'H': ['hp_al', 'low_hp', 'fight_g', 'dk1', 'dg1'],
    'D': ['dg5', 'dk5', 'dt5', 'dd5'],
    'B': ['bty', 'objb'],
    'I': ['leg', 'leg_d1', 'it_bad'],
    'L': ['ultr'] + [f'lvl_r{k}' for k in range(1, 6)],
    'G': ['gsh', 'gsat'],
}
WINDOW_ONLY = [g for g in GROUPS if g != 'I']
LAG1 = 50.0  # s: one-frame deltas look back to the latest frame at least this old
# base respawn wait by level 1..18 (levels 19-20 exist since the 2026 top-lane quest: +2.5 s each)
BRW = [10, 10, 12, 12, 14, 16, 20, 25, 28, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50, 52.5, 55, 57.5]


def respawn(level, clock):
    t = clock / 60
    if t < 15:
        f = 0.0
    elif t < 30:
        f = math.ceil(2 * (t - 15)) * 0.00425
    elif t < 45:
        f = 0.1275 + math.ceil(2 * (t - 30)) * 0.003
    else:
        f = min(0.5, 0.2175 + math.ceil(2 * (t - 45)) * 0.0145)
    return BRW[min(max(int(level), 1), 20) - 1] * (1 + f)


_LEG = {}


def legendary(patch):
    """Set of legendary item ids for the patch (nearest downloaded Data Dragon table, else the latest)."""
    mm = '.'.join(str(patch).split('.')[:2])
    if mm not in _LEG:
        f = f'{D}/v3/items/{mm}.1.json'
        if not os.path.exists(f):
            f = sorted(glob.glob(f'{D}/v3/items/*.json'), key=lambda s: tuple(map(int, os.path.basename(s).split('.')[:2])))[-1]
        _LEG[mm] = {int(k) for k, v in json.load(open(f))['data'].items()
                    if not v.get('into') and v['gold']['total'] >= 2200 and v['gold'].get('purchasable')
                    and 'Boots' not in v.get('tags', []) and v.get('maps', {}).get('11', True)}
    return _LEG[mm]


def raw_frames(d, min_gold=0):
    """The frame filter of ingame_data.one (min_gold=0) or live.features (min_gold=5000), on a raw frame table."""
    d = d[d.state.isin(['in_game', 'paused', 'finished'])]
    d = d.assign(_t=pd.to_datetime(d.ts, format='ISO8601')).sort_values('_t').drop_duplicates('_t')
    return d[(d.b_gold + d.r_gold) > min_gold].reset_index(drop=True)


def drakes(s):
    return [x for x in str(s or '').split(',') if x]


def window_feats(d, clock):
    """Groups T S R H D B L G from a filtered raw frame table d and its game clock (seconds), one row per frame."""
    n = len(d)
    c = np.asarray(clock, float)
    mid = np.r_[c[0], (c[1:] + c[:-1]) / 2]
    out = {}
    gold = (d.b_gold - d.r_gold).to_numpy(float)
    tot = (d.b_gold + d.r_gold).to_numpy(float)

    # T: inhibitors down, baron and elder buffs (kill events from cumulative counters)
    def events(cum):
        cum = np.asarray(cum, float)
        inc = np.r_[0, np.clip(np.diff(cum), 0, None)]
        return [(mid[k], int(inc[k])) for k in range(n) if inc[k] > 0]

    def active(ev, dur, cap=None, frac=False):
        v = np.zeros(n)
        for k in range(n):
            s = 0.0
            for tk, m in ev:
                if tk <= c[k] and c[k] - tk < dur:
                    s += m * ((dur - (c[k] - tk)) / dur if frac else 1)
            v[k] = min(s, cap) if cap else s
        return v
    lb = d.b_drakes.map(drakes)
    lr = d.r_drakes.map(drakes)
    out['inh_dn'] = active(events(d.b_inhibs), 300, 3) - active(events(d.r_inhibs), 300, 3)
    out['bar_buf'] = active(events(d.b_barons), 180, frac=True) - active(events(d.r_barons), 180, frac=True)
    out['eld_buf'] = active(events(lb.map(lambda l: l.count('elder'))), 150, frac=True) - \
        active(events(lr.map(lambda l: l.count('elder'))), 150, frac=True)

    # S: soul race and drake types
    nb = lb.map(lambda l: sum(x != 'elder' for x in l)).to_numpy(float)
    nr = lr.map(lambda l: sum(x != 'elder' for x in l)).to_numpy(float)
    out['drk_sq'] = nb ** 2 - nr ** 2
    nosoul = (nb < 4) & (nr < 4)
    out['soul_pt'] = np.where(nosoul, (nb == 3).astype(float) - (nr == 3).astype(float), 0)
    for k in TYPES:
        out[f'dt_{k}'] = (lb.map(lambda l: l.count(k)) - lr.map(lambda l: l.count(k))).to_numpy(float)

    # per-player arrays (n x 10)
    P = lambda col: np.stack([d[f'p{i}_{col}'].to_numpy(float) for i in range(1, 11)], 1)
    pg, lvl, kil, dth = P('totalGold'), P('level'), P('kills'), P('deaths')
    hp, mhp = P('currentHealth'), np.maximum(P('maxHealth'), 1)
    dead = hp <= 0
    side = np.r_[np.ones(5), -np.ones(5)]

    # R: respawn time left of dead players. The death lies between the row that first shows it (obs) and the row before, so
    # elapsed ~ U(c - c[obs], min(T, c - c[obs - 1])): exact on a dense live feed, the mid-interval guess at 60 s sampling
    rem = np.zeros((n, 10))
    died = np.vstack([np.zeros((1, 10), bool), np.diff(dth, axis=0) > 0])
    obs = np.full(10, -1)
    for k in range(n):
        obs = np.where(died[k], k, obs)
        for i in range(10):
            if dead[k, i]:
                T = respawn(lvl[k, i], c[k])
                if obs[i] < 1:
                    rem[k, i] = T / 2
                    continue
                lo, hi = c[k] - c[obs[i]], min(T, c[k] - c[obs[i] - 1])
                rem[k, i] = max(2.0, T - (lo + max(hi, lo)) / 2)
    out['dead_s'] = (rem / 60) @ side
    out['dead_g'] = (rem / 60 * pg / 1000) @ side

    # H: fight in progress
    alive = ~dead
    frac = np.where(alive, hp / mhp, np.nan)
    with np.errstate(invalid='ignore'):
        mb = np.nan_to_num(np.nanmean(frac[:, :5], 1), nan=0.0)
        mr = np.nan_to_num(np.nanmean(frac[:, 5:], 1), nan=0.0)
    out['hp_al'] = mb - mr
    low = alive & (hp / mhp < 0.35)
    out['low_hp'] = low[:, :5].sum(1) - low[:, 5:].sum(1).astype(float)
    # "since the previous frame" = since the latest frame at least LAG1 s older (the previous row at 60 s sampling; on a dense
    # live feed, the frame ~a minute back), so training and live see the same window
    kb, kr = d.b_kills.to_numpy(float), d.r_kills.to_numpy(float)
    j1 = np.searchsorted(c, c - LAG1, side='right') - 1
    j1 = np.where(j1 >= 0, j1, np.arange(n))
    out['fight_g'] = ((kb + kr) - (kb + kr)[j1]) * gold / 1000
    out['dk1'] = (kb - kr) - (kb - kr)[j1]
    out['dg1'] = (gold - gold[j1]) / 1000

    # D: 5-minute deltas (frame nearest to clock - 300, at or before it; the first frame if none)
    j = np.clip(np.searchsorted(c, c - 300, side='right') - 1, 0, None)
    tw = (d.b_towers - d.r_towers).to_numpy(float)
    out['dg5'] = (gold - gold[j]) / 1000
    out['dk5'] = (kb - kr) - (kb - kr)[j]
    out['dt5'] = tw - tw[j]
    out['dd5'] = (nb - nr) - (nb - nr)[j]

    # B: shutdown bounties (streak = kills since the last death) and the objective-bounty proxy
    base = np.zeros(10)
    bty = np.zeros((n, 10))
    for k in range(n):
        base = np.where(died[k], kil[k], base)
        streak = kil[k] - base
        bty[k] = np.minimum(1000, 100 * np.maximum(0, streak - 1) + 0.15 * np.maximum(0, pg[k] - pg[k].mean()))
    out['bty'] = (bty / 1000) @ side
    out['objb'] = np.where(c >= 840, (np.maximum(0, -gold - 3000) - np.maximum(0, gold - 3000)) / 1000, 0)

    # L: level thresholds
    ranks = (lvl >= 6).astype(float) + (lvl >= 11) + (lvl >= 16)
    out['ultr'] = ranks @ side
    for k in range(5):
        out[f'lvl_r{k + 1}'] = lvl[:, k] - lvl[:, k + 5]

    # G: relative gold
    out['gsh'] = np.where(tot > 0, gold / np.maximum(tot, 1), 0)
    out['gsat'] = np.tanh(gold / 5000)
    return pd.DataFrame(out)


def item_feats(gid, ts, patch):
    """Group I from details60 at the frame timestamps (ts: Series of UTC timestamps)."""
    fp = f'{D}/details60/{gid}.parquet'
    n = len(ts)
    leg = np.zeros(n)
    bad = np.ones(n)
    if os.path.exists(fp):
        dd = pd.read_parquet(fp)
        sec = (pd.to_datetime(dd.ts60, format='ISO8601') - pd.Timestamp(0, tz='UTC')).dt.total_seconds() // 1
        dd = dd.assign(sec=sec).drop_duplicates('sec').set_index('sec')
        L = legendary(patch)
        cnt = lambda s: sum(int(x) in L for x in str(s or '').split(',') if x)
        for k, s in enumerate(((ts - pd.Timestamp(0, tz='UTC')).dt.total_seconds() // 1).to_numpy()):
            if s in dd.index:
                r = dd.loc[s]
                v = sum(cnt(r[f'p{i}_items']) for i in B) - sum(cnt(r[f'p{i}_items']) for i in R)
                empty = sum(not str(r[f'p{i}_items'] or '') for i in range(1, 11))
                if empty < 3:  # glitch frames have empty item lists
                    leg[k], bad[k] = v, 0
    legf = pd.Series(np.where(bad == 1, np.nan, leg)).ffill().fillna(0).to_numpy()
    d1 = np.where(bad == 1, 0, np.r_[0, np.diff(legf)])
    return pd.DataFrame(dict(leg=np.where(bad == 1, 0, leg), leg_d1=d1, it_bad=bad))


def one(args):
    from ingame_data import one as frames_one  # lazy: the live tracker runs a verbatim copy of this file (lollive src/v4feat.py)
    gid, patch = args
    o = frames_one(gid)
    if o is None:
        return None
    d = raw_frames(pd.read_parquet(f'{D}/frames60/{gid}.parquet'))
    assert len(d) == len(o), gid
    w = window_feats(d, o.clock.to_numpy())
    it = item_feats(gid, o.ts.reset_index(drop=True), patch)
    return pd.concat([o[['game_id', 'ts']].reset_index(drop=True), w, it], axis=1)


def main():
    t = pd.read_parquet(f'{D}/game_table.parquet', columns=['game_id', 'patch'])
    jobs = [(g, p) for g, p in zip(t.game_id, t.patch) if os.path.exists(f'{D}/frames60/{g}.parquet')]
    if len(sys.argv) > 1:
        jobs = jobs[:int(sys.argv[1])]
    with ProcessPoolExecutor(24) as ex:
        parts = [x for x in ex.map(one, jobs, chunksize=32) if x is not None]
    f = pd.concat(parts, ignore_index=True)
    os.makedirs(f'{D}/v4', exist_ok=True)
    f.to_parquet(f'{D}/v4/feat.parquet' if len(sys.argv) == 1 else f'{D}/v4/feat_sample.parquet')
    print(f.shape, f.game_id.nunique())
    pd.set_option('display.width', 250)
    print(f.drop(columns=['game_id', 'ts']).describe().round(3).T.to_string())


if __name__ == '__main__':
    main()
