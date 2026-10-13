"""v3 step 2: drake / baron / elder status at every 60 s frame, from the feed alone (no vision needed).

Spawn rules, checked on 1,500 games' first kill times (patch major 14 = 2024, 15 = 2025, 16 = 2026):
  drake: first at 5:00, respawns 5:00 after a kill. Once a team has 4 elemental drakes (soul), only elder spawns,
         6:00 after the soul drake, and respawns 6:00 after each elder kill.
  baron: first at 20:00 (2024, 2026) or 25:00 (2025, the Atakhan season; first baron kill 1% quantile 25:30), respawns 6:00 after a kill.
The feed gives cumulative drake lists and baron counts, so whether an objective is *up* at a frame is exact up to the
spawn time of the next one: a kill between two 60 s frames is put at their midpoint (±30 s). The live tracker sees kills
at feed resolution, so its timers are exact.

Features (blue minus red where sided):
  drake_up, elder_up, baron_up      1 if the monster is alive at this frame
  t_drake, t_baron                  seconds until it spawns (0 while up), capped at 300 / 360, scaled to [0, 1]
  soul_pt                           (blue has 3 drakes) - (red has 3 drakes), soul not yet taken
  soul_up                           drake_up * soul_pt: the next drake decides soul
  baron_lead_up, elder_lead_up      objective up x gold lead sign is left to the model (gold x up interactions in v3_model)
Writes data/v3/objectives.parquet keyed (game_id, ts).
"""
import glob, os, sys
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingame_data import one as frames_one

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data')
DRAKE0, DRAKE_RE, ELDER_RE, BARON_RE = 300, 300, 360, 360


def baron0(patch):
    return 1500 if str(patch).split('.')[0] in ('15', '25') else 1200


def timeline(clock, nb_drk, nr_drk, n_eld, n_bar, patch):
    """clock and cumulative counts per frame -> per-frame objective state (lists of dicts).

    Kills are dated at the midpoint between the frame that first shows them and the one before."""
    out = []
    next_drake, next_baron, elder_phase = DRAKE0, baron0(patch), False
    prev = (0, 0, 0, 0)
    for k, c in enumerate(clock):
        cur = (nb_drk[k], nr_drk[k], n_eld[k], n_bar[k])
        if k:
            tk = (clock[k - 1] + c) / 2
            if cur[0] + cur[1] > prev[0] + prev[1]:
                if max(cur[0], cur[1]) >= 4:
                    elder_phase = True
                    next_drake = tk + ELDER_RE
                else:
                    next_drake = tk + DRAKE_RE
            if cur[2] > prev[2]:
                next_drake = tk + ELDER_RE
                elder_phase = True
            if cur[3] > prev[3]:
                next_baron = tk + BARON_RE
        prev = cur
        up_d = c >= next_drake
        up_b = c >= next_baron
        soul_pt = 0 if elder_phase else int(cur[0] == 3) - int(cur[1] == 3)
        out.append(dict(drake_up=int(up_d and not elder_phase), elder_up=int(up_d and elder_phase), baron_up=int(up_b),
                        t_drake=min(max(next_drake - c, 0), 360) / 360, t_baron=min(max(next_baron - c, 0), 360) / 360,
                        soul_pt=soul_pt, soul_up=soul_pt * int(up_d and not elder_phase)))
    return out


def counts(d):
    lb = d.b_drakes.fillna('').str.split(',').map(lambda l: [x for x in l if x])
    lr = d.r_drakes.fillna('').str.split(',').map(lambda l: [x for x in l if x])
    nb = lb.map(lambda l: sum(x != 'elder' for x in l)).to_numpy()
    nr = lr.map(lambda l: sum(x != 'elder' for x in l)).to_numpy()
    ne = (lb.map(lambda l: l.count('elder')) + lr.map(lambda l: l.count('elder'))).to_numpy()
    return nb, nr, ne, (d.b_barons + d.r_barons).to_numpy()


def one(args):
    gid, patch = args
    o = frames_one(gid)
    if o is None:
        return None
    d = pd.read_parquet(f'{D}/frames60/{gid}.parquet')
    d = d[d.state.isin(['in_game', 'paused', 'finished'])]
    d = d.assign(_t=pd.to_datetime(d.ts, format='ISO8601')).sort_values('_t').drop_duplicates('_t')
    d = d[(d.b_gold + d.r_gold) > 0].reset_index(drop=True)
    tl = timeline(o.clock.to_numpy(), *counts(d), patch)
    return pd.concat([o[['game_id', 'ts']].reset_index(drop=True), pd.DataFrame(tl)], axis=1)


def main():
    t = pd.read_parquet(f'{D}/game_table.parquet', columns=['game_id', 'patch'])
    jobs = [(g, p) for g, p in zip(t.game_id, t.patch) if os.path.exists(f'{D}/frames60/{g}.parquet')]
    with ProcessPoolExecutor(16) as ex:
        parts = [x for x in ex.map(one, jobs, chunksize=64) if x is not None]
    f = pd.concat(parts, ignore_index=True)
    os.makedirs(f'{D}/v3', exist_ok=True)
    f.to_parquet(f'{D}/v3/objectives.parquet')
    print(f.shape, f.game_id.nunique())
    print(f.drop(columns=['game_id', 'ts']).describe().round(3).T.to_string())


if __name__ == '__main__':
    main()
