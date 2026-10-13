"""Replay video: the broadcast VOD on top, and a live panel below with the model's win probability and the Polymarket price.

Inputs: data/viz/{gid}*.parquet/json from viz_game.py, and the VOD in data/vod/. Sync: VOD second v shows wall time
first_frame + v (the VOD's `firstFrameTime` from getEventDetails). This was checked against the broadcast timer and gold lead
at 10:00 and 30:00 (both within about 1 s of the feed). The panel shows only what was known at that wall time: the feed-state
model at the feed timestamp, and the last Polymarket trade.
Usage: python render_replay.py GAME_ID VOD.mp4 FIRST_FRAME_ISO OUT.mp4 [--from V0 --to V1] [--speed K]
"""
import argparse, json, os, shutil, subprocess, tempfile
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = '../data'
FPS = 4                     # panel frames per second of video time (per output second when speed > 1)
W, H = 1280, 360
SURF, INK, INK2, MUTED, GRID = '#1a1a19', '#ffffff', '#c3c2b7', '#8a897f', '#383835'
MODEL, MARKET = '#3987e5', '#d95926'   # reference palette, dark steps, slots 1-2
EV_GLYPH = {'tower': 'T', 'inhib': 'I', 'drake': 'D', 'baron': 'B', 'fight': 'F'}


def load(gid):
    s = pd.read_parquet(f'{D}/viz/{gid}.parquet')
    ev = pd.read_parquet(f'{D}/viz/{gid}_events.parquet')
    info = json.load(open(f'{D}/viz/{gid}.json'))
    t = s.t.astype('int64').to_numpy() / 1e6
    ig = s.clock.notna().to_numpy()
    tg0 = (t[ig] - s.clock.to_numpy()[ig])[0]  # wall time of game clock 0
    x = np.where(ig, s.clock / 60, (t - tg0) / 60)
    p_model = np.where(ig, s.p_model, np.nan)
    return dict(t=t, x=x, pm=p_model, pk=s.p_last.to_numpy(), vol=s.usd_1s.to_numpy(), ig=ig, s=s,
                ev=ev.assign(x=ev.clock / 60), info=info, tg0=tg0)


def pct(p):
    return '–' if not np.isfinite(p) else f'{100 * p:.0f}%'


def render_chunk(args):
    gid, times, idx0, outdir = args
    G = load(gid)
    I = G['info']
    blue, red = I['blue'], I['red']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': INK, 'axes.labelcolor': INK2,
                         'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.edgecolor': GRID})
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=SURF)
    ax = fig.add_axes([0.05, 0.32, 0.625, 0.58], facecolor=SURF)
    axv = fig.add_axes([0.05, 0.15, 0.625, 0.12], facecolor=SURF, sharex=ax)
    xmin, xmax = G['x'][0], G['x'][-1] + 0.3
    for a in (ax, axv):
        a.set_xlim(xmin, xmax)
        for sp in ('top', 'right'):
            a.spines[sp].set_visible(False)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.text(0.005, 0.99, f'▲ {blue} favoured', transform=ax.transAxes, fontsize=8, color=MUTED, va='top')
    ax.text(0.005, 0.01, f'▼ {red} favoured', transform=ax.transAxes, fontsize=8, color=MUTED, va='bottom')
    ax.axhline(.5, color=GRID, lw=1, zorder=0)
    ax.axvline(0, color=GRID, lw=1, ls=':', zorder=0)
    ax.text(0.15, 0.10, 'game start', color=MUTED, fontsize=8, transform=ax.get_xaxis_transform())
    ax.grid(axis='y', color=GRID, lw=0.5, alpha=.6)
    ax.tick_params(labelbottom=False, length=0)
    ax.set_title(f'P({blue} wins)', loc='left', color=INK2, fontsize=10, pad=4)
    axv.set_yticks([])
    axv.spines['left'].set_visible(False)
    axv.set_xlabel('game minute (feed clock)', fontsize=9, color=MUTED, labelpad=2)
    axv.text(0.0, 1.0, 'Polymarket $ traded / 10 s', transform=axv.transAxes, fontsize=8, color=MUTED, va='bottom')
    axv.text(1.0, 1.0, 'events: T tower  I inhib  D drake  B baron  F fight (top row GEN, bottom KT)'.replace('GEN', blue).replace('KT', red),
             transform=axv.transAxes, fontsize=8, color=MUTED, va='bottom', ha='right')
    vol10 = pd.Series(G['vol']).rolling(10, min_periods=1).sum().to_numpy()
    vmax = np.nanmax(vol10)
    axv.set_ylim(0, vmax * 1.05)
    pre = G['x'] < 0
    ax.plot([xmin, 0], [I['prior']] * 2, color=MODEL, lw=1.5, ls=(0, (3, 3)), alpha=.8, zorder=2)
    lm, = ax.plot([], [], color=MODEL, lw=2, solid_capstyle='round', zorder=3)
    lk, = ax.step([], [], where='post', color=MARKET, lw=2, zorder=2)
    dm, = ax.plot([], [], 'o', ms=8, color=MODEL, mec=SURF, mew=2, zorder=5)
    dk, = ax.plot([], [], 'o', ms=8, color=MARKET, mec=SURF, mew=2, zorder=5)
    tm = ax.text(0, 0, '', color=INK, fontsize=9, va='center', zorder=6)
    tk = ax.text(0, 0, '', color=INK, fontsize=9, va='center', zorder=6)
    now = ax.axvline(xmin, color=MUTED, lw=1, alpha=.7)
    vfill = [None]
    ev_art = []
    for e in G['ev'].itertuples():
        y = 0.965 if e.s > 0 else 0.035
        ev_art.append((e.tau, ax.text(e.x, y, EV_GLYPH[e.kind], ha='center', va='center', fontsize=7, color=INK2,
                                      bbox=dict(boxstyle='round,pad=0.15', fc=GRID, ec='none'), visible=False, zorder=4)))
    # right-hand readout
    fig.text(0.70, 0.93, f"{I['blue']} vs {I['red']} · Worlds 2025 SF · game 1", fontsize=12, color=INK, weight='bold')
    fig.text(0.70, 0.86, f"Polymarket {I['market']} · ${I['vol'] / 1e6:.1f}M traded", fontsize=8.5, color=MUTED)
    fig.text(0.70, 0.72, f'MODEL  P({blue})', fontsize=9, color=INK2)
    fig.text(0.855, 0.72, f'POLYMARKET  P({blue})', fontsize=9, color=INK2)
    fig.patches.append(plt.Rectangle((0.70, 0.555), 0.006, 0.13, transform=fig.transFigure, color=MODEL))
    fig.patches.append(plt.Rectangle((0.855, 0.555), 0.006, 0.13, transform=fig.transFigure, color=MARKET))
    hm = fig.text(0.713, 0.56, '', fontsize=30, color=INK, weight='bold')
    hk = fig.text(0.868, 0.56, '', fontsize=30, color=INK, weight='bold')
    gap = fig.text(0.70, 0.46, '', fontsize=10.5, color=INK2)
    st = fig.text(0.70, 0.17, '', fontsize=10, color=INK2, family='DejaVu Sans Mono', linespacing=1.5)
    fig.text(0.70, 0.025, 'Model: logistic on live feed state + pre-game prior, trained on\n2024-04..2025-06 only (this game is out of sample).',
             fontsize=7.5, color=MUTED, linespacing=1.3)
    t, x, S = G['t'], G['x'], G['s']
    for j, tw in enumerate(times):
        i = int(np.searchsorted(t, tw, side='right')) - 1
        if i < 0:
            i = 0
        xs = x[:i + 1]
        lm.set_data(xs, G['pm'][:i + 1]); lk.set_data(xs, G['pk'][:i + 1])
        pmn, pkn = G['pm'][i], G['pk'][i]
        dm.set_data([x[i]], [pmn] if np.isfinite(pmn) else [np.nan]); dk.set_data([x[i]], [pkn])
        now.set_xdata([x[i], x[i]])
        # direct labels beside the live dots, pushed apart when close
        ym, yk = (pmn if np.isfinite(pmn) else np.nan), pkn
        if np.isfinite(ym) and abs(ym - yk) < .07:
            mid = (ym + yk) / 2; sgn = 1 if ym >= yk else -1
            ym, yk = mid + sgn * .035, mid - sgn * .035
        tm.set_position((x[i] + .4, ym)); tm.set_text('model' if np.isfinite(ym) else '')
        tk.set_position((x[i] + .4, yk)); tk.set_text('market')
        if vfill[0] is not None:
            vfill[0].remove()
        vfill[0] = axv.fill_between(xs, 0, vol10[:i + 1], color=MARKET, alpha=.55, lw=0, step='post')
        for tau, a in ev_art:
            a.set_visible(tau <= tw)
        hm.set_text(pct(pmn) if G['ig'][i] else f"{pct(G['info']['prior'])}*")
        hk.set_text(pct(pkn))
        if G['ig'][i]:
            d = 100 * (pmn - pkn)
            side = blue if d > 0 else red
            gap.set_text(f'model − market: {d:+.0f} pp  (model leans {side})' if abs(d) >= 1 else 'model − market: ~0 pp')
            r = S.iloc[i]
            st.set_text(f"{blue} − {red}\n"
                        f"gold   {r.gold:+7,.0f}   kills  {r.kills:+3.0f}\n"
                        f"towers {r.towers:+7.0f}   drakes {r.drakes:+3.0f}\n"
                        f"barons {r.barons:+7.0f}   alive  {r.alive:+3.0f}")
        else:
            gap.set_text('* pre-game prior (draft + team/player ratings)')
            st.set_text('')
        fig.savefig(f'{outdir}/{idx0 + j:06d}.png', facecolor=SURF)
    plt.close(fig)
    return len(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('gid'); ap.add_argument('vod'); ap.add_argument('first_frame'); ap.add_argument('out')
    ap.add_argument('--from', dest='v0', type=float, default=0); ap.add_argument('--to', dest='v1', type=float, default=None)
    ap.add_argument('--speed', type=float, default=1.0)
    ap.add_argument('--workers', type=int, default=44)
    a = ap.parse_args()
    dur = float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', a.vod]))
    v1 = min(a.v1 or dur, dur)
    f0 = pd.Timestamp(a.first_frame).value / 1e9
    vt = np.arange(a.v0, v1, a.speed / FPS)  # video seconds of each panel frame
    tmp = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(a.out)))
    chunks = np.array_split(np.arange(len(vt)), a.workers)
    jobs = [(a.gid, f0 + vt[c], int(c[0]), tmp) for c in chunks if len(c)]
    with ProcessPoolExecutor(a.workers) as ex:
        n = sum(ex.map(render_chunk, jobs))
    print('panel frames', n)
    vf = f'[0:v]setpts=(PTS-STARTPTS)/{a.speed},fps=30,scale={W}:720[top];' \
         f'[1:v]fps=30,format=yuv420p[bot];[top][bot]vstack=inputs=2[v]'
    cmd = ['ffmpeg', '-v', 'error', '-y', '-ss', str(a.v0), '-t', str(v1 - a.v0), '-i', a.vod,
           '-framerate', str(FPS), '-i', f'{tmp}/%06d.png', '-filter_complex', vf, '-map', '[v]']
    if a.speed == 1:
        cmd += ['-map', '0:a', '-c:a', 'aac', '-b:a', '128k']
    cmd += ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '24', '-shortest', '-movflags', '+faststart', a.out]
    subprocess.run(cmd, check=True)
    shutil.rmtree(tmp)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
